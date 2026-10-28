'''
Budgeted maintenance of an experiment's derived-data caches: <exp>_analysis/clarion_cache (add-back trees,
add-back histograms, gamma-gamma matrices, timing spectra) and <exp>_analysis/hist_cache (ddas_interface and
calibrated histograms). Everything in them is rebuilt on demand, so the policy is simply "keep the most recently
read files":

- the runs module may declare CACHE_MAX_GB (and CACHE_KEEP_DAYS, default 7); when a library function writes a new
  cache file it calls maybe_prune(experiment), which deletes the least recently read files until the caches fit
  the budget, never touching a file read within CACHE_KEEP_DAYS. Without CACHE_MAX_GB nothing is ever deleted.
- "last read" is the file's access time (the share is mounted relatime, and every cache hit also calls touch()),
  or its modification time if that is newer.
- deletions are appended to <exp>_analysis/cache_prune.log.

Command line (dry run unless --apply):
    python -m tpc_and_decay_station.cache_maintenance e23035 [--max-gb 150] [--keep-days 7] [--apply]
'''
import os
import sys
import time
import datetime
import argparse

CACHE_SUBDIRS = ('clarion_cache', 'hist_cache')
LOG_NAME = 'cache_prune.log'
STAMP_NAME = 'cache_prune.stamp'
_SKIP = {LOG_NAME, STAMP_NAME}

def _experiment_dir(experiment, exp_dir=None):
    if exp_dir is not None:
        return str(exp_dir)
    from tpc_and_decay_station import ddas_interface   # lazy: ddas_interface imports this module
    return str(ddas_interface.get_experiment_dir(experiment))

def touch(path):
    '''Record a cache hit: set the access time to now, keeping the modification time.'''
    try:
        st = os.stat(path)
        os.utime(path, (time.time(), st.st_mtime))
    except OSError:
        pass

def cache_files(experiment, exp_dir=None):
    '''[(path, size_bytes, last_read_epoch), ...] of every file in the experiment's caches.'''
    root = _experiment_dir(experiment, exp_dir)
    out = []
    for sub in CACHE_SUBDIRS:
        top = os.path.join(root, sub)
        if not os.path.isdir(top):
            continue
        stack = [top]
        while stack:
            d = stack.pop()
            try:
                entries = list(os.scandir(d))
            except OSError:
                continue
            for e in entries:
                if e.is_dir(follow_symlinks=False):
                    stack.append(e.path)
                elif e.is_file(follow_symlinks=False) and e.name not in _SKIP:
                    try:
                        st = e.stat()
                    except OSError:
                        continue
                    out.append((e.path, st.st_size, max(st.st_atime, st.st_mtime)))
    return out

def cache_report(experiment, exp_dir=None):
    '''Size, file count and last-read range per cache folder, as a printable string.'''
    root = _experiment_dir(experiment, exp_dir)
    files = cache_files(experiment, exp_dir)
    now = time.time()
    groups = {}
    for path, size, last in files:
        rel = os.path.relpath(path, root)
        parts = rel.split(os.sep)
        key = os.sep.join(parts[:2]) if parts[0] == 'clarion_cache' and len(parts) > 2 else parts[0]
        g = groups.setdefault(key, [0, 0, None, None])
        g[0] += size; g[1] += 1
        g[2] = last if g[2] is None else min(g[2], last)
        g[3] = last if g[3] is None else max(g[3], last)
    lines = [f'caches of {experiment} under {root}: {sum(f[1] for f in files)/1e9:.1f} GB in {len(files)} files']
    for key, (size, n, oldest, newest) in sorted(groups.items(), key=lambda kv: -kv[1][0]):
        lines.append(f'  {key:32s} {size/1e9:7.1f} GB {n:6d} files   last read {(now-newest)/86400:5.1f} to {(now-oldest)/86400:5.1f} days ago')
    return '\n'.join(lines)

def prune_cache(experiment, max_gb, keep_days=7, dry_run=True, exp_dir=None, verbose=True):
    '''
    Delete the least recently read cache files until the caches of the experiment fit max_gb, skipping files read
    in the last keep_days. Returns (deleted, freed_bytes, total_before, total_after) with deleted = [(path, size,
    days_since_read), ...]. With dry_run the list is computed and printed but nothing is removed.
    '''
    root = _experiment_dir(experiment, exp_dir)
    files = cache_files(experiment, exp_dir)
    now = time.time()
    total = sum(size for _, size, _ in files)
    budget = float(max_gb) * 1e9
    deleted, freed = [], 0
    if total > budget:
        for path, size, last in sorted(files, key=lambda f: f[2]):   # oldest read first
            if total - freed <= budget:
                break
            if now - last < keep_days * 86400:
                continue
            deleted.append((path, size, (now - last) / 86400))
            freed += size
    if verbose:
        verb = 'would delete' if dry_run else 'deleting'
        print(f'{experiment} caches: {total/1e9:.1f} GB, budget {max_gb} GB, keep files read in the last {keep_days} days: '
              f'{verb} {len(deleted)} files, {freed/1e9:.1f} GB', flush=True)
        for path, size, days in deleted:
            print(f'  {verb} {size/1e9:6.2f} GB  last read {days:5.1f} d ago  {os.path.relpath(path, root)}', flush=True)
    if not dry_run and deleted:
        removed = []
        for path, size, days in deleted:
            try:
                os.remove(path)
                removed.append((path, size, days))
            except FileNotFoundError:
                pass                      # another process got there first
            except OSError as e:
                print(f'cache_maintenance: could not delete {path}: {e}', file=sys.stderr, flush=True)
        try:
            with open(os.path.join(root, LOG_NAME), 'a') as log:
                stamp = datetime.datetime.now().isoformat(timespec='seconds')
                log.write(f'{stamp} prune: total {total/1e9:.1f} GB, budget {max_gb} GB, keep {keep_days} d, '
                          f'deleted {len(removed)} files, {sum(s for _, s, _ in removed)/1e9:.1f} GB\n')
                for path, size, days in removed:
                    log.write(f'{stamp}   {size/1e9:.2f} GB last read {days:.1f} d ago {os.path.relpath(path, root)}\n')
        except OSError:
            pass
        deleted = removed
        freed = sum(s for _, s, _ in removed)
    return deleted, freed, total, total - (freed if not dry_run else 0)

def maybe_prune(experiment, min_interval_s=600):
    '''
    Called by the library after it writes a cache file. Prunes to the runs module's CACHE_MAX_GB (nothing happens
    when it is not declared), at most once per min_interval_s across all processes (a stamp file in the experiment
    folder), and never lets a maintenance problem break the analysis call.
    '''
    try:
        from tpc_and_decay_station import ddas_interface
        max_gb = ddas_interface.get_experiment_setting(experiment, 'CACHE_MAX_GB', None)
        if max_gb is None:
            return
        keep_days = ddas_interface.get_experiment_setting(experiment, 'CACHE_KEEP_DAYS', 7)
        root = str(ddas_interface.get_experiment_dir(experiment))
        stamp = os.path.join(root, STAMP_NAME)
        try:
            if time.time() - os.stat(stamp).st_mtime < min_interval_s:
                return
        except OSError:
            pass
        with open(stamp, 'a'):
            os.utime(stamp, None)
        prune_cache(experiment, max_gb, keep_days=keep_days, dry_run=False, verbose=False)
    except Exception as e:                    # pragma: no cover
        print(f'cache_maintenance: skipped ({e!r})', file=sys.stderr, flush=True)

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('experiment')
    ap.add_argument('--max-gb', type=float, default=None, help="budget; default: the runs module's CACHE_MAX_GB")
    ap.add_argument('--keep-days', type=float, default=None, help="never delete files read this recently; default: the runs module's CACHE_KEEP_DAYS or 7")
    ap.add_argument('--apply', action='store_true', help='delete (default is a dry run)')
    args = ap.parse_args(argv)
    from tpc_and_decay_station import ddas_interface
    print(cache_report(args.experiment))
    max_gb = args.max_gb if args.max_gb is not None else ddas_interface.get_experiment_setting(args.experiment, 'CACHE_MAX_GB', None)
    if max_gb is None:
        print(f'no budget: pass --max-gb or declare CACHE_MAX_GB in {args.experiment}_runs.py')
        return 0
    keep_days = args.keep_days if args.keep_days is not None else ddas_interface.get_experiment_setting(args.experiment, 'CACHE_KEEP_DAYS', 7)
    prune_cache(args.experiment, max_gb, keep_days=keep_days, dry_run=not args.apply)
    return 0

if __name__ == '__main__':
    sys.exit(main())
