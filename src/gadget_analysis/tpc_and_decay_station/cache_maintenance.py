'''
Budgeted maintenance of an experiment's derived-data caches under ddas_interface.get_cache_dir(experiment)
(/egr/research-tpc/shared/proc_runs/<exp>/analysis_cache by default): clarion_cache (add-back trees, add-back
histograms, gamma-gamma matrices, timing spectra) and hist_cache (ddas_interface and calibrated histograms). Everything in them is rebuilt on demand, so the policy is simply "keep the most recently
read files":

- the runs module may declare CACHE_MAX_GB (and CACHE_KEEP_DAYS, default 7); when a library function writes a new
  cache file it calls maybe_prune(experiment), which deletes the least recently read files until the caches fit
  the budget, never touching a file read within CACHE_KEEP_DAYS. Without CACHE_MAX_GB nothing is ever deleted.
- "last read" is the file's access time (the share is mounted relatime, and every cache hit also calls touch()),
  or its modification time if that is newer.
- deletions are appended to cache_prune.log in the cache root.
- the share is NFS without ACL inheritance, so make_shared_dir() and share_file() give what the library creates
  there the permission bits that let every user of the experiment write, replace and prune it (README, Caches).

Command line (dry run unless --apply):
    python -m gadget_analysis.tpc_and_decay_station.cache_maintenance e23035 [--max-gb 150] [--keep-days 7] [--apply]
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

# Permission bits for what the library creates in the shared data tree. The share is NFSv4 without ACL inheritance:
# the parents are setgid (new entries keep the group) but umask 022 would leave every new folder rwxr-xr-x, owned by
# whoever ran first, and nobody else could write there afterwards. Directories get rwx for everyone plus setgid, never
# the sticky bit (pruning has to delete other users' files); cache files get rw for everyone so that any user's cache
# hit can touch() them.
SHARED_DIR_MODE = 0o2777
SHARED_FILE_MODE = 0o666

def make_shared_dir(path, mode=SHARED_DIR_MODE):
    '''
    os.makedirs(path, exist_ok=True) for a folder in the shared data tree, then chmod the components this call created
    to mode (they belong to the caller, so the chmod cannot be refused, and umask does not apply to chmod). Folders that
    already exist are left as they are: only their owner can change them. Returns path as given.
    '''
    missing = []
    p = os.path.abspath(path)
    while not os.path.isdir(p):
        missing.append(p)
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    os.makedirs(path, exist_ok=True)
    for d in reversed(missing):
        try:
            os.chmod(d, mode)
        except OSError:
            pass                      # created by someone else between the check and the makedirs: theirs to set
    return path

def share_file(path, mode=SHARED_FILE_MODE):
    '''chmod a file this process has just written (cache file, log) so that other users can touch() and append to it; a file owned by someone else is left alone.'''
    try:
        os.chmod(path, mode)
    except OSError:
        pass

def _cache_root(experiment, exp_dir=None):
    '''The cache root (exp_dir overrides it, for tests).'''
    if exp_dir is not None:
        return str(exp_dir)
    from gadget_analysis.tpc_and_decay_station import ddas_interface   # lazy: ddas_interface imports this module
    return str(ddas_interface.get_cache_dir(experiment))

def touch(path):
    '''
    Record a cache hit: set the access time to now. The file's owner keeps its modification time; anyone else may only
    set both times to now (explicit times need ownership), which the eviction policy treats the same.
    '''
    try:
        st = os.stat(path)
        try:
            os.utime(path, (time.time(), st.st_mtime))
        except PermissionError:
            os.utime(path, None)
    except OSError:
        pass

def cache_files(experiment, exp_dir=None):
    '''[(path, size_bytes, last_read_epoch), ...] of every file in the experiment's caches (clarion_cache, hist_cache under the cache root).'''
    root = _cache_root(experiment, exp_dir)
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
    root = _cache_root(experiment, exp_dir)
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
    root = _cache_root(experiment, exp_dir)
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
            log_path = os.path.join(root, LOG_NAME)
            is_new = not os.path.exists(log_path)
            with open(log_path, 'a') as log:
                stamp = datetime.datetime.now().isoformat(timespec='seconds')
                log.write(f'{stamp} prune: total {total/1e9:.1f} GB, budget {max_gb} GB, keep {keep_days} d, '
                          f'deleted {len(removed)} files, {sum(s for _, s, _ in removed)/1e9:.1f} GB\n')
                for path, size, days in removed:
                    log.write(f'{stamp}   {size/1e9:.2f} GB last read {days:.1f} d ago {os.path.relpath(path, root)}\n')
            if is_new:
                share_file(log_path)
        except OSError:
            pass
        deleted = removed
        freed = sum(s for _, s, _ in removed)
    return deleted, freed, total, total - (freed if not dry_run else 0)

def _write_stamp(stamp):
    '''
    Set the stamp's time to now by replacing the file: os.replace needs only write access to the folder, so any user can
    take the prune turn whoever wrote the previous stamp (appending to it would need to own that file).
    '''
    tmp = f'{stamp}.{os.getpid()}.tmp'
    with open(tmp, 'w'):
        pass
    share_file(tmp)
    os.replace(tmp, stamp)

def maybe_prune(experiment, min_interval_s=600):
    '''
    Called by the library after it writes a cache file. Prunes to the runs module's CACHE_MAX_GB (nothing happens
    when it is not declared), at most once per min_interval_s across all processes (a stamp file in the experiment
    folder), and never lets a maintenance problem break the analysis call.
    '''
    try:
        from gadget_analysis.tpc_and_decay_station import ddas_interface
        max_gb = ddas_interface.get_experiment_setting(experiment, 'CACHE_MAX_GB', None)
        if max_gb is None:
            return
        keep_days = ddas_interface.get_experiment_setting(experiment, 'CACHE_KEEP_DAYS', 7)
        root = str(ddas_interface.get_cache_dir(experiment))
        stamp = os.path.join(root, STAMP_NAME)
        try:
            if time.time() - os.stat(stamp).st_mtime < min_interval_s:
                return
        except OSError:
            pass
        _write_stamp(stamp)
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
    from gadget_analysis.tpc_and_decay_station import ddas_interface
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
