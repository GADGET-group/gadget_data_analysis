import os
import re
import pickle


import numpy as np
import scipy.spatial.distance
import scipy.optimize as opt
import scipy.interpolate as interpolate
import h5py
import matplotlib.pylab as plt
import matplotlib.colors as colors
import matplotlib.cm 
from matplotlib.colors import LinearSegmentedColormap
import tqdm
from sklearn import datasets, linear_model
from gadget_analysis.tpc_data_processing.dev import railed_pad_repair
import scipy.signal
import numba

try:
    import cupy as cp
    import cupyx.scipy.special as cpspecial
    import torch
    from torch_ransac3d.line import line_fit
    USE_GPU = True
except Exception as e:
    cp = np
    import scipy.special as cpspecial
    cp.asnumpy = lambda x: x
    USE_GPU = False


import skimage.measure

VETO_PADS = (253, 254, 508, 509, 763, 764, 1018, 1019)
ZERO_PAD_PROBE_EVENTS = 20 #events probed per file to detect zero padding (remove_zero_pad)
#baseline parameters that can be given a veto-pad specific value with a 'veto_' prefixed key (e.g.
#veto_smart_bins_away_to_check); used only when veto_background_subtract_mode is set
VETO_BASELINE_PARAMS = ('smart_bins_away_to_check', 'num_smart_background_ave_bins', 'ic_counts_threshold',
                        'smart2_min_bins_in_peak', 'smart2_min_sigma', 'require_peak_within',
                        'smart_walk_threshold', 'smart_one_sided_baseline',
                        'num_background_bins', 'background_convolution_kernel')

class _BaselineParams:
    '''
    Read-only view of the baseline settings of a raw_h5_file. With veto=True, an attribute X is taken from
    'veto_X' when that is set (not None), e.g. background_subtract_mode -> veto_background_subtract_mode;
    otherwise (and always with veto=False) from X itself.

    None therefore means "not set for the veto pads", never "use the default": a veto_X of None falls through to
    X, whatever X is set to. In particular veto_smart_walk_threshold = None does not ask for the ic threshold on
    the veto pads once smart_walk_threshold is set; to walk the veto pads with the ic threshold, set
    veto_smart_walk_threshold to that threshold's value (veto_ic_counts_threshold or ic_counts_threshold).
    '''
    def __init__(self, h5, veto):
        self._h5 = h5
        self._veto = veto
    def __getattr__(self, name):
        if self._veto:
            value = getattr(self._h5, 'veto_' + name, None)
            if value is not None:
                return value
        return getattr(self._h5, name)
FIRST_DATA_BIN = 6 #first time bin is dumped, because it is junk
NUM_PADS = 1024

#instance attributes that hold cached results or properties of the open file rather than settings: never written by
#save_config_file and ignored by load_config (configs saved before 2026-10-28 contain them)
_CONFIG_STATE_KEYS = frozenset(('cached_event', 'cached_event_xyte', 'cached_event_xyze', 'cached_data', 'cached_xyte',
                                'cached_xyze', 'batched_data_cache', 'num_time_bins'))

def _literal(s):
    '''
    ast.literal_eval of the str() of a list/tuple of numbers, also accepting what literal_eval rejects: inf and -inf
    (how str() writes infinite floats, e.g. the default require_peak_within) and numpy scalar reprs such as
    np.float64(1.5) (numpy >= 2).
    '''
    import ast
    s = re.sub(r'\bnp\.(?:float|int|uint|bool)[0-9_]*\(([^()]*)\)', r'\1', s)
    s = re.sub(r'(?<![\w.])inf\b', '1e999', s) #1e999 is parsed as float inf
    return ast.literal_eval(s)

_M32 = 0xFFFFFFFF

def _mul32(a, c):
    '''(a * c) mod 2^32 for an int64 tensor a with values in [0, 2^32) and a constant c < 2^32, without int64 overflow'''
    return (a * (c & 0xFFFF) + (((a * (c >> 16)) & 0xFFFF) << 16)) & _M32

def _hash32(x):
    '''lowbias32 integer hash (C. Wellons) of an int64 tensor with values in [0, 2^32)'''
    x = x ^ (x >> 16)
    x = _mul32(x, 0x7feb352d)
    x = x ^ (x >> 15)
    x = _mul32(x, 0x846ca68b)
    return x ^ (x >> 16)

def _ransac_sample_indices(traces, max_iterations, device):
    '''
    Sample indices (B, max_iterations, 2) in [0, N) for batched_line_fit_2d, instead of torch.randint: a hash of each
    trace's values (float64 bit patterns and positions) and of the iteration. A trace gets the same samples in any
    batch, so the processing is deterministic and does not depend on how a run is batched or split over workers.
    '''
    B, N = traces.shape
    bits = torch.from_numpy(np.ascontiguousarray(traces, dtype=np.float64)).view(torch.int64).to(device)
    pos = _hash32((torch.arange(N, dtype=torch.int64, device=device) + 0x9E3779B9) & _M32)
    key = _hash32(_hash32((bits & _M32) ^ pos) ^ ((bits >> 32) & _M32)).sum(dim=1) & _M32
    key = _hash32(key)
    draw = _hash32((torch.arange(2 * max_iterations, dtype=torch.int64, device=device) + 0x85EBCA6B) & _M32)
    return (_hash32(key[:, None] ^ draw[None, :]) % N).view(B, max_iterations, 2)

def batched_line_fit_2d(traces, device=None):
    # traces: numpy array (B, 512). device: torch device, default cuda, resolved here rather than in the signature so
    # that the module imports without torch (calculate_background_2d then uses the sklearn RANSAC)
    if device is None:
        device = torch.device('cuda')
    if traces.shape[0] == 0:
        return np.empty((0, traces.shape[1]), dtype=np.float32)

    traces_torch = torch.tensor(traces, dtype=torch.float32, device=device)
    B, N = traces_torch.shape

    # calculate mad_thresh for each batch
    # median across dim 1
    medians = traces_torch.median(dim=1, keepdim=True).values
    abs_dev = torch.abs(traces_torch - medians)
    mad_thresh = abs_dev.median(dim=1).values
    mad_thresh = torch.clamp(mad_thresh, min=1.0) # fallback

    max_iterations = 100
    idx = _ransac_sample_indices(traces, max_iterations, device)
    
    x_coords = torch.arange(N, dtype=torch.float32, device=device).unsqueeze(0).expand(B, N)
    x_coords_expanded = x_coords.unsqueeze(1).expand(B, max_iterations, N)
    traces_expanded = traces_torch.unsqueeze(1).expand(B, max_iterations, N)
    
    x_sampled = torch.gather(x_coords_expanded, 2, idx)
    y_sampled = torch.gather(traces_expanded, 2, idx)
    
    x1, x2 = x_sampled[..., 0], x_sampled[..., 1]
    y1, y2 = y_sampled[..., 0], y_sampled[..., 1]
    
    A = y1 - y2
    B_coef = x2 - x1
    C = x1 * y2 - x2 * y1
    
    norm = torch.sqrt(A**2 + B_coef**2) + 1e-8
    A = A / norm
    B_coef = B_coef / norm
    C = C / norm
    
    dist = torch.abs(
        A.unsqueeze(2) * x_coords.unsqueeze(1) + 
        B_coef.unsqueeze(2) * traces_torch.unsqueeze(1) + 
        C.unsqueeze(2)
    )
    
    inliers = dist <= mad_thresh.view(B, 1, 1)
    inlier_counts = inliers.sum(dim=2)
    
    # Invalidate iterations where x1 == x2 (which yields A=0, B=0, C=0)
    valid_mask = (x1 != x2)
    inlier_counts[~valid_mask] = -1
    
    best_iters = inlier_counts.argmax(dim=1)
    
    b_idx = torch.arange(B, device=device)
    best_A = A[b_idx, best_iters]
    best_B = B_coef[b_idx, best_iters]
    best_C = C[b_idx, best_iters]
    
    baselines = -(best_A.unsqueeze(1) * x_coords + best_C.unsqueeze(1)) / (best_B.unsqueeze(1) + 1e-8)
    return baselines.cpu().numpy()

@numba.jit(nopython=True, cache=True, parallel=True)
def _smart2_numba_peak_trimming_2d(traces, ransac_baselines, min_bins_in_peak, bins_away, min_sigma):
    B, N = traces.shape
    baselines = np.empty_like(traces)
    
    for b in numba.prange(B):
        trace = traces[b]
        ransac_baseline = ransac_baselines[b]
        n = N
        
        # 1. Find connected components (starts and ends of peaks)
        starts = []
        ends = []
        in_peak = False
        for i in range(n):
            if trace[i] > ransac_baseline[i]:
                if not in_peak:
                    starts.append(i)
                    in_peak = True
            else:
                if in_peak:
                    ends.append(i)
                    in_peak = False
        if in_peak:
            ends.append(n)
            
        if len(starts) == 0:
            baselines[b] = np.copy(trace)
            continue
            
        # 2. Find the largest peak
        most_counts = -1.0
        largest_index = -1
        for i in range(len(starts)):
            num_bins = ends[i] - starts[i]
            if num_bins > min_bins_in_peak:
                max_counts = trace[starts[i]]
                for j in range(starts[i]+1, ends[i]):
                    if trace[j] > max_counts:
                        max_counts = trace[j]
                if max_counts > most_counts:
                    most_counts = max_counts
                    largest_index = i
                    
        if largest_index == -1:
            baselines[b] = np.copy(trace)
            continue
            
        # 3. Peak trimming
        s = starts[largest_index]
        e = ends[largest_index]
        
        peak_start = max(0, s - bins_away)
        peak_end = min(n - 1, e + bins_away)
        
        # 4. compute peak_thresh from MAD (std of non-peak regions)
        len1 = peak_start
        len2 = n - peak_end
        if (len1 + len2) > 0:
            non_peak_diffs = np.empty(len1 + len2, dtype=np.float64)
            for i in range(len1):
                non_peak_diffs[i] = trace[i] - ransac_baseline[i]
            for i in range(len2):
                non_peak_diffs[len1 + i] = trace[peak_end + i] - ransac_baseline[peak_end + i]
            
            std_val = np.std(non_peak_diffs) 
        else:
            std_val = 0.0
            
        peak_thresh = min_sigma * std_val
        
        # 5. argmax checking
        max_trace_val = -1e9
        max_trace_idx = -1
        for i in range(n):
            if trace[i] > max_trace_val:
                max_trace_val = trace[i]
                max_trace_idx = i
                
        if (trace[max_trace_idx] - ransac_baseline[max_trace_idx]) < peak_thresh:
            baselines[b] = np.copy(trace)
            continue
            
        # 6. while loops
        while peak_start < peak_end:
            idx_check = min(peak_start + bins_away, n - 1)
            if trace[peak_start] < trace[idx_check] - peak_thresh:
                break
            peak_start += 1
            
        while peak_end > peak_start:
            idx_check = max(0, peak_end - bins_away)
            if trace[peak_end] < trace[idx_check] - peak_thresh:
                break
            peak_end -= 1
            
        if peak_start == peak_end:
            baselines[b] = np.copy(trace)
            continue
            
        # 7. assign baseline
        baseline = np.copy(trace)
        for i in range(peak_start, peak_end):
            baseline[i] = ransac_baseline[i]
            
        baselines[b] = baseline

    return baselines


@numba.njit(cache=True)
def _pairwise_block(a, lo, n):
    # numpy's DOUBLE_pairwise_sum for n <= 128, so that sums match np.sum bit for bit
    if n < 8:
        r = -0.0
        for i in range(n):
            r += a[lo + i]
        return r
    r0 = a[lo]; r1 = a[lo + 1]; r2 = a[lo + 2]; r3 = a[lo + 3]
    r4 = a[lo + 4]; r5 = a[lo + 5]; r6 = a[lo + 6]; r7 = a[lo + 7]
    i = 8
    while i < n - (n % 8):
        r0 += a[lo + i]; r1 += a[lo + i + 1]; r2 += a[lo + i + 2]; r3 += a[lo + i + 3]
        r4 += a[lo + i + 4]; r5 += a[lo + i + 5]; r6 += a[lo + i + 6]; r7 += a[lo + i + 7]
        i += 8
    res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
    while i < n:
        res += a[lo + i]
        i += 1
    return res


@numba.njit(cache=True)
def _pairwise_sum(a, n):
    # np.sum of a[:n] (float64) in numpy's order: blocks of <= 128, recursive halving (n/2 rounded down to a
    # multiple of 8) above that, evaluated here with an explicit stack
    if n <= 128:
        return _pairwise_block(a, 0, n)
    los = np.empty(64, np.int64); ns = np.empty(64, np.int64); state = np.zeros(64, np.int64)
    vals = np.empty(64, np.float64)
    sp = 0
    los[0] = 0; ns[0] = n; state[0] = 0
    result = 0.0
    while sp >= 0:
        lo = los[sp]; m = ns[sp]
        if m <= 128:
            v = _pairwise_block(a, lo, m)
            sp -= 1
            if sp < 0:
                result = v
            elif state[sp] == 1:
                vals[sp] = v; state[sp] = 2
            else:
                vals[sp] = vals[sp] + v; state[sp] = 3
            continue
        if state[sp] == 0:
            n2 = m // 2
            n2 -= n2 % 8
            state[sp] = 1
            los[sp + 1] = lo; ns[sp + 1] = n2; state[sp + 1] = 0
            sp += 1
        elif state[sp] == 2:
            n2 = m // 2
            n2 -= n2 % 8
            los[sp + 1] = lo + n2; ns[sp + 1] = m - n2; state[sp + 1] = 0
            sp += 1
        else:
            v = vals[sp]
            sp -= 1
            if sp < 0:
                result = v
            elif state[sp] == 1:
                vals[sp] = v; state[sp] = 2
            else:
                vals[sp] = vals[sp] + v; state[sp] = 3
    return result


@numba.njit(cache=True)
def _smart1_baselines_2d(traces, bins_away, ave_bins, walk_threshold, peak_lo, peak_hi, one_sided_flat):
    '''
    numba version of the 'smart' (smart1) branch of calculate_background_2d. Same operations in the same
    order (np.argmax semantics, same walk, same fit samples, int64 sums of x and x^2, numpy pairwise sums
    of y and x*y, same slope/offset expressions), so the result is bit-identical to the Python loop.
    traces: C-contiguous float64 (B, N). walk_threshold: margin of the peak walk (smart_walk_threshold, or
    ic_counts_threshold when that is None). one_sided_flat: flat baseline at the mean of the flanking bins when
    they lie on one side of the peak only (smart_one_sided_baseline 'flat'), else the line through them. With
    one_sided_flat a walk that reaches the last bin leaves no right flank (the last bin belongs to the peak region,
    as bin 0 does when the walk reaches the left edge) and a walk that reaches both edges zeroes the trace; without
    it the last bin is the single right flank bin, as before 2026-10-28.
    '''
    B, N = traces.shape
    out = np.empty_like(traces)
    m = 2 * ave_bins if ave_bins > 0 else 0
    xs = np.empty(m, np.int64)
    ys = np.empty(m, np.float64)
    xy = np.empty(m, np.float64)
    for b in range(B):
        t = traces[b]
        for k in range(N):
            out[b, k] = t[k]
        peak = 0  # np.argmax: first maximum, a NaN wins
        best = t[0]
        if not np.isnan(best):
            for k in range(1, N):
                v = t[k]
                if np.isnan(v):
                    peak = k
                    break
                if v > best:
                    best = v
                    peak = k
        if peak < peak_lo or peak > peak_hi:
            continue
        i = peak
        while i > 0:
            j = max(0, i - bins_away)
            if t[i] < t[j] + walk_threshold:
                break
            i -= 1
        peak_start = i
        i = peak
        while i < N - 1:
            j = min(N - 1, i + bins_away)
            if t[i] < t[j] + walk_threshold:
                break
            i += 1
        peak_end = i
        if one_sided_flat and peak_end == N - 1:
            #the walk reached the last bin: no right flank, as there is no left flank when it reaches bin 0 (the
            #right flank would otherwise be the last sample alone, which lies on the pulse)
            peak_end = N
            if peak_start == 0:
                #both edges reached: no flanking bins at all, so no baseline estimate; the trace is its own
                #baseline (zeroed), as for a peak outside require_peak_within
                continue
        n = 0
        for x in range(max(0, peak_start - ave_bins), peak_start):
            xs[n] = x; n += 1
        n_left = n
        for x in range(peak_end, min(peak_end + ave_bins, N)):
            xs[n] = x; n += 1
        if one_sided_flat and n > 0 and (n_left == 0 or n_left == n):
            #flanking bins on one side of the peak only (the walk reached a trace edge): flat baseline at their
            #mean; a line through them would be extrapolated over the whole peak region
            for k in range(n):
                ys[k] = t[xs[k]]
            slope = 0.0
            offset = _pairwise_sum(ys, n) / n
        elif n > 1:
            sum_x = 0
            sum_x2 = 0
            for k in range(n):
                ys[k] = t[xs[k]]
                xy[k] = xs[k] * ys[k]
                sum_x += xs[k]
                sum_x2 += xs[k] * xs[k]
            sum_y = _pairwise_sum(ys, n)
            sum_xy = _pairwise_sum(xy, n)
            denominator = n * sum_x2 - sum_x * sum_x
            if denominator != 0:
                slope = (n * sum_xy - sum_x * sum_y) / denominator
                offset = (sum_y - slope * sum_x) / n
            else:
                slope = 0.0
                offset = sum_y / n
        elif n == 1:
            slope = 0.0
            offset = t[xs[0]]
        else:
            slope = 0.0
            offset = 0.0
        for x in range(peak_start, peak_end):
            out[b, x] = offset + slope * x
    return out


_CUSOLVER_OK = None
def _cusolver_available():
    '''
    cp.linalg.svd imports cupy's cusolver on every call. In the gadget_analysis environment that import fails
    once torch is loaded (libcusolver.so.11 needs cublasSetEnvironmentMode, torch's libcublas 12.1 lacks it),
    so every call raised and fell back to np.linalg.svd, paying for the failed import each time. Probe once.
    '''
    global _CUSOLVER_OK
    if _CUSOLVER_OK is None:
        try:
            from cupy_backends.cuda.libs import cusolver  # noqa: F401
            _CUSOLVER_OK = True
        except Exception:
            _CUSOLVER_OK = False
    return _CUSOLVER_OK


class raw_h5_file:
    def __init__(self, file_path, zscale, flat_lookup_csv):
        self.file_path = file_path
        self.h5_file = h5py.File(file_path, 'r')
        self.padxy = np.loadtxt(os.path.join(os.path.dirname(__file__), 'padxy.txt'), delimiter=',')
        
        self.flat_lookup_file_path = flat_lookup_csv        
        if not os.path.isabs(self.flat_lookup_file_path) and not os.path.exists(self.flat_lookup_file_path):
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.flat_lookup_file_path = os.path.join(repo_root, self.flat_lookup_file_path)
            
        self.flat_lookup = np.loadtxt(self.flat_lookup_file_path, delimiter=',', dtype=int)
        self.pad_plane = np.genfromtxt(os.path.join(os.path.dirname(__file__),'PadPlane.csv'),delimiter=',', filling_values=-1) #used for mapping pad numbers to a 2D grid
        self.pad_to_xy_index = {} #maps pad number to (x_index,y_index)
        for y in range(len(self.pad_plane)):
            for x in range(len(self.pad_plane[0])):
                pad = self.pad_plane[x,y]
                if pad != -1:
                    self.pad_to_xy_index[int(pad)] = (x,y)

        self.chnls_to_pad = {} #maps tuples of (asad, aget, channel) to pad number
        self.chnls_to_xy_coord = {} #maps tuples of (asad, aget, channel) to (x,y) coordinates in mm
        self.chnls_to_xy_index = {}
        self.pad_to_chnl = {}
        for line in self.flat_lookup:
            chnls = tuple(line[0:4])
            pad = line[4]
            self.chnls_to_pad[chnls] = pad
            self.pad_to_chnl[pad] = chnls
            self.chnls_to_xy_coord[chnls] = self.padxy[pad]
            self.chnls_to_xy_index[chnls] = self.pad_to_xy_index[pad]
        #round xy to nearest 10nths place to avoid issues with different floating point formats
        self.xy_to_pad = {tuple(np.round(self.padxy[pad], 1)):pad for pad in range(len(self.padxy))}
        self.xy_to_chnls = {tuple(np.round(self.chnls_to_xy_coord[chnls], 1)):chnls 
                            for chnls in self.chnls_to_xy_coord}
        self.xy_index_to_pad = {self.pad_to_xy_index[pad]:pad for pad in self.pad_to_xy_index}
        
        self.zscale = zscale #conversion factor from time bin to mm

        self.pad_backgrounds = None #initialize with determine_pad_backgrounds

        #color map for plotting
        cdict={'red':  ((0.0, 0.0, 0.0),
                    (0.25, 0.0, 0.0),
                    (0.5, 0.8, 1.0),
                    (0.75, 1.0, 1.0),
                    (1.0, 0.4, 1.0)),

            'green': ((0.0, 0.0, 0.0),
                    (0.25, 0.0, 0.0),
                    (0.5, 0.9, 0.9),
                    (0.75, 0.0, 0.0),
                    (1.0, 0.0, 0.0)),

            'blue':  ((0.0, 0.0, 0.4),
                    (0.25, 1.0, 1.0),
                    (0.5, 1.0, 0.8),
                    (0.75, 0.0, 0.0),
                    (1.0, 0.0, 0.0))
            }
        cdict['alpha'] = ((0.0, 0.0, 0.0),
                        (0.3,0.2, 0.2),
                        (0.8,1.0, 1.0),
                        (1.0, 1.0, 1.0))
        self.cmap = LinearSegmentedColormap('test',cdict)

        self.background_subtract_mode = 'none' #none, fixed window, smart, smart2, or convolution
        self.background_convolution_kernel = None#bin backgrounds are determined by convolving the trace with this array
        self.remove_outliers = False
        self.num_background_bins = (0,0) #number of time bins to use for per event background subtraction
        self.length_counts_threshold = 300 #threshold to use when calculating range
        self.ic_counts_threshold = 100 #threshold to use when calculating energy
        #allowed modes = 'all data', 'peak only', or 'near peak'
        #all data mode will make use of the entire trace from each pad
        #'peak only' will make use of only the max value the trace reached on each pad
        #'near peak' will only use data within some window of the peak of each trace
        self.data_select_mode = 'all data' 
        self.near_peak_window_width = 100 #+/- time bins to include
        #number of bins past the start/end of trace to use when computing baseline
        self.num_smart_background_ave_bins = 5 
        #number of bins to go left/right when finding the start/end of a peak, when comparting difference to ic_threshold
        self.smart_bins_away_to_check = 3
        #margin (ADC) of the smart peak walk: the peak region grows while the trace is at least this much above the
        #trace smart_bins_away_to_check bins further out. None (default): ic_counts_threshold, as before 2026-10-28.
        #With a threshold of 0 a slowly rising pedestal or a negative-going pulse carries the walk to a trace edge.
        self.smart_walk_threshold = None
        #baseline under the peak when the flanking bins lie on one side of the peak only (the walk reached a trace
        #edge): 'flat' (default) = their mean; 'line' = the line through them, extrapolated over the peak region
        #(the behaviour before 2026-10-28, which gave +-2000 ADC ramps on 127-bin e25058 traces). With 'flat' a walk
        #that reaches the last bin leaves no right flank (the last bin belongs to the peak region, as bin 0 does at
        #the left edge) and a walk that reaches both edges zeroes the trace; with 'line' the last bin is the single
        #right flank bin.
        self.smart_one_sided_baseline = 'flat'
        self.smart2_min_bins_in_peak = 10
        self.smart2_min_sigma = 2
        self.smart2_ransac_percentile = 100
        self.require_peak_within = (-np.inf, np.inf)#currentlt implemented for near peak mode only. Zero entire trace if peak is not within this window
        self.include_counts_on_veto_pads = False #if counts on veto pads should be included for energy calibraiton
        #if True, will try to fit a skew normal distrubution to bins just to the left and right of the railed region, and use this to replace the railed region
        self.reconstruct_railed_pads = False
        self.railed_pad_reconstruct_fit_bins = (8,8) #bins to the (left, right) of the railed region to use for fitting
        #'lm': railed_pad_repair.repair_lm (numba fit, ~60x faster); 'scipy': railed_pad_repair.repair (curve_fit),
        #which the files processed before 2026-10-23 used. See railed_pad_repair for how they compare.
        self.railed_pad_repair_method = 'lm'
        #if True, time-bin columns that are zero in every trace of every probed event at the start or end of the
        #traces (written by the merger when the GET depth did not match, e.g. e23035 GET 214, 215, 216, 229, 234, 235:
        #512 columns with the last 112 zero) are removed when events are read, so the run looks like a run of its
        #true depth. A no-op for files without padding. See _zero_pad_info.
        self.remove_zero_pad = False
        #hybrid baseline: if set (e.g. 'smart2'), traces of VETO_PADS are baseline-subtracted with this mode and
        #with the veto_<param> values that are set (VETO_BASELINE_PARAMS), all other pads with
        #background_subtract_mode. None (default): every pad uses background_subtract_mode.
        self.veto_background_subtract_mode = None

        self.batched_data_cache = {}
        self.batch_event_cache_size = 1

        self.apply_gain_match = False
        self.pad_gains=np.ones(NUM_PADS)
        #pad gain map of a processing config (row pad_gain_file,str,<path>, relative to <exp>_analysis/pad_gains/). Not
        #used by the processing and not part of the settings hash: the runs module's get_pad_gains applies it to the
        #processed per-pad charges (process_runs.get_config_pad_gain_file), so gain maps coexist as differently named configs.
        self.pad_gain_file = None

        #cobo and asad selction. Can be "all" or a list of ints
        self.asads='all'
        self.cobos='all'
        self.pads='all'

        #caching avoids redoing baseline subtraction unnecessarily. The caches are keyed on the event number (and,
        #for xyte/xyze, on the threshold and include_veto_pads) and know nothing about the settings: call
        #invalidate_cache() after changing any setting that affects get_data (it also clears batched_data_cache,
        #which keeps the processed events of the current batch whatever cache_enable is).
        self.cache_enable = False
        self.cached_event = self.cached_event_xyze = self.cached_event_xyte = np.inf
        self.cached_data = None
        self.cached_xyte = self.cached_xyze = None

        # Cache for fast lookup of new bucketed group structure
        self._batch_existence_cache = {}

        #look at first event and figure out number of time bins
        try:
            first_event_data = self._get_event_dataset(self.get_event_num_bounds()[0], kind='data')
            #from the dataset shape: an event without any rows still carries the trace length
            self.num_time_bins = first_event_data.shape[1]-FIRST_DATA_BIN
        except (KeyError, IndexError):
            self.num_time_bins = 512 # Default value if there are no events in the h5 file

    def _event_dataset_path(self, event_number, kind='data'):
        """
        Path of an event's dataset inside the 'get' group, supporting both the new bucketed
        ('batch_XXXX') structure and the legacy flat structure.
        """
        dataset_name = "evt%d_%s" % (event_number, kind)
        batch_index = int(event_number // 5000)
        batch_name = "batch_%04d" % batch_index

        # Check cache to avoid repeatedly querying HDF5 metadata
        if batch_name not in self._batch_existence_cache:
            self._batch_existence_cache[batch_name] = batch_name in self.h5_file['get']

        # Nested bucketed path, or fallback to root flat path
        if self._batch_existence_cache[batch_name]:
            return batch_name + '/' + dataset_name
        return dataset_name

    def _get_event_dataset(self, event_number, kind='data'):
        """
        Helper method to retrieve the dataset (h5py Dataset) for a given event.
        """
        return self.h5_file['get'][self._event_dataset_path(event_number, kind)]

    def _read_event(self, event_number, kind='data'):
        '''
        Contents of an event's dataset as a numpy array (same values and dtype as self._get_event_dataset(...)[()]),
        read with the low-level h5py API: 50-60 us per dataset instead of 130-210 us for the group and dataset
        lookups of the high-level API (measured 2026-10-23). Raises KeyError if the event is missing, like
        _get_event_dataset.
        '''
        gid = self.__dict__.get('_get_group_id')
        if gid is None:
            gid = self.__dict__['_get_group_id'] = self.h5_file['get'].id
        dsid = h5py.h5d.open(gid, self._event_dataset_path(event_number, kind).encode())
        out = np.empty(dsid.shape, dtype=dsid.dtype)
        if out.size:
            dsid.read(h5py.h5s.ALL, h5py.h5s.ALL, out)
        return out

    #num_time_bins is kept in the instance __dict__ (so save_config_file still writes it); with remove_zero_pad
    #it reports the depth after the padding is removed
    @property
    def num_time_bins(self):
        n = self.__dict__.get('num_time_bins', 512)
        if self.__dict__.get('remove_zero_pad', False):
            info = self._zero_pad_info()
            if info['n_cols'] == n and (info['lead'] or info['trail']):
                return n - info['lead'] - info['trail']
        return n

    @num_time_bins.setter
    def num_time_bins(self, value):
        self.__dict__['num_time_bins'] = value

    def _zero_pad_info(self):
        '''
        Detect zero padding once per file: probe ZERO_PAD_PROBE_EVENTS events spread over the run; for each, count the
        data columns at the start (lead) and end (trail) that are exactly 0 in every trace. The padding is the minimum
        over the probed events (a column is removed only if it is all-zero in every probed event). Real data never has
        a whole column at 0 (pedestals are ~250-400 ADC), so files without padding give lead = trail = 0.
        Requires >= 3 probed events with data and one common trace length; otherwise nothing is removed (warning).
        '''
        info = self.__dict__.get('_zero_pad_cache')
        if info is None:
            import warnings
            leads, trails, ncols = [], [], set()
            first, last = self.get_event_num_bounds()
            if last >= first:
                for evt in np.unique(np.linspace(first, last, ZERO_PAD_PROBE_EVENTS).astype(np.int64)):
                    try:
                        d = self._read_event(evt, kind='data')
                    except KeyError:
                        continue
                    if d.ndim != 2 or len(d) == 0 or d.shape[1] <= FIRST_DATA_BIN:
                        continue
                    z = np.all(d[:, FIRST_DATA_BIN:] == 0, axis=0)
                    if z.all():
                        continue #an all-zero event says nothing about the padding
                    leads.append(int(np.argmin(z)))
                    trails.append(int(np.argmin(z[::-1])))
                    ncols.add(int(z.size))
            if len(leads) >= 3 and len(ncols) == 1:
                info = dict(lead=min(leads), trail=min(trails), n_cols=ncols.pop(), probed=len(leads))
            else:
                warnings.warn('remove_zero_pad: %s: %d usable probe events, trace lengths %s; no padding removed'
                              % (self.file_path, len(leads), sorted(ncols)))
                info = dict(lead=0, trail=0, n_cols=None, probed=len(leads))
            self.__dict__['_zero_pad_cache'] = info
        return info

    def _remove_zero_pad(self, data):
        '''Rows [cobo, asad, aget, channel, pad, ..., samples] of one event, padding columns removed (if any).'''
        info = self._zero_pad_info()
        if not (info['lead'] or info['trail']):
            return data
        d = np.asarray(data)
        if d.ndim != 2 or d.shape[1] - FIRST_DATA_BIN != info['n_cols']:
            raise ValueError('remove_zero_pad: event with %s columns in %s, padding was detected for %s data columns'
                             % (d.shape, self.file_path, info['n_cols']))
        return np.concatenate((d[:, :FIRST_DATA_BIN],
                               d[:, FIRST_DATA_BIN + info['lead']:FIRST_DATA_BIN + info['n_cols'] - info['trail']]), axis=1)

    def close(self):
        if self.h5_file:
            self.h5_file.close()
            self.h5_file = None
        self.__dict__.pop('_get_group_id', None)

    def invalidate_cache(self):
        '''
        Forget every cached result: the per-event data, xyte and xyze caches, the processed events of the current
        batch (batched_data_cache) and the railed-pad cache. Call after changing any setting that affects get_data
        or get_xyze (baseline mode and parameters, data_select_mode, outlier removal, gain match, zero-pad removal,
        the cobo/asad/pad selection, zscale, ...): the caches are keyed on the event number only. load_config and
        the raw event viewer call it; the per-file caches (zero padding, channel tables) are kept.
        '''
        self.cached_event = self.cached_event_xyte = self.cached_event_xyze = np.inf
        self.cached_data = self.cached_xyte = self.cached_xyze = None
        self.batched_data_cache.clear()
        self.__dict__.pop('_railed_pads_cache', None)

    def load_config(self, config_path):
        '''
        Apply the settings of a csv config (rows name,type,value, as written by save_config_file). Every row is
        parsed before any is applied, so a bad row leaves the instance unchanged (ValueError). Rows naming cached
        state or properties of the file (_CONFIG_STATE_KEYS, written by save_config_file before 2026-10-28) are
        ignored. The caches are invalidated afterwards.
        '''
        import csv
        values = {}
        with open(config_path, 'r') as f:
            for row in csv.reader(f):
                if len(row) < 3:
                    continue
                name = row[0].strip()
                type_name = row[1].strip()
                val_str = row[2].strip()
                if name in _CONFIG_STATE_KEYS:
                    continue
                if type_name == 'str':
                    val = val_str
                elif type_name == 'int':
                    val = int(val_str)
                elif type_name == 'float':
                    val = float(val_str) #accepts inf and -inf
                elif type_name == 'bool':
                    val = val_str.lower() in ('true', '1', 't', 'y', 'yes')
                elif type_name in ('list', 'tuple'):
                    val = _literal(val_str)
                    val = tuple(val) if type_name == 'tuple' else list(val)
                elif type_name == 'ndarray':
                    val = np.array(_literal(val_str))
                else:
                    continue
                values[name] = val
        for name, val in values.items():
            setattr(self, name, val)
        self.invalidate_cache()

    def save_config_file(self, config_path):
        '''
        Write the settings as a csv config (name,type,value) that load_config reads back. Lookup tables, the open
        file and the cached state (_CONFIG_STATE_KEYS) are not written. A setting that is None cannot be expressed
        and is skipped, so a config cannot unset a setting that is set.
        '''
        import csv
        import os
        exclude = {'xy_to_pad', 'xy_to_chnls', 'xy_index_to_pad', 'pad_plane', 'padxy', 'flat_lookup',
                   'pad_to_xy_index', 'chnls_to_pad', 'chnls_to_xy_coord', 'chnls_to_xy_index', 'pad_to_chnl',
                   'cmap', 'h5_file', 'pad_backgrounds', 'file_path'} | _CONFIG_STATE_KEYS
        def plain(x):
            #python scalars inside lists/tuples, so that str() writes literals (1.5, not np.float64(1.5))
            return x.item() if isinstance(x, np.generic) else x
        os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
        with open(config_path, 'w', newline='') as f:
            writer = csv.writer(f)
            for k, v in self.__dict__.items():
                if k in exclude or k.startswith('_'):
                    continue
                t = type(v)
                if t == str:
                    writer.writerow([k, 'str', v])
                elif t == int or isinstance(v, np.integer):
                    writer.writerow([k, 'int', str(v)])
                elif t == float or isinstance(v, np.floating):
                    writer.writerow([k, 'float', str(v)])
                elif t == bool or isinstance(v, np.bool_):
                    writer.writerow([k, 'bool', str(v)])
                elif t in (list, tuple):
                    writer.writerow([k, t.__name__, str(t(plain(x) for x in v))])
                elif t == np.ndarray:
                    writer.writerow([k, 'ndarray', str(v.tolist())])

    def get_pad_from_xy(self, xy):
        '''
        xy: tuple of (x,y) to lookup pad number for
        '''
        xy = tuple(np.round(xy, 1))
        return self.xy_to_pad[xy]
    
    def get_chnl_from_xy(self, xy):
        '''
        xy: tuple of (x,y) to lookup pad number for
        '''
        xy = tuple(np.round(xy, 1))
        return self.xy_to_chnls[xy]

    def get_timestamp(self, event_number):
        '''
        Returns timestamp in seconds.
        '''
        #timestamps are stored in units of 10ns
        return self._read_event(event_number, kind='header')[1]/1e8

    def get_timestamps_array(self):
        '''
        Returns an array with the timestamps of each event, in seconds
        '''
        first, last = self.get_event_num_bounds()
        if last < first:
            return np.array([])
        num_events = last - first + 1
        to_return = np.zeros(num_events)
        for i, evt in tqdm.tqdm(enumerate(range(first, last+1))):
            to_return[i] = self.get_timestamp(evt)
        return to_return

    def get_railed_pads(self, event_number):
        '''
        Returns a list of pads which railed in the current event
        '''
        #get_data computes this from the raw samples while it reads a batch; avoids reading the event twice
        railed_cache = self.__dict__.get('_railed_pads_cache')
        if railed_cache is not None and event_number in railed_cache and self.asads == 'all' \
                and self.cobos == 'all' and self.pads == 'all':
            return railed_cache.pop(event_number)
        data = self._read_event(event_number, kind='data')
        if self.remove_zero_pad:
            data = self._remove_zero_pad(data)
        traces = data[:,FIRST_DATA_BIN:]
        which_ones_railed = np.logical_not(np.all(traces<4095, axis=1))
        ch_info = data[which_ones_railed,0:4]
        return [self.chnls_to_pad[tuple(info)] for info in ch_info]

        

    def get_data(self, event_number):
        '''
        Get data for event, with background subtraction and pad outlier removal applied as specified
        by member variables.

        Outlier removal:
        1. Populate a pad plane image with all zeros, except for pads which fired
        2. Use skimage.measure.label to determine to label pads based on connectivity
        3. Make a new datacube which just has the pads from the largest blob

        Veto pads should NOT be removed during outlier removal.
        Does NOT apply thresholding. However, this is applied in get_xyte and and get_xyze.
        '''
        if self._vectorized_bookkeeping_ok():
            return self._get_data_vectorized(event_number)
        if self.cache_enable and event_number == self.cached_event:
            return np.array(self.cached_data, copy=True)

        if self.batch_event_cache_size > 1 and event_number in self.batched_data_cache:
            data = self.batched_data_cache.pop(event_number)
            if self.cache_enable:
                self.cached_data = data
                self.cached_event = event_number
            return np.array(data, copy=True)

        first_evt, last_evt = self.get_event_num_bounds()
        batch_end = min(event_number + self.batch_event_cache_size, last_evt + 1)
        batch_events = range(event_number, batch_end)

        event_datas = {}
        event_valid_indices = {}
        event_pad_images = {}

        massive_traces = []
        massive_trace_lens = []
        massive_pads = []

        for evt in batch_events:
            try:
                data = self._read_event(evt, kind='data')
            except KeyError:
                continue
            if self.remove_zero_pad:
                data = self._remove_zero_pad(data)

            if self.asads != 'all' or self.cobos != 'all' or self.pads != 'all':
                to_copy = []
                for i, line in enumerate(data):
                    chnl_info = tuple(line[0:4])
                    cobo, asad, *rest = chnl_info
                    if chnl_info in self.chnls_to_pad:
                        pad = self.chnls_to_pad[chnl_info]
                        if (self.cobos == 'all' or cobo in self.cobos) and \
                                (self.asads == 'all' or asad in self.asads) and \
                                (self.pads == 'all' or pad in self.pads):
                            to_copy.append(i)
                data = np.array(data[to_copy], dtype=float, copy=True)
            else:
                data = np.array(data, copy=True, dtype=float)

            if len(data) == 0:
                event_datas[evt] = data
                event_valid_indices[evt] = []
                event_pad_images[evt] = None
                massive_trace_lens.append(0)
                continue

            pad_image = np.zeros(np.shape(self.pad_plane)) if self.remove_outliers else None
            valid_indices = []
            valid_pads = []

            for i, line in enumerate(data):
                chnl_info = tuple(line[0:4])
                if chnl_info in self.chnls_to_pad:
                    pad = self.chnls_to_pad[chnl_info]
                    valid_indices.append(i)
                    valid_pads.append(pad)
                    if self.reconstruct_railed_pads:
                        if np.max(line[FIRST_DATA_BIN:]) == 4095:
                            line[FIRST_DATA_BIN:] = self._repair_railed_trace(line[FIRST_DATA_BIN:])
                    if self.remove_outliers:
                        x,y = self.pad_to_xy_index[pad]
                        pad_image[x,y] = 1

            event_datas[evt] = data
            event_valid_indices[evt] = valid_indices
            event_pad_images[evt] = pad_image

            if len(valid_indices) > 0 and self.background_subtract_mode != 'none':
                traces_2d = data[valid_indices, FIRST_DATA_BIN:]
                massive_traces.append(traces_2d)
                massive_pads.append(np.array(valid_pads, dtype=np.int64))
                massive_trace_lens.append(len(valid_indices))
            else:
                massive_trace_lens.append(0)

        if len(massive_traces) > 0 and self.background_subtract_mode != 'none':
            massive_traces_2d = np.concatenate(massive_traces, axis=0)
            massive_baselines = self._calculate_batch_backgrounds(massive_traces_2d, np.concatenate(massive_pads))

            idx = 0
            for i, evt in enumerate(batch_events):
                if evt not in event_datas: continue
                trace_len = massive_trace_lens[i]
                if trace_len > 0:
                    baselines = massive_baselines[idx:idx+trace_len]
                    valid_indices = event_valid_indices[evt]
                    event_datas[evt][valid_indices, FIRST_DATA_BIN:] -= baselines
                    idx += trace_len

        for evt in batch_events:
            if evt not in event_datas: continue
            data = event_datas[evt]
            
            if len(data) == 0:
                self.batched_data_cache[evt] = data
                continue

            if self.remove_outliers:
                labeled_image = skimage.measure.label(event_pad_images[evt], background=0)
                labels, counts = np.unique(labeled_image[labeled_image!=0], return_counts=True)
                if len(counts) > 0:
                    bigest_label = labels[np.argmax(counts)]
                    new_data = []
                    for line in data:
                        chnl_info = tuple(line[0:4])
                        if chnl_info in self.chnls_to_pad:
                            pad = self.chnls_to_pad[chnl_info]
                            x,y = self.pad_to_xy_index[pad]
                            if labeled_image[x,y] == bigest_label or pad in VETO_PADS:
                                new_data.append(line)
                        else:
                            continue
                    data = np.array(new_data)
                else:
                    data = np.array([])

            if self.data_select_mode == 'peak only':
                for line in data:
                    peak_index = np.argmax(line[FIRST_DATA_BIN:])
                    line[FIRST_DATA_BIN:FIRST_DATA_BIN+peak_index] = 0
                    if peak_index < len(line[FIRST_DATA_BIN:]):
                        line[FIRST_DATA_BIN+peak_index+1:] = 0
            elif self.data_select_mode == 'near peak':
                for line in data:
                    peak_index = np.argmax(line[FIRST_DATA_BIN:])
                    if peak_index < self.require_peak_within[0] or peak_index > self.require_peak_within[1]:
                        line[FIRST_DATA_BIN:] = 0
                    else:
                        if peak_index - self.near_peak_window_width > 0:
                            line[FIRST_DATA_BIN:FIRST_DATA_BIN+peak_index - self.near_peak_window_width] = 0
                        if peak_index + self.near_peak_window_width < len(line[FIRST_DATA_BIN:]):
                            line[FIRST_DATA_BIN+peak_index + self.near_peak_window_width:] = 0
                            
            if self.apply_gain_match:
                for line in data:
                    chnl_info = tuple(line[0:4])
                    if chnl_info in self.chnls_to_pad:
                        pad = self.chnls_to_pad[chnl_info]
                        line[FIRST_DATA_BIN:] *= self.pad_gains[pad]

            self.batched_data_cache[evt] = data

        if event_number in self.batched_data_cache:
            data = self.batched_data_cache.pop(event_number)
        else:
            data = np.array([])
            
        if self.cache_enable:
            self.cached_data = data
            self.cached_event = event_number
        return np.array(data, copy=True)

    def _repair_railed_trace(self, trace):
        '''Repaired copy of a railed trace (reconstruct_railed_pads), with the method set by railed_pad_repair_method.'''
        if self.railed_pad_repair_method == 'lm':
            return railed_pad_repair.repair_lm(trace, *self.railed_pad_reconstruct_fit_bins)
        if self.railed_pad_repair_method == 'scipy':
            return railed_pad_repair.repair(trace, *self.railed_pad_reconstruct_fit_bins)
        raise ValueError('railed_pad_repair_method must be lm or scipy, not %r' % (self.railed_pad_repair_method,))

    def _chnl_tables(self):
        '''Lookup arrays equivalent to the chnls_to_pad / pad_to_xy_index dicts (built once per instance).'''
        tables = self.__dict__.get('_chnl_tables_cache')
        if tables is None:
            keys = np.array(list(self.chnls_to_pad.keys()), dtype=np.int64)
            dims = keys.max(axis=0) + 1
            lut = np.full(int(np.prod(dims)), -1, dtype=np.int64)
            lut[np.ravel_multi_index(keys.T, dims)] = np.array(list(self.chnls_to_pad.values()), dtype=np.int64)
            npads = max(max(self.chnls_to_pad.values()), max(self.pad_to_xy_index)) + 1
            px = np.full(npads, -1, np.int64)
            py = np.full(npads, -1, np.int64)
            for pad, (x, y) in self.pad_to_xy_index.items():
                px[pad] = x
                py[pad] = y
            is_veto = np.zeros(npads, bool)
            is_veto[[p for p in VETO_PADS if p < npads]] = True
            tables = dict(dims=dims, lut=lut, px=px, py=py, is_veto=is_veto)
            self.__dict__['_chnl_tables_cache'] = tables
        return tables

    def _pads_of_rows(self, chnl_cols):
        '''
        Pad number of each row from its (cobo, asad, aget, channel) columns, -1 where
        `tuple(line[0:4]) in self.chnls_to_pad` would be False (only exact integer values match).
        '''
        tables = self._chnl_tables()
        c = np.asarray(chnl_cols)
        if c.shape[0] == 0:
            return np.empty(0, np.int64)
        ci = np.floor(c).astype(np.int64) if c.dtype.kind == 'f' else c.astype(np.int64)
        ok = np.all((ci >= 0) & (ci < tables['dims']) & (ci == c), axis=1)
        pads = np.full(len(c), -1, np.int64)
        if ok.any():
            pads[ok] = tables['lut'][np.ravel_multi_index(ci[ok].T, tables['dims'])]
        return pads

    def _vectorized_bookkeeping_ok(self):
        return self.asads == 'all' and self.cobos == 'all' and self.pads == 'all' \
            and self.data_select_mode == 'all data' and not self.apply_gain_match

    def _get_data_vectorized(self, event_number):
        '''
        get_data for all cobos/asads/pads, data_select_mode 'all data' and no gain match: identical output,
        with the per-line dictionary loops replaced by array lookups. Also fills the railed pad cache that
        get_railed_pads uses for the events of the current batch.
        '''
        if self.cache_enable and event_number == self.cached_event:
            return np.array(self.cached_data, copy=True)

        if self.batch_event_cache_size > 1 and event_number in self.batched_data_cache:
            data = self.batched_data_cache.pop(event_number)
            if self.cache_enable:
                self.cached_data = data
                self.cached_event = event_number
            return np.array(data, copy=True)

        tables = self._chnl_tables()
        first_evt, last_evt = self.get_event_num_bounds()
        batch_end = min(event_number + self.batch_event_cache_size, last_evt + 1)
        batch_events = range(event_number, batch_end)

        railed_cache = {}
        self.__dict__['_railed_pads_cache'] = railed_cache #only the current batch is kept
        event_datas, event_valid_indices, event_pads = {}, {}, {}
        massive_traces, massive_trace_lens, massive_pads = [], [], []

        for evt in batch_events:
            try:
                data = self._read_event(evt, kind='data')
            except KeyError:
                continue
            if self.remove_zero_pad:
                data = self._remove_zero_pad(data)
            data = np.array(data, copy=True, dtype=float)

            if len(data) == 0:
                event_datas[evt] = data
                event_valid_indices[evt] = []
                event_pads[evt] = None
                massive_trace_lens.append(0)
                railed_cache[evt] = []
                continue

            pads = self._pads_of_rows(data[:, 0:4])
            #railed pads from the raw samples, as get_railed_pads computes them. Only cached if every railed
            #channel is mapped; otherwise get_railed_pads reads the event and raises KeyError as before.
            railed_rows = np.logical_not(np.all(data[:, FIRST_DATA_BIN:] < 4095, axis=1))
            if np.all(pads[railed_rows] >= 0):
                railed_cache[evt] = [self.chnls_to_pad[tuple(int(v) for v in row)] for row in data[railed_rows, 0:4]]

            valid = pads >= 0
            valid_indices = np.nonzero(valid)[0]
            if self.reconstruct_railed_pads:
                for i in np.nonzero(valid & (np.max(data[:, FIRST_DATA_BIN:], axis=1) == 4095))[0]:
                    line = data[i]
                    line[FIRST_DATA_BIN:] = self._repair_railed_trace(line[FIRST_DATA_BIN:])

            event_datas[evt] = data
            event_valid_indices[evt] = valid_indices
            event_pads[evt] = pads

            if len(valid_indices) > 0 and self.background_subtract_mode != 'none':
                massive_traces.append(data[valid_indices, FIRST_DATA_BIN:])
                massive_pads.append(pads[valid_indices])
                massive_trace_lens.append(len(valid_indices))
            else:
                massive_trace_lens.append(0)

        if len(massive_traces) > 0 and self.background_subtract_mode != 'none':
            massive_traces_2d = np.concatenate(massive_traces, axis=0)
            massive_baselines = self._calculate_batch_backgrounds(massive_traces_2d, np.concatenate(massive_pads))
            idx = 0
            for i, evt in enumerate(batch_events):
                if evt not in event_datas: continue
                trace_len = massive_trace_lens[i]
                if trace_len > 0:
                    event_datas[evt][event_valid_indices[evt], FIRST_DATA_BIN:] -= massive_baselines[idx:idx+trace_len]
                    idx += trace_len

        for evt in batch_events:
            if evt not in event_datas: continue
            data = event_datas[evt]
            if len(data) == 0:
                self.batched_data_cache[evt] = data
                continue
            if self.remove_outliers:
                pads = event_pads[evt]
                pv = pads[pads >= 0]
                pad_image = np.zeros(np.shape(self.pad_plane))
                pad_image[tables['px'][pv], tables['py'][pv]] = 1
                labeled_image = skimage.measure.label(pad_image, background=0)
                labels, counts = np.unique(labeled_image[labeled_image!=0], return_counts=True)
                if len(counts) > 0:
                    bigest_label = labels[np.argmax(counts)]
                    keep = np.zeros(len(data), bool)
                    keep[pads >= 0] = (labeled_image[tables['px'][pv], tables['py'][pv]] == bigest_label) | tables['is_veto'][pv]
                    data = np.array(data[keep]) if keep.any() else np.array([])
                else:
                    data = np.array([])
            self.batched_data_cache[evt] = data

        if event_number in self.batched_data_cache:
            data = self.batched_data_cache.pop(event_number)
        else:
            data = np.array([])
        if self.cache_enable:
            self.cached_data = data
            self.cached_event = event_number
        return np.array(data, copy=True)

    def _event_rows(self, event_data, include_veto_pads):
        '''rows of event_data with a pad mapping (and not a veto pad unless include_veto_pads), and their pads'''
        if event_data.ndim != 2 or len(event_data) == 0:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        pads = self._pads_of_rows(event_data[:, 0:4])
        keep = pads >= 0
        if not include_veto_pads:
            keep &= ~self._chnl_tables()['is_veto'][np.where(keep, pads, 0)]
        rows = np.nonzero(keep)[0]
        return rows, pads[rows]

    def calculate_background(self, trace, debug_plots=False, pad=None):
        '''
        Wrapper for calculate_background_2d that takes a 1D trace and returns a 1D baseline.
        pad: pad number of the trace (optional); for a veto pad the veto baseline settings are used when
        veto_background_subtract_mode is set, as in get_data.
        '''
        trace_2d = np.ascontiguousarray(trace, dtype=np.float64).reshape(1, -1)
        veto = pad is not None and pad in VETO_PADS and self.veto_background_subtract_mode is not None
        return self.calculate_background_2d(trace_2d, debug_plots, veto=veto)[0]

    def _calculate_batch_backgrounds(self, traces_2d, pads):
        '''
        Baselines for the rows of a get_data batch (pads: pad number of each row). Without
        veto_background_subtract_mode this is calculate_background_2d(traces_2d). With it, veto-pad rows and
        the other rows are computed separately, each with its own mode and parameters.
        '''
        if self.veto_background_subtract_mode is None:
            return self.calculate_background_2d(traces_2d)
        is_veto = np.isin(pads, VETO_PADS)
        baselines = np.empty_like(traces_2d)
        if np.any(~is_veto):
            baselines[~is_veto] = self.calculate_background_2d(traces_2d[~is_veto])
        if np.any(is_veto):
            baselines[is_veto] = self.calculate_background_2d(traces_2d[is_veto], veto=True)
        return baselines

    def calculate_background_2d(self, traces_2d, debug_plots=False, mode=None, veto=False):
        '''
        Return calculated background for each timebin for a 2D array of traces.

        Traces should only contain data bins
        veto: use the veto-pad settings (veto_background_subtract_mode and the veto_<param> keys that are set)
        mode: overrides the mode (default background_subtract_mode, or veto_background_subtract_mode if veto)
        '''
        p = _BaselineParams(self, veto)
        if mode is None:
            mode = p.background_subtract_mode
        if mode == 'smart2':
            '''
            RANSAC lowest 25% of points to get initial baseline estimate
            Find peak with maximum value and more than 5 consecutive bins above the baseline found by RANSAC. Trim trace to 10 bins to each side of the found block of bins.
            Starting from the left of the trimmed trace, find the first place where the current time bin is more than MAD (from RANSAC) counts below the sample 10 samples later, and further trim the trace here. Do the same thing from the right.
            Go 10 bins away from each end of the peak, and fit a line to the next 20 bins to get a local estimate of the baseline
            '''
            if USE_GPU:
                ransac_baseline_2d = batched_line_fit_2d(traces_2d)
            else:
                ransac_baseline_2d = np.zeros_like(traces_2d)
                x = np.arange(traces_2d.shape[1]).reshape(-1, 1)
                for b in range(traces_2d.shape[0]):
                    ransac = linear_model.RANSACRegressor(random_state=0) #deterministic, like the GPU version
                    ransac.fit(x, traces_2d[b])
                    ransac_baseline_2d[b] = ransac.predict(x)
                
            return _smart2_numba_peak_trimming_2d(
                np.ascontiguousarray(traces_2d, dtype=np.float64), 
                np.ascontiguousarray(ransac_baseline_2d, dtype=np.float64), 
                p.smart2_min_bins_in_peak, 
                p.smart_bins_away_to_check, 
                p.smart2_min_sigma
            )

        if mode == 'smart':
            walk_thr = p.ic_counts_threshold if p.smart_walk_threshold is None else p.smart_walk_threshold
            if p.smart_one_sided_baseline not in ('flat', 'line'):
                raise ValueError("smart_one_sided_baseline must be 'flat' or 'line', not %r" % (p.smart_one_sided_baseline,))
            one_sided_flat = p.smart_one_sided_baseline == 'flat'
        if mode == 'smart' and not debug_plots and traces_2d.dtype == np.float64:
            #numba version of the loop below, bit-identical (get_data always passes float64)
            return _smart1_baselines_2d(np.ascontiguousarray(traces_2d), int(p.smart_bins_away_to_check),
                                        int(p.num_smart_background_ave_bins), float(walk_thr),
                                        float(p.require_peak_within[0]), float(p.require_peak_within[1]), one_sided_flat)

        B, N = traces_2d.shape
        baselines = np.zeros_like(traces_2d)

        for b in range(B):
            trace = traces_2d[b]
            #apply consnant offset of average value of a pad within a time window
            if mode == 'fixed window':
                baselines[b] = np.average(trace[p.num_background_bins[0]:p.num_background_bins[1]])
            #rolling average to each side of a pad
            elif mode == 'convolution':
                baselines[b] = np.convolve(trace, p.background_convolution_kernel, mode='same')
            #in the case of none, just return an array of 0s
            elif mode == 'none':
                baselines[b] = np.zeros(len(trace))
            elif mode == 'smart':
                peak_index = np.argmax(trace)
                #zero traces with peaks outside specified region by returing the trace as the baseline
                if peak_index < p.require_peak_within[0] or peak_index > p.require_peak_within[1]:
                    baselines[b] = trace
                    continue
                #find  start of the peak, defined as where the going bin smart_bins_away_to_check
                #is no longer at least walk_thr below the current bin
                i = peak_index
                while i>0:
                    j = max(0, i - p.smart_bins_away_to_check)
                    if trace[i] < trace[j] + walk_thr:
                        break
                    i -= 1
                peak_start = i
                #find end
                i = peak_index
                while i < len(trace) - 1:
                    j = min(len(trace) - 1, i + p.smart_bins_away_to_check)
                    if trace[i] < trace[j] + walk_thr:
                        break
                    i += 1
                peak_end = i
                if one_sided_flat and peak_end == len(trace) - 1:
                    #no right flank when the walk reaches the last bin (see _smart1_baselines_2d)
                    peak_end = len(trace)
                    if peak_start == 0:
                        baselines[b] = trace #both edges reached: zero the trace
                        continue
                #fit a line through the points just outside the peak region
                xs = np.concatenate([np.arange(max(0, peak_start - p.num_smart_background_ave_bins), peak_start),
                                               np.arange(peak_end, min(peak_end + p.num_smart_background_ave_bins, len(trace)))])
                ys = trace[xs]
                
                n = len(xs)
                n_left = peak_start - max(0, peak_start - p.num_smart_background_ave_bins)
                if one_sided_flat and n > 0 and (n_left == 0 or n_left == n):
                    #flanking bins on one side only: flat baseline at their mean (see _smart1_baselines_2d)
                    slope, offset = 0.0, np.sum(ys) / n
                elif n > 1:
                    sum_x = np.sum(xs)
                    sum_y = np.sum(ys)
                    sum_x2 = np.sum(xs * xs)
                    sum_xy = np.sum(xs * ys)
                    denominator = n * sum_x2 - sum_x * sum_x
                    if denominator != 0:
                        slope = (n * sum_xy - sum_x * sum_y) / denominator
                        offset = (sum_y - slope * sum_x) / n
                    else:
                        slope, offset = 0.0, sum_y / n
                else:
                    slope, offset = 0.0, ys[0] if n > 0 else 0.0
                
                baselines[b] = trace
                x_peak = np.arange(peak_start, peak_end)
                baselines[b, x_peak] = offset + slope * x_peak
            elif mode == 'snip':
                window = p.smart_bins_away_to_check # window size (p is the parameter view, needed for the next trace)
                
                # LLS transform: log(log(y + 1) + 1)
                y = np.array(trace, dtype=float, copy=True)
                baseline = np.log(np.log(y + 1) + 1)
                
                # Pad with reflection to handle boundary bins cleanly
                baseline = np.pad(baseline, pad_width=window, mode='reflect')
                
                for i in range(1, window + 1):
                    temp = (baseline[:-2*i] + baseline[2*i:]) / 2.0
                    baseline[i:-i] = np.minimum(baseline[i:-i], temp)
                
                # Remove padding
                baseline = baseline[window:-window]
                
                # Inverse LLS transform
                baseline = np.exp(np.exp(baseline) - 1) - 1
                
                baselines[b] = baseline
            else:
                assert False #invalid mode

        return baselines

    def get_xyte(self, event_number, threshold=-np.inf, include_veto_pads=False):
        '''
        Returns: xs, ys, ts, es
                 Where each of these is an array s.t. each "pixel" in the in the raw TPC data is represented.
                 eg, (xs[i], ys[i], ts[i]) gives the position of a pad and time bin number,
                 and es[i] gives the charge that arrived at that pad at the given time.
                 Only data where the charge deposition is greater than the threshold is included.
        With cache_enable the result is cached per (event_number, threshold, include_veto_pads), and copies of the
        cached arrays are handed out so that a caller may modify them in place.
        '''
        key = (event_number, threshold, include_veto_pads)
        if self.cache_enable and isinstance(self.cached_event_xyte, tuple) and self.cached_event_xyte == key:
            return tuple(np.array(a, copy=True) for a in self.cached_xyte)
        event_data =  self.get_data(event_number)
        #rows with a pad mapping (minus veto pads unless requested); channels without a mapping are skipped
        rows, pads = self._event_rows(event_data, include_veto_pads)
        if len(rows) > 0:
            xy = self.padxy[pads] #same as chnls_to_xy_coord
            xs = np.repeat(xy[:, 0], self.num_time_bins)
            ys = np.repeat(xy[:, 1], self.num_time_bins)
            es = event_data[rows, FIRST_DATA_BIN:].flatten()
        else:
            xs = np.repeat([], self.num_time_bins)
            ys = np.repeat([], self.num_time_bins)
            es = np.array([]).flatten()

        #make time bins data
        ts = np.tile(np.arange(0, self.num_time_bins), int(len(xs)/self.num_time_bins))

        #apply thresholding
        if threshold != -np.inf:
            xs = xs[es>threshold]
            ys = ys[es>threshold]
            ts = ts[es>threshold]
            es = es[es>threshold]
        if self.cache_enable:
            self.cached_event_xyte = key
            self.cached_xyte = (xs, ys, ts, es)
            return tuple(np.array(a, copy=True) for a in self.cached_xyte)
        return xs, ys, ts, es

    def get_xyze(self, event_number, threshold=-np.inf, include_veto_pads=False):
        '''
        Same as xyte, but scales time bins to get z coordinate. Cached and copied like get_xyte.
        '''
        key = (event_number, threshold, include_veto_pads)
        if self.cache_enable and isinstance(self.cached_event_xyze, tuple) and self.cached_event_xyze == key:
            return tuple(np.array(a, copy=True) for a in self.cached_xyze)
        x,y,t,e = self.get_xyte(event_number, threshold=threshold, include_veto_pads=include_veto_pads)
        to_return = (x,y, t*self.zscale ,e)
        if self.cache_enable:
            self.cached_xyze= to_return
            self.cached_event_xyze = key
            return tuple(np.array(a, copy=True) for a in to_return)
        return to_return

    def get_track_axis(self, event=None, threshold=None, return_all_svd_results=False, xyze=None, return_np=True):
        '''
        Uses SVD on all points above some threshold to get track direction.
        Returns point, unit vector in track direction.
        If threshold==None, uses track length ic threshold.
        
        Either event number must be passed, or xyze which contans tuple of cupy arrays of 
        xyze points. This can be useful to avoid transfering data to GPU twice

        return_np specifies if data should be moved back to cpu from gpu 
        '''
        if threshold == None:
            threshold = self.length_counts_threshold
        if type(event) != type(None) and USE_GPU and not _cusolver_available():
            #cupy cannot load cusolver here, so cp.linalg.svd would raise and the numpy fallback below would run;
            #do exactly that directly (same cupy-centred points, np.linalg.svd), without the failed import per call
            x,y,z,e = self.get_xyze(event, threshold, False)
            points = cp.asarray(np.stack((x, y, z), axis=1))
            points_mean = points.mean(axis=0)
            uu, dd, vv = np.linalg.svd(cp.asnumpy(points - points_mean), full_matrices=False)
            if return_np:
                points_mean = cp.asnumpy(points_mean)
            else:
                dd, vv = cp.array(dd), cp.array(vv)
            if return_all_svd_results:
                return points_mean, dd, vv
            return points_mean, vv
        if type(event) != type(None):
            x,y,z,e = self.get_xyze(event, threshold, False)
            x, y,z,e = cp.array(x), cp.array(y), cp.array(z), cp.array(e)
        else:
            x, y, z, e = xyze
        points = cp.concatenate((x[:, cp.newaxis], y[:, cp.newaxis], z[:, cp.newaxis]), axis=1)
        points_mean = points.mean(axis=0)
        try:
            uu, dd, vv = cp.linalg.svd(points - points_mean, full_matrices=False)
        except Exception as e:
            # print('caught exceptions while trying to perform SVD of event '+str(event))
            # print(e)
            pts_np = cp.asnumpy(points - points_mean)
            uu, dd, vv = np.linalg.svd(pts_np, full_matrices=False)
            uu, dd, vv = cp.array(uu), cp.array(dd), cp.array(vv)

        if return_np:
            points_mean = cp.asnumpy(points_mean)
            vv = cp.asnumpy(vv)
            if return_all_svd_results:
                dd = cp.asnumpy(dd)
                #uu = cp.asnumpy(uu)
        if return_all_svd_results:
            return points_mean, dd, vv
        else:
            return points_mean, vv #vv[0] holds direction vector of 1st priciple component, etc

    
    def get_event_num_bounds(self):
        '''
        Returns first event number, last event number (inclusive). The merger writes [meta][meta] as
        [first event, its timestamp, last event, its timestamp] and then appends one more event after the one it
        recorded as last, so a file holds events first..meta[2]+1 (seen on every e23035 and e25058 file checked).
        The last event is therefore the last consecutive event dataset from meta[2] on. Determined once per
        instance. For an empty file (first > last) the meta values are returned as they are.
        '''
        bounds = self.__dict__.get('_event_bounds_cache')
        if bounds is None:
            meta = self.h5_file['meta']['meta']
            first, last = int(meta[0]), int(meta[2])
            if last >= first:
                get_group = self.h5_file['get']
                for _ in range(8): #the merger appends one event; a bounded walk in case that ever changes
                    if self._event_dataset_path(last + 1, 'data') not in get_group:
                        break
                    last += 1
            bounds = (first, last)
            self.__dict__['_event_bounds_cache'] = bounds
        return bounds

    def get_settings_str(self):
        import json
        settings = {
            'zscale': self.zscale,
            'background_subtract_mode': self.background_subtract_mode,
            'remove_outliers': self.remove_outliers,
            'num_background_bins': self.num_background_bins,
            'length_counts_threshold': self.length_counts_threshold,
            'ic_counts_threshold': self.ic_counts_threshold,
            'data_select_mode': self.data_select_mode,
            'near_peak_window_width': self.near_peak_window_width,
            'num_smart_background_ave_bins': self.num_smart_background_ave_bins,
            'smart_bins_away_to_check': self.smart_bins_away_to_check,
            'smart2_min_bins_in_peak': self.smart2_min_bins_in_peak,
            'smart2_min_sigma': self.smart2_min_sigma,
            'smart2_ransac_percentile': self.smart2_ransac_percentile,
            'require_peak_within': self.require_peak_within,
            'include_counts_on_veto_pads': self.include_counts_on_veto_pads,
            'reconstruct_railed_pads': self.reconstruct_railed_pads,
            'railed_pad_reconstruct_fit_bins': self.railed_pad_reconstruct_fit_bins,
            'apply_gain_match': self.apply_gain_match,
            'asads': self.asads,
            'cobos': self.cobos,
            'pads': self.pads
        }
        
        def serialize(obj):
            if isinstance(obj, (list, tuple, np.ndarray)):
                if isinstance(obj, np.ndarray):
                    obj = obj.tolist()
                return [serialize(i) for i in obj]
            elif isinstance(obj, float) and np.isinf(obj):
                return "inf" if obj > 0 else "-inf"
            return obj
            
        #added only when on, so that the settings hash of every existing configuration (and of the ROOT files
        #processed with it, checked in process_runs._load_run_quantities) is unchanged
        if self.remove_zero_pad:
            settings['remove_zero_pad'] = True
        #the repair method changes the processed data only when railed pads are repaired; 'scipy' (the method before
        #2026-10-23) keeps the hashes of the files processed with it
        if self.reconstruct_railed_pads and self.railed_pad_repair_method != 'scipy':
            settings['railed_pad_repair_method'] = self.railed_pad_repair_method
        #smart (smart1) walk threshold and one-sided rule: added only when a smart baseline is in use and they deviate
        #from the behaviour before 2026-10-28 (walk with ic_counts_threshold, line through one-sided flanking bins),
        #so the hashes of the existing configurations and of their processed files are unchanged
        if self.background_subtract_mode == 'smart' or self.veto_background_subtract_mode == 'smart':
            if self.smart_walk_threshold is not None:
                settings['smart_walk_threshold'] = self.smart_walk_threshold
            if self.smart_one_sided_baseline != 'line':
                settings['smart_one_sided_baseline'] = self.smart_one_sided_baseline
        #likewise the hybrid (veto-pad) baseline settings, only when the veto mode is set
        if self.veto_background_subtract_mode is not None:
            settings['veto_background_subtract_mode'] = self.veto_background_subtract_mode
            for name in VETO_BASELINE_PARAMS:
                value = getattr(self, 'veto_' + name, None)
                if value is not None:
                    settings['veto_' + name] = value

        serialized_settings = {k: serialize(v) for k, v in settings.items()}
        return json.dumps(serialized_settings, sort_keys=True)

    def get_settings_hash(self):
        import hashlib
        settings_str = self.get_settings_str()
        return hashlib.sha256(settings_str.encode()).hexdigest()[:16]


    def get_pad_traces(self, event_number, include_veto_pads=True):
        '''
        returns [pads which fired], [[time series data for first pad], [time series data for 2nd pad], ...]
        Pad numbers are determined from AGET, COBO, and channel number, rather than the pad number written during
        the merging process.
        '''
        event_data =  self.get_data(event_number)
        rows, pads = self._event_rows(event_data, include_veto_pads)
        return list(pads), [event_data[i, FIRST_DATA_BIN:] for i in rows]

    def get_pad_trace_array(self, event_number, include_veto_pads=True):
        '''
        Same pads and traces as get_pad_traces, as arrays: pads (n,), traces (n, num_time_bins). Lets callers
        reduce all traces at once (e.g. traces.sum(axis=1), which equals np.sum of each trace bit for bit).
        '''
        event_data = self.get_data(event_number)
        rows, pads = self._event_rows(event_data, include_veto_pads)
        if len(rows) == 0:
            return pads, np.empty((0, self.num_time_bins))
        return pads, event_data[rows, FIRST_DATA_BIN:]

    def get_num_pads_fired(self, event_number):
        event = self.get_data(event_number)
        return len(event)
    
    def get_track_length_angle(self, event_number, return_np=True):
        '''
        1. Determin track principle axis, and use this to get azimuthal angle
        2. Find point furthest along this axis, in each direction

        Returns: length, angle from z-axis in radians
        '''
        xs, ys, zs, es = self.get_xyze(event_number, self.length_counts_threshold, include_veto_pads=False)
        if len(xs) < 2:
            return 0, 0, 0
        #move points to GPU
        xs, ys, zs, es = cp.array(xs), cp.array(ys), cp.array(zs), cp.array(es)
        track_center, vv = self.get_track_axis(xyze=(xs, ys, zs, es), return_np=False)
        if type(vv) == type(None):
            dxy = dz = 0
            angle = np.inf
        else:
            track_direction = vv[0]/cp.sqrt(cp.sum(vv[0]*vv[0]))
            angle = cp.arctan2(np.sqrt(track_direction[0]**2 + track_direction[1]**2),cp.abs(track_direction[2]))

            points = cp.concatenate((xs[:, cp.newaxis], 
                        ys[:, cp.newaxis], 
                        zs[:, cp.newaxis]), 
                        axis=1)
            rbar = points - track_center
            rdotv = cp.dot(rbar, track_direction)
            first_point = points[cp.argmin(rdotv)]
            last_point = points[cp.argmax(rdotv)]
            dr = last_point - first_point
            dxy, dz = cp.sqrt(dr[0]**2 + dr[1]**2), np.abs(dr[2])
            if return_np:
                dxy, dz, angle = cp.asnumpy(dxy), cp.asnumpy(dz), cp.asnumpy(angle)
        return dxy, dz, angle
    
   
    def get_histogram_arrays(self):
        max_veto_counts_list, pads_railed_list, angle_hist, counts_hist, dxy_hist, dz_hist = [], [], [], [], [], []
        first, last = self.get_event_num_bounds()
        for i in tqdm.tqdm(range(first, last+1)):
            max_veto_counts, dxy, dz, energy, angle, pads_railed = self.process_event(i)
            dxy_hist.append(dxy)
            dz_hist.append(dz)
            counts_hist.append(energy)
            angle_hist.append(angle)
            pads_railed_list.append(pads_railed)
            max_veto_counts_list.append(max_veto_counts)

        return np.array(max_veto_counts_list), np.array(dxy_hist), np.array(dz_hist), np.array(counts_hist), np.array(angle_hist), pads_railed_list

    def process_event(self, event_num):
        '''
        Returns: max veto counts, dxy, dz, energy, angle, pads_railed
        max veto counts is max counts in any single time bin on a single veto pad
        dxy is the track length in the pad plane, dz is the other component of track length
        TODO: currently pads_railed only works if baseline subtraction is off. Fix this
        '''
        should_veto=False
        counts = 0
        pads_railed = []
        max_veto_pad_counts = -np.inf
        for pad, trace in zip(*self.get_pad_traces(event_num)):
            trace_max = np.max(trace)
            if pad in VETO_PADS:
                if  trace_max > max_veto_pad_counts:
                    max_veto_pad_counts = trace_max
            if self.include_counts_on_veto_pads or not pad in VETO_PADS: #don't inlcude veto pad energy
                counts += np.sum(trace[trace>self.ic_counts_threshold])
        
        dxy, dz, angle = self.get_track_length_angle(event_num)
        return max_veto_pad_counts, dxy, dz, counts, angle, self.get_railed_pads(event_num)
    


    def do_gain_match(self, event_numbers:list, save_results:bool, save_path='', show_debug_figures=False, 
                      mode='raw', bounds=(0.1, 10), thresh_to_replace=2000):
        print('gain matching using %d events'%len(event_numbers))
        print('getting list of energy per pad for each event')
        pad_counts = []
        for event in tqdm.tqdm(event_numbers):
            pads, traces = self.get_pad_traces(event, include_veto_pads=False)
            pad_counts.append(np.zeros(NUM_PADS))
            for pad, trace in zip(pads, traces):
                pad_counts[-1][pad] = np.sum(trace)
        pad_counts = np.array(pad_counts)
        all_events_pad_counts = np.sum(pad_counts, axis=0)  

        if show_debug_figures:
            data = {}
            for i in range(len(all_events_pad_counts)):
                if i in self.pad_to_xy_index:
                    data[i] = all_events_pad_counts[i]
            self.show_padplane_image(data, title='without dead pad replacement')

        #replace dead pixels with average of adjacent pixels
        for pad in range(len(all_events_pad_counts)):
            if all_events_pad_counts[pad] < thresh_to_replace and pad in self.pad_to_xy_index:
                #find good adjacent pads to use for average
                adjacent_pads = []
                pad_xy = self.pad_to_xy_index[pad]
                adj_xy = [(pad_xy[0]+d[0], pad_xy[1] + d[1]) 
                          for d in [(0,1), (0,-1), (1,0), (-1,0)]]
                for xy in adj_xy:
                    if xy in self.xy_index_to_pad:
                        adj_pad = self.xy_index_to_pad[xy]
                        if all_events_pad_counts[adj_pad] >= thresh_to_replace:
                            adjacent_pads.append(adj_pad)
                print('replace pad %d with average of pads '%pad, adjacent_pads)
                #do replacement
                if len(adjacent_pads) > 0:
                    for i in range(len(pad_counts)):
                        pad_counts[i, pad] = np.mean(pad_counts[i, adjacent_pads])
                else:
                    print("can't replace, no good adjacent pads")
       
        all_events_pad_counts = np.sum(pad_counts, axis=0)  
        if show_debug_figures:
            data = {}
            for i in range(len(all_events_pad_counts)):
                if i in self.pad_to_xy_index:
                    data[i] = all_events_pad_counts[i]
            self.show_padplane_image(data, title='with dead pad replacement')

        event_adc_counts = np.sum(pad_counts, axis=1)
        print('average event adc counts:', np.mean(event_adc_counts))
        pad_counts /= np.mean(event_adc_counts)
        print('performing minimization')
        pad_counts_dev = cp.asarray(pad_counts)
        num_events = len(pad_counts)
        def raw_objective_function(gains):
            #print(np.shape(pad_counts), np.shape(gains))
            #on the GPU when cupy is available (cp is numpy otherwise): the einsum dominates the Powell search
            adc_counts_in_each_event = cp.einsum('ij,j', pad_counts_dev, cp.asarray(gains))
            return float(np.sqrt(float(cp.asnumpy(cp.sum((adc_counts_in_each_event - 1)**2)))/num_events)*2.355)
        
        num_grid_points = 20
        pad_xy_index = np.array([self.pad_to_xy_index[pad] if pad in self.pad_to_xy_index else (0,0) for pad in range(NUM_PADS)])
        min_xy, max_xy = np.min(pad_xy_index), np.max(pad_xy_index)
        def get_interp_pad_gains(x):
            #first 4 points should be fixed at outside of image
            points = np.linspace(min_xy, max_xy, num_grid_points)
            values = np.reshape(x, (num_grid_points, num_grid_points))
            interpolator = interpolate.RegularGridInterpolator((points, points), values, method='cubic')
            return interpolator(pad_xy_index)
        
        def interp_obj_func(x):
            gains = get_interp_pad_gains(x)
            return raw_objective_function(gains)

        # res = opt.minimize(objective_function, np.ones(NUM_PADS), method="Powell",
        #                    callback=callback, 
        #                    bounds=[[0.5, 2]]*NUM_PADS)
        if mode == 'interp':
            guess = np.ones(num_grid_points**2)
            to_min = interp_obj_func
            min_bounds = [bounds]*num_grid_points**2
        elif mode == 'raw':
            guess = np.ones(NUM_PADS)
            to_min = raw_objective_function
            min_bounds = [bounds]*NUM_PADS


        def callback(intermediate_result):
                print(intermediate_result)
                gains = intermediate_result.x
                print(np.mean(gains), np.std(gains), np.min(gains), np.max(gains))

        res = opt.minimize(to_min, guess, 
                           callback=callback,
                           options={'maxfun':1000000},
                           bounds=min_bounds)
        if mode == 'interp':
            res.orig_x = res.x
            res.x = get_interp_pad_gains(res.x)
        for pad in VETO_PADS:
            res.x[pad] = 1
        res.gain_match_mode = mode
        print(res)
        self.pad_gains = res.x
        if save_results:
            res.events_used = event_numbers
            with open(save_path, 'wb') as f:
                pickle.dump(res, f)



    def show_pad_backgrounds(self, fig_name=None, block=True):
        ave_image = np.zeros(np.shape(self.pad_plane))
        std_image = np.zeros(np.shape(self.pad_plane))
        for pad in self.pad_backgrounds:
            x,y = self.pad_to_xy_index[pad]
            ave, std = self.pad_backgrounds[pad]
            ave_image[x,y] = ave
            std_image[x,y] = std

        fig=plt.figure(fig_name)
        plt.clf()
        ave_ax = plt.subplot(1,2,1)
        ave_ax.set_title('average counts')
        ave_shown = ave_ax.imshow(ave_image, cmap=self.cmap)
        fig.colorbar(ave_shown, ax=ave_ax)

        std_ax = plt.subplot(1,2,2)
        std_ax.set_title('standard deviation')
        std_shown=std_ax.imshow(std_image, cmap=self.cmap)
        fig.colorbar(std_shown, ax=std_ax)
        #plt.colorbar(ax=std_plot)
        #plt.colorbar())
        plt.show(block=block)

    def plot_traces(self, event_num, block=True, fig_name=None):
        '''
        Note: veto pads are plotted as dotted lines
        '''
        plt.figure(fig_name, figsize=(8, 6))
        plt.clf()
        pads, pad_data = self.get_pad_traces(event_num)
        for pad, data in zip(pads, pad_data):
            r = pad/1024*.8
            g = (pad%512)/512*.8
            b = (pad%256)/256*.8
            if pad in VETO_PADS:
                plt.plot(data, '--', color=(r,g,b), label='%d'%pad)
            else:
                plt.plot(data, color=(r,g,b), label='%d'%pad)
        plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), title="Pad #", borderaxespad=0.1, ncol=2)
        # plt.tight_layout()
        plt.show(block=block)

    def plot_3d_traces(self, event_num, threshold=-np.inf, block=True, fig_name=None):
        fig = plt.figure(fig_name, figsize=(6,6))
        plt.clf()
        ax = plt.axes(projection='3d')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_xlim3d(-200, 200)
        ax.set_ylim3d(-200, 200)
        ax.set_zlim3d(0, 400)

        xs, ys, zs, es = self.get_xyze(event_num, threshold=threshold)

        #TODO: make generic, these are P10 values
        calib_point_1 = (0.806, 156745)
        calib_point_2 = (1.679, 320842)
        energy_1, channel_1 = calib_point_1
        energy_2, channel_2 = calib_point_2
        energy_scale_factor = (energy_2 - energy_1) / (channel_2 - channel_1)
        energy_offset = energy_1 - energy_scale_factor * channel_1

        ax.view_init(elev=45, azim=45)
        ax.scatter(xs, ys, zs, c=es, cmap=self.cmap)
        cbar = fig.colorbar(ax.get_children()[0])
        max_veto_counts, dxy, dz, energy, angle, pads_railed = self.process_event(event_num)
        length = np.sqrt(dxy**2 + dz**2)
        plt.title('event %d, total counts=%d \n length=%f mm, angle=%f deg\n # pads railed=%d'%(event_num, energy, 
                                                                                               length,
                                                                                               np.degrees(angle), len(pads_railed)))
        plt.show(block=block)
    
    def get_2d_image(self, data):
        #data should be a dictionary of numbers to put in the image, indexed by pad
        image = -1*np.ones(np.shape(self.pad_plane))
        for pad in self.pad_to_xy_index:
            x,y = self.pad_to_xy_index[pad]
            if pad in data:
                image[y,x] = data[pad]
            else:
                image[y,x] = np.inf
            
        #image[image<0]=0
        return image

    def show_padplane_image(self, data, trace_dict=None, block=False, fig_name=None, title=''):
        '''
        Shows a figure with pad plane displaying "data", above the sum of all "traces" (if not None).
        Clicking on a pad will show the corresponding trace 
        data: Dictionary of pixel brightnesses indexed by pad number
        traces: optional. If passed, sould contain trace to dispay when a pad is clicked. Summed version of this 
                will be shown below the pad plane.
        '''
        image = self.get_2d_image(data)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), constrained_layout=True, num=fig_name, clear=True)
        fig.suptitle(title)

        # Top plot: padplane image
        cmap = matplotlib.cm.get_cmap('viridis').copy()
        cmap.set_under(color='black')
        im = ax1.imshow(image, cmap=cmap, vmin=np.min(image[image>=0]), vmax=np.max(image[image<np.inf]))#norm=colors.LogNorm()
        fig.colorbar(im, ax=ax1)
        ax1.set_title('Padplane Image')

        # Bottom plot: sum of traces
        if trace_dict != None:
            summed_trace = np.sum([trace_dict[pad] for pad in trace_dict], axis=0)
            ax2.plot(summed_trace)
            ax2.set_title('Summed Trace')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('ADC Counts')
            if type(trace_dict) != type(None):
                def onclick(event):
                    x, y = int(np.round(event.xdata)), int(np.round(event.ydata))
                    pad = self.xy_index_to_pad.get((x, y))
                    if pad in trace_dict:
                        plt.figure()
                        plt.title('cobo %d, asad %d, aget %d, chnl %d, pad %d'%(*self.pad_to_chnl[pad], pad))
                        plt.plot(trace_dict[pad])
                        plt.xlabel('Time')
                        plt.ylabel('ADC Counts')
                        plt.show(block=False)

            fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show(block=block)
        return fig
    
    def show_2d_projection(self, event_number, block=True, fig_name=None):
        pads, traces = self.get_pad_traces(event_number)
        trace_dict = {pad: trace for pad, trace in zip(pads, traces)}
        data = {pad:np.sum(trace_dict[pad]) for pad in trace_dict}
        should_veto, dxy, dz, energy, angle, pads_railed_list = self.process_event(event_number)
        length = np.sqrt(dxy**2 + dz**2)
        title='event %d, total counts=%d, length=%f mm, angle=%f, veto=%d'%(event_number, energy, length, np.degrees(angle), should_veto)
        # return self.show_padplane_image(data, trace_dict=trace_dict, block=block, fig_name=fig_name, title=title)
    
        image = self.get_2d_image(data)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), constrained_layout=True, num=fig_name, clear=True)
        fig.suptitle(title)

        # Top plot: padplane image
        cmap = matplotlib.cm.get_cmap('viridis').copy()
        cmap.set_under(color='black')
        im = ax1.imshow(image, cmap=cmap, vmin=np.min(image[image>=0]), vmax=np.max(image[image<np.inf]))#norm=colors.LogNorm()
        fig.colorbar(im, ax=ax1)
        ax1.set_title('Padplane Image')

        # Bottom plot: sum of traces
        if trace_dict != None:
            summed_trace = np.sum([trace_dict[pad] for pad in trace_dict], axis=0)
            ax2.plot(summed_trace)
            ax2.set_title('Summed Trace')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('ADC Counts')
            if type(trace_dict) != type(None):
                def onclick(event):
                    x, y = int(np.round(event.xdata)), int(np.round(event.ydata))
                    pad = self.xy_index_to_pad.get((x, y))
                    if pad in trace_dict:
                        self.show_traces_w_baseline_estimate(event_number, False, pad_to_plot=pad)

            fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show(block=block)
        return fig



    def show_traces_w_baseline_estimate(self, event_num, block=True, fig_name=None, pad_to_plot=None):
        '''
        plots traces without background subtraction, with backgrounds shown as ... lines
        If pad_to_plot is specified, plot only the one pad
        '''
        plt.figure(fig_name)
        plt.clf()
        #read the raw traces of this one event with the baseline subtraction and data selection off; the caches are
        #invalidated before and after, so that neither the processed events of a cached batch nor the raw read of
        #this one leak across the settings change
        old_background_mode = self.background_subtract_mode
        old_cache = self.cache_enable
        old_batch = self.batch_event_cache_size
        old_mode = self.data_select_mode
        self.invalidate_cache()
        self.cache_enable = False
        self.batch_event_cache_size = 1
        self.background_subtract_mode = 'none'
        self.data_select_mode = 'all data'
        try:
            pads, pad_data = self.get_pad_traces(event_num)
        finally:
            self.cache_enable = old_cache
            self.batch_event_cache_size = old_batch
            self.background_subtract_mode = old_background_mode
            self.data_select_mode = old_mode
            self.invalidate_cache()
        for pad, data in zip(pads, pad_data):
            if pad_to_plot is not None and pad != pad_to_plot:
                continue
            r = pad/1024*.8
            g = (pad%512)/512*.8
            b = (pad%256)/256*.8
            if pad in VETO_PADS:
                plt.plot(data, '--', color=(r,g,b), label='%d'%pad)
            else:
                plt.plot(data, color=(r,g,b), label='%d'%pad)
        for pad, data in zip(pads, pad_data):
            if pad_to_plot is not None and pad != pad_to_plot:
                continue
            r = pad/1024*.8
            g = (pad%512)/512*.8
            b = (pad%256)/256*.8
            plt.plot(self.calculate_background(data, pad=pad), '.', color=(r,g,b), label='%d baseline'%pad)
        plt.legend(loc='upper right')
        plt.show(block=block)
