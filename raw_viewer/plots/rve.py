import os
import pickle

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.colors
import ROOT
import numpy as np
from tqdm import tqdm
from scipy import optimize
from matplotlib.ticker import MultipleLocator

from raw_viewer import process_runs
from  raw_viewer import raw_h5_file
from raw_viewer import ddas_interface
from e23035_analysis import e23035_runs

experiment = 'e25058'#_prep_vault'
 
if experiment =='e25058':
    run_range= [215,216,228,255,260,269]#,81,82,83,71,72
if experiment =='e25058_20Mg':
    run_range =[216,228,255,260,269]
# else: #during experiment
#     #run_range = np.concatenate((np.arange(140,219+1), np.arange(263, 279+1)))#60Ga with SCA set ~300 keV
#     #run_range = np.arange(263, 279+1)#60Ga with SCA set ~300 keV & 0.5 us gate delay
#     #run_range = np.arange(220, 263+1)#60Ga with SCA set ~1000 keV
#     #run_range = np.concatenate((np.arange(281,286+1), np.arange(296,301+1)))#59Zn with field cage on
#     run_range = [285]

exclude_runs = []#[1,9, 73, 113]

#get_runs=range(145,151)
#runs=[280] #background after experiment
#runs=range(145, 150+1)#152+1)
#
#srun_range = [299,300,301]
get_runs = []
for run in run_range:
    if not np.isnan(run):
        if run not in exclude_runs and os.path.exists(process_runs.get_h5_path(experiment, run)):
            get_runs.append(run)

get_runs = np.sort(get_runs)
print('get_runs:', get_runs)

load_ddas = False

if load_ddas:
    ddas_runs = e23035_runs.get_DDAS_run_number(get_runs)
    print(ddas_runs)
    print('getting times since beam off for each event')
    times_since_beam_off = []
    for ddas_run, get_run in tqdm(zip(ddas_runs, get_runs)):
        get_ts = process_runs.get_quantity('timestamps', experiment, [get_run])
        times_since_beam_off.append(ddas_interface.get_time_since_beam_off(experiment, ddas_run)[:-1])
        print(get_run, len(times_since_beam_off[-1]), len(get_ts))
    times_since_beam_off = np.concatenate(times_since_beam_off)
else:
    times_since_beam_off = process_runs.get_time_since_beam_off(experiment, get_runs)
veto_thresh =  150
rve_bins = (300, 300)
phist_bins = np.linspace(0, 4, 1001)
alphahist_bins = 100

print('loading stuff')


lengths = process_runs.get_lengths(experiment, get_runs)
#cpp = process_runs.get_quantity('pad_charge', experiment, get_runs)
#veto_counts = process_runs.get_veto_counts(exp, runs)
veto_max = process_runs.get_max_veto_counts(experiment, get_runs)
charge_widths = process_runs.get_quantity('charge_width', experiment,get_runs)
#energy = process_runs.get_gm_ic(experiment, get_runs, pad_gains)
if experiment == 'e23035':
    energy = e23035_runs.get_energy_MeV(get_runs)
else:
    gain_match_path = '/egr/research-tpc/shared/e25058_analysis/fft6_res3.pkl'
    with open(gain_match_path, 'rb') as f:
        gain_match_result = pickle.load(f)
    #pad_gains = gain_match_result.x[:1024]
    pad_gains = gain_match_result.pad_gains
    energy = process_runs.get_gm_ic(experiment, get_runs, pad_gains)
angles = process_runs.get_angle(experiment, get_runs)

##&(angles>np.radians(5))#process_runs.get_outer_ring_counts(experiment, runs)<113#

pads_railed = process_runs.get_quantity('railed_pads', experiment, get_runs)
num_pads_railed = np.array([len(prl) for prl in pads_railed])
if experiment == 'e23035':
    veto_mask = e23035_runs.get_veto_mask(get_runs)
    endpoints = process_runs.get_quantity('endpoints', experiment, get_runs)
    min_z = np.min(endpoints[:,:,2], axis=1)
    veto_mask = veto_mask&(min_z>5)
else:
    veto_mask = (veto_max < veto_thresh) &(times_since_beam_off>0.05)
veto_mask = veto_mask & (num_pads_railed==0) #& (angles>np.radians(20)) 

    
if load_ddas:
    print(len(times_since_beam_off), len(energy))
    assert len(times_since_beam_off) == len(energy)

print('starting plotting')
plt.figure()
plt_mask = veto_mask&(lengths>1)&(lengths<400)#
plt.title('runs: '+str(get_runs))
plt.hist2d(energy[plt_mask], lengths[plt_mask], bins=rve_bins, norm=matplotlib.colors.LogNorm())
plt.colorbar()
plt.xlabel('Energy (MeV)')
plt.ylabel('Range (mm)')
plt.xlim(0, 6)
plt.ylim(0, 140)


# self.poly_selector = matplotlib.widgets.PolygonSelector(ax,self.set_cut_polygon)
# self.poly_selector = matplotlib.widgets.PolygonSelector(ax,self.set_cut_polygon)
# if len(self.rve_cut_verticies) > 0:
#     self.poly_selector.verts = self.rve_cut_verticies
# fig.show()
m = (108.4-32.2)/(3.628-2.25)
m2 = (23.3-35.5)/(1-2.4)
#alpha_mask = veto_mask&(lengths<(energy*m2+35.5 - m2*2.4))&(energy>2.7)#((lengths<(energy*m+32.2 - m*2.25))|((lengths<(energy*m2+35.5 - m2*2.4))&(energy<2.4)))
if experiment == 'e23035':
    alpha_mask = e23035_runs.get_alpha_mask(get_runs)

    m = (159.2-26.2)/(2.81-0.619)
    #proton_mask = veto_mask&(~alpha_mask)&(lengths<(energy*m+26.6 - m*0.619))&(energy>0.3)
    proton_mask = e23035_runs.get_proton_mask(get_runs)
    #proton_mask = veto_mask&(~alpha_mask)&(lengths<(energy*m+26.6 - m*0.619))&(energy>0.95)&(energy<2.2)&(energy>1.5)&(lengths>55)
    # palpha_cut = veto_mask&(energy>1.6)&(energy<1.8)&(lengths>27.5)&(lengths<40)
    # print(np.where(palpha_cut))

    plt.figure()
    plt.hist(energy[proton_mask], phist_bins)
    plt.title('proton energy spectrum, runs: '+str(get_runs))
    plt.xlabel('energy (MeV)')
    plt.ylabel('counts/keV')
    #plt.yscale('log')

    plt.figure()
    plt.title('protons selected in RVE, runs: '+str(get_runs))
    plt.hist2d(energy[plt_mask], lengths[plt_mask], bins=rve_bins, norm=matplotlib.colors.LogNorm())
    plt.scatter(energy[proton_mask], lengths[proton_mask], marker='.', alpha=0.5, color='red')
    plt.colorbar()
    print(str(get_runs) + "has " + str(len(proton_mask[proton_mask])) + " protons")

    timestamps = process_runs.get_quantity('timestamps', experiment, get_runs)
    time_since_last_event = timestamps - np.roll(timestamps, 1)
    time_since_last_event[0] = .15 #we don't actuallly know what this is for the first event, so just putting a typical value for start of window

    plt.figure()
    plt.hist(energy[alpha_mask], alphahist_bins)
    plt.title('alpha energy spectrum, runs: '+str(get_runs))
    plt.xlabel('energy (MeV)')

    plt.figure()
    plt.title('alphas selected in RVE, runs: '+str(get_runs))
    plt.hist2d(energy[plt_mask], lengths[plt_mask], bins=rve_bins, norm=matplotlib.colors.LogNorm())
    plt.scatter(energy[alpha_mask], lengths[alpha_mask], marker='.', alpha=0.5, color='red')
    plt.colorbar()
    print(str(get_runs) + "has " + str(len(alpha_mask[alpha_mask])) + " alphas")


    #TODO: correct for times runs were not instantly started again after previous run ended
    run_t_offset = [0]
    run_ts = []
    for ddas_run in get_runs:
        run_ts.append(process_runs.get_quantity('timestamps', experiment, [ddas_run]))
    for i in range(1, len(get_runs)):
        if run_ts[i][0] <= run_ts[i-1][-1]:
            run_t_offset.append(run_t_offset[-1] + run_ts[i-1][-1])
        else:
            run_t_offset.append(run_t_offset[-1])
    for i in range(len(get_runs)):
        run_ts[i] = run_ts[i] + run_t_offset[i]
    run_ts = np.concatenate(run_ts)

    plt.figure()
    plt.title('alphas')
    tve_bins = (100, 15)
    plt.hist2d(energy[alpha_mask], run_ts[alpha_mask]/3600, bins=tve_bins, norm=matplotlib.colors.LogNorm())
    plt.xlabel('energy (MeV)')
    plt.ylabel('time since start of experiment (hours)')
    plt.colorbar()

if load_ddas:
    plt.figure()
    plt.title('protons')
    tsbo=times_since_beam_off*1e3#process_runs.get_time_since_beam_off(experiment, runs)
    tve_bins = (50, 50)
    plt.hist2d(energy[proton_mask&(tsbo>0)], tsbo[proton_mask&(tsbo>0)], bins=tve_bins)#, norm=matplotlib.colors.LogNorm())
    plt.xlabel('energy (MeV)')
    plt.ylabel('time since beam off (ms)')
    plt.colorbar()

    plt.figure()
    plt.title('alphas')
    tve_bins = (50, 10)
    plt.hist2d(energy[alpha_mask&(tsbo>0)], tsbo[alpha_mask&(tsbo>0)], bins=tve_bins)#, norm=matplotlib.colors.LogNorm())
    plt.xlabel('energy (MeV)')
    plt.ylabel('time since beam off (ms)')
    plt.colorbar()

    plt.figure()
    plt.hist(tsbo, bins=100)
    plt.xlabel('time since beam off')

plt.show(block=False)

# hist = ROOT.TH1D('Ep', 'Ep', 4000, 0, 4)
# hist.Fill(energy[proton_mask])
# hist.Draw()

# np.save('to_fit.npy', energy[proton_mask])

plt.show(block=False)

def fit_decay_exponential(mask, time_bin_edges, guess=(100,100)):
    plt.figure()
    counts, bins, patches = plt.hist(tsbo[mask], time_bin_edges)
    bin_centers = (bins[:-1] + bins[1:])/2
    sigma = np.sqrt(counts)
    sigma[sigma==1] = 1
    to_fit = lambda t, A, tau: A*np.exp(-t/tau)
    #plt.plot(bin_centers, to_fit(bin_centers, *guess))
    popt, pcov = optimize.curve_fit(to_fit, bin_centers, counts, guess, sigma)
    plt.plot(bin_centers, to_fit(bin_centers, *popt))
    plt.figure()
    plt.plot(bin_centers, counts - to_fit(bin_centers,*popt))
    plt.show(block=False)
    print('half life: ', popt[1]*np.log(2), "+/-", np.sqrt(pcov[1,1])*np.log(2), 'ms')
    return popt, pcov

def fit_double_decay_exponential(mask, time_bin_edges, guess=(100,60, 100, 100)):
    plt.figure()
    counts, bins, patches = plt.hist(tsbo[mask], time_bin_edges)
    bin_centers = (bins[:-1] + bins[1:])/2
    sigma = np.sqrt(counts)
    sigma[sigma==0] = 1
    to_fit = lambda t, A1, tau1, A2, tau2: A1*np.exp(-t/tau1) + A2*np.exp(-t/tau2)
    #plt.plot(bin_centers, to_fit(bin_centers, *guess))
    popt, pcov = optimize.curve_fit(to_fit, bin_centers, counts, guess, sigma)
    plt.plot(bin_centers, to_fit(bin_centers, *popt))
    plt.figure()
    plt.plot(bin_centers, counts - to_fit(bin_centers,*popt))
    plt.show(block=False)
    print('half lives: ', popt[1]*np.log(2), "+/-", np.sqrt(pcov[1,1])*np.log(2), 'ms, ',
           popt[3]*np.log(2), "+/-", np.sqrt(pcov[3,3])*np.log(2), 'ms')
    return popt, pcov

rve_cut_select_mask = None
def set_cut_polygon(verticies):
        '''
        verticies: (counts, ranges)
        '''
        global rve_cut_select_mask
        print(verticies)
        rve_cut_verticies = verticies
        selected_rve_path = matplotlib.path.Path(rve_cut_verticies)
        rve_points = np.vstack((energy[plt_mask], lengths[plt_mask])).transpose()
        rve_cut_select_mask = selected_rve_path.contains_points(rve_points)

poly_selector = None
def define_cut_on_gui():
    global poly_selector
    #open a RvE histogram with the current settings
    fig, ax = plt.subplots()
    ax.hist2d(energy[plt_mask], lengths[plt_mask], bins=rve_bins, norm=matplotlib.colors.LogNorm())
    ax.set_xlabel('Energy (in MeV)')
    ax.set_ylabel('range (mm)')
    poly_selector = matplotlib.widgets.PolygonSelector(ax,set_cut_polygon)
    # if len(self.rve_cut_verticies) > 0:
    #     self.poly_selector.verts = self.rve_cut_verticies
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 140)
    fig.show()

evt_runs, evt_nums = process_runs.get_run_and_event_numbers(experiment, get_runs)
def show_selected_event(i):
    run = evt_runs[plt_mask][rve_cut_select_mask][i]
    evt = evt_nums[plt_mask][rve_cut_select_mask][i]
    print('run %d evt %d energy %f MeV'%(run, evt, energy[plt_mask][rve_cut_select_mask][i]))
    show_event(run,evt)

def show_all_selected_tracks():
    """
    Loops through all events in the rve_cut_select_mask, 
    displaying 2D and 3D tracks side-by-side.
    """
    # Calculate how many events are in the mask
    num_selected = np.sum(rve_cut_select_mask)
    print(f"Displaying {num_selected} selected events...| Total Events{num_selected}")

    for i in range(num_selected):
        show_selected_event_side_by_side(i)
        # Wait for 5 seconds before the next iteration
        plt.pause(5)
        plt.close('all') # Clears the plots for the next event

def show_selected_event_side_by_side(i):
    run = evt_runs[plt_mask][rve_cut_select_mask][i]
    evt = evt_nums[plt_mask][rve_cut_select_mask][i]
    en = energy[plt_mask][rve_cut_select_mask][i]
    
    print(f'Displaying: Run {run} | Evt {evt} | Energy {en:.3f} MeV | Index {i}')
    
    # Create the main container window
    #plt.figure(figsize=(14, 7))
    
    h5file = process_runs.get_h5_file(experiment, run)
    
    # Position 1: Left side for 2D
    #plt.subplot(1, 2, 1)
    h5file.show_2d_projection(evt, block=False)
    #plt.show(block=False)
    #plt.title("2D Projection")
    #plt.figure(figsize=(14, 7))
    # Position 2: Right side for 3D
    # Note: If your plot_3d_traces handles its own 'projection=3d', 
    # this call should hopefully populate the right side of the current figure.
    #plt.subplot(1, 2, 2)
    h5file.plot_3d_traces(evt, threshold=h5file.length_counts_threshold, block=False)
    h5file.plot_traces(evt, block=False)
    
    #plt.suptitle(f"Run {run} - Event {evt} ({en:.3f} MeV)")
    #plt.show(block=False)
def show_event(run, evt):
    h5file = process_runs.get_h5_file(experiment, run)
    h5file.show_2d_projection(evt, block=False)
    h5file.plot_3d_traces(evt, threshold=h5file.length_counts_threshold, block=False)
    h5file.plot_traces(evt, block=False)

def plot_energy_spectrum(bins=400):
    """
    Plots a 1D projection of the selected events onto the energy (X) axis.
    Call this after define_cut_on_gui() to see the proton energy spectrum.
    """
    global rve_cut_select_mask
   
    # 1. Verification
    if rve_cut_select_mask is None:
        print("Error: No cut defined. Run define_cut_on_gui() first.")
        return

    # 2. Extract selected energy data
    # plt_mask handles vetoes/timing; rve_cut_select_mask handles your GUI polygon
    selected_energy = energy[plt_mask][rve_cut_select_mask]

    if len(selected_energy) == 0:
        print("Warning: No events found in the current cut.")
        return

    # 3. Plotting
    plt.figure(figsize=(10, 6))
    plt.hist(selected_energy, bins=bins, histtype='step', color='black', lw=1.5)
   
    plt.title(f'Energy Spectrum | From (Run {get_runs})')
    plt.xlabel('Energy (MeV)')
    plt.ylabel('Counts')
   
    # Optional: Log scale helps see small peaks/background
    # plt.yscale('log')
   
    plt.grid(True, alpha=0.3)
    plt.show()
   
    print(f"Projected {len(selected_energy)} events onto the energy axis.")

def plot_energy_spectrum_fixed_bin_width(bin_width=0.01):
    """
    Plots a 1D projection of the selected events onto the energy (X) axis.
    Call this after define_cut_on_gui() to see the proton energy spectrum.
    
    bin_width: The width of each bin in MeV. Default is 0.01 MeV (10 keV).
    """
    global rve_cut_select_mask
   
    # 1. Verification
    if rve_cut_select_mask is None:
        print("Error: No cut defined. Run define_cut_on_gui() first.")
        return

    # 2. Extract selected energy data
    selected_energy = energy[plt_mask][rve_cut_select_mask]
    selected_lengths = lengths[plt_mask][rve_cut_select_mask]
    if len(selected_energy) == 0:
        print("Warning: No events found in the current cut.")
        return

    # 3. Create fixed bins (from 0 to 8 MeV, stepping by bin_width)
    # This guarantees the bins are always the exact same size regardless of the cut
    fixed_bins = np.arange(0, 8.0 + bin_width, bin_width)

    # 4. Plotting
    plt.figure(figsize=(10, 6))
    plt.hist(selected_energy, bins=fixed_bins, histtype='step', color='black', lw=1.5)
   
    plt.title(f'Energy Spectrum | From (Run {get_runs})')
    plt.xlabel('Energy (MeV)')
    plt.ylabel(f'Counts / {bin_width*1000:.0f} keV') # Updates y-axis to show bin width
   
    plt.grid(True, alpha=0.3)
    
    # Optional: Automatically zoom the x-axis to the data you actually selected
    plt.xlim(np.min(selected_energy) - 0.2, np.max(selected_energy) + 0.2)
    
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.hist(selected_lengths, bins=fixed_bins, histtype='step', color='black', lw=1.5)
   
    plt.title(f'Range Spectrum | Selected Region from (Run {get_runs})')
    plt.xlabel('Range (mm)')
    plt.ylabel(f'Counts / {bin_width*1000:.0f} keV') # Updates y-axis to show bin width
   
    plt.grid(True, alpha=0.3)
    
    # Optional: Automatically zoom the x-axis to the data you actually selected
    #plt.xlim(np.min(selected_lengths) - 0.2, np.max(selected_lengths) + 0.2)
    
    plt.show()
   
    print(f"Projected {len(selected_energy)} events onto the energy axis.")
def save_spectrum_data(filename="proton_spectrum"):
    """
    Saves the energy values of events within the current GUI cut.
    Path: wheel274/e25058_analysis/gadget_data_analysis/raw_viewer/plots/spectra
    """
    global rve_cut_select_mask
   
    # 1. Verification
    if rve_cut_select_mask is None:
        print("Error: No cut defined. Run define_cut_on_gui() first.")
        return

    # 2. Set the specific path requested
    # Using absolute path to avoid ambiguity in the interpreter
    save_dir = "/egr/research-tpc/wheel274/e25058_analysis/gadget_data_analysis/raw_viewer/plots/spectra"
   
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")

    # 3. Extract the energy data from the current selection
    # Energy array is filtered by both the global veto/timing mask and your manual polygon
    selected_energy = energy[plt_mask][rve_cut_select_mask]

    if len(selected_energy) == 0:
        print("Warning: Current selection is empty. Nothing to save.")
        return

    # 4. Save the data
    if not filename.endswith(".npy"):
        filename += ".npy"
       
    full_path = os.path.join(save_dir, filename)
    np.save(full_path, selected_energy)
   
    print("-" * 30)
    print(f"SUCCESS: Saved {len(selected_energy)} events.")
    print(f"Location: {full_path}")
    print("-" * 30)
def fit_peak_and_calibrate(true_energy_mev):
    """
    Draws a 2D cut on the RvE plot and a 1D histogram side-by-side. 
    Interactively updates the 1D Gaussian fit and calibration in real-time 
    as the user adjusts the polygon cut.
    """
    # 1. Retrieve the uncalibrated ADC counts
    raw_adc = energy[plt_mask]
    raw_lengths = lengths[plt_mask]
    
    # Create a figure with two subplots side-by-side
    fig, (ax_2d, ax_1d) = plt.subplots(1, 2, figsize=(15, 6))
    fig.canvas.manager.set_window_title('Interactive Peak Fitter')
    
    # 2. Setup the 2D Histogram
    counts, xedges, yedges, im = ax_2d.hist2d(raw_adc, raw_lengths, bins=rve_bins, norm=matplotlib.colors.LogNorm())
    ax_2d.set_title("Draw/adjust polygon here")
    ax_2d.set_xlabel("Integrated Charge (ADC)")
    ax_2d.set_ylabel("Range (mm)")
    
    ax_1d.set_title("1D Fit will appear here")
    ax_1d.set_xlabel("Integrated Charge (ADC)")
    ax_1d.set_ylabel("Counts")
    
    # We need a dictionary to store the final results so we can return them after the window closes
    results = {'centroid': None, 'calib_factor': None}
    
    # 3. The Interactive Callback
    def onselect(verts):
        path = matplotlib.path.Path(verts)
        # Find all points inside the drawn polygon
        inside = path.contains_points(np.column_stack([raw_adc, raw_lengths]))
        gated_adc = raw_adc[inside]
        
        # Clear the 1D plot for the new updated data
        ax_1d.clear()
        
        if len(gated_adc) < 10 or np.std(gated_adc) == 0:
            ax_1d.set_title("Not enough points selected for a fit.")
            fig.canvas.draw_idle()
            return
            
        # Histogram the 1D data
        hist, bin_edges = np.histogram(gated_adc, bins=50)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        def gaussian(x, amp, mu, sigma):
            return amp * np.exp(-(x - mu)**2 / (2 * sigma**2))
        
        initial_guess = [np.max(hist), np.mean(gated_adc), np.std(gated_adc)]
        
        try:
            # Run the fit
            popt, _ = optimize.curve_fit(gaussian, bin_centers, hist, p0=initial_guess, maxfev=100000)
            amp, centroid, sigma = popt
            
            # Update the stored results
            calibration_factor = true_energy_mev / centroid
            results['centroid'] = centroid
            results['calib_factor'] = calibration_factor
            
            # Plot the new histogram and fit line
            ax_1d.hist(gated_adc, bins=50, alpha=0.6, label='Gated Data')
            x_fit = np.linspace(np.min(gated_adc), np.max(gated_adc), 200)
            ax_1d.plot(x_fit, gaussian(x_fit, *popt), 'r-', linewidth=2, label=f'Centroid: {centroid:.1f}')
            
            ax_1d.set_title(f"Calib Factor: {calibration_factor:.6e} MeV/ADC")
            ax_1d.set_xlabel("Integrated Charge (ADC)")
            ax_1d.set_ylabel("Counts")
            ax_1d.legend()
            
        except Exception as e:
            ax_1d.set_title("Fit failed to converge.")
            print(f"Interactive fit failed: {e}")
            
        # Force matplotlib to redraw the figure with the new 1D plot
        fig.canvas.draw_idle()

    # Initialize the interactive polygon tool
    selector = matplotlib.widgets.PolygonSelector(ax_2d, onselect, props=dict(color='r', alpha=0.5))
    
    print("Interactive window opened. Adjust the polygon. Close the window to save the calibration.")
    plt.show(block=True) # Execution pauses here until the user closes the plot window
    
    # 4. Print and return the final stored results when the window is closed
    if results['centroid'] is not None:
        print(f"\n--- Final Calibration Results ---")
        print(f"Target Energy: {true_energy_mev} MeV")
        print(f"Centroid (ADC): {results['centroid']:.2f}")
        print(f"Gain Match Factor: {results['calib_factor']:.6e} MeV/ADC")
        print(f"To apply this calibration, multiply your raw ADC array by {results['calib_factor']:.6e}")
        return results['centroid'], results['calib_factor']
    else:
        print("No valid fit was made. Returning None.")
        return None, None
    def range_vs_energy(time_interval):
        print('starting plotting')
        plt.figure()
        plt_mask = veto_mask&(lengths>1)&(lengths<400)
        plt.title('runs: '+str(get_runs) + 'time interval'+ str(time_interval) + "mins")
        plt.hist2d(energy[plt_mask], lengths[plt_mask], bins=rve_bins, norm=matplotlib.colors.LogNorm())
        plt.colorbar()
        plt.xlabel('energy (MeV)')
        plt.ylabel('range (mm)')
        print(ts)

def get_cobos_present(experiment, run, num_events_to_check = 1000):
    num_asads_expected = 4 #terminate once 4 asads have been seen
    h5 = process_runs.get_h5_file(experiment, run)
    h5.background_subtract_mode = 'none' #disable background subtrction since only care about channels
    cobos = []
    evt_bounds = h5.get_event_num_bounds()
    for i in range(evt_bounds[0], min(evt_bounds[0] + num_events_to_check, evt_bounds[1])):
        l = np.unique(h5.get_data(i)[:,0])
        for j in l:
            if j not in cobos:
                cobos.append(j)
        if len(cobos) >= num_asads_expected:
            return np.sort(cobos)
    return np.sort(cobos)
def average():
    global rve_cut_select_mask
    # 1. Verification
    if rve_cut_select_mask is None:
        print("Error: No cut defined. Run define_cut_on_gui() first.")
        return

    # 2. Extract selected energy data
    selected_energy = energy[plt_mask][rve_cut_select_mask]
    selected_lengths = lengths[plt_mask][rve_cut_select_mask]
    average_energy = selected_energy.mean()
    average_length = selected_lengths.mean()
    print(average_energy)
    print(average_length)
def get_total_run_duration(runs=None, exp=None):
    """
    Calculate the total duration of all selected runs.

    Duration for each run:
        last event timestamp - first event timestamp

    Returns
    -------
    total_seconds : float
        Sum of the durations of all runs.
    """

    if exp is None:
        exp = experiment

    if runs is None:
        runs = get_runs

    total_seconds = 0.0

    print("\nRun durations")
    print("-" * 45)

    for run in runs:
        timestamps = process_runs.get_quantity(
            'timestamps',
            exp,
            [int(run)]
        )

        timestamps = np.asarray(timestamps, dtype=float)
        timestamps = timestamps[np.isfinite(timestamps)]

        if len(timestamps) < 2:
            print(f"Run {run}: insufficient timestamps")
            continue

        run_duration = timestamps[-1] - timestamps[0]
        total_seconds += run_duration

        print(
            f"Run {run}: "
            f"{run_duration:.2f} seconds "
            f"({run_duration / 60:.2f} minutes)"
        )

    print("-" * 45)
    print(f"Total duration: {total_seconds:.2f} seconds")
    print(f"Total duration: {total_seconds / 60:.2f} minutes")
    print(f"Total duration: {total_seconds / 3600:.3f} hours")

    return total_seconds
def fit_peak(
        peak_guess,
        fit_window=None,
        fit_half_width=0.15,
        bin_width=0.005,
        sigma_guess=0.030,
        particle_name="events",
        rve_xlim=None,
        rve_ylim=(0, 140)
    ):
    """
    Draw a polygon cut on the Range-vs-Energy plot and fit the
    selected energy peak with:

        Gaussian peak + linear background

    The fitted Gaussian area is the background-subtracted number
    of detected particles in the peak.

    Parameters
    ----------
    peak_guess : float
        Approximate peak energy in MeV.
        Example: 0.800 for an 800-keV peak.

    fit_window : tuple or None
        Explicit fit limits in MeV, such as (0.65, 0.95).
        If None, peak_guess +/- fit_half_width is used.

    fit_half_width : float
        Half-width of fit window when fit_window is None.

    bin_width : float
        Energy histogram bin width in MeV.
        0.005 MeV = 5 keV.

    sigma_guess : float
        Initial Gaussian sigma estimate in MeV.

    particle_name : str
        Label such as "protons", "alphas", or "events".

    rve_xlim : tuple or None
        X-axis limits for Range-vs-Energy plot.
        If None, automatically zooms around the fit region.

    rve_ylim : tuple
        Range-axis limits in mm.

    Returns
    -------
    result : dict or None
        Dictionary containing peak area, centroid, width,
        uncertainties, fit quality, and polygon vertices.
    """

    from matplotlib.widgets import PolygonSelector
    from math import erf, sqrt

    # ----------------------------------------------------------
    # Determine the energy fit limits
    # ----------------------------------------------------------
    if fit_window is None:
        fit_low = peak_guess - fit_half_width
        fit_high = peak_guess + fit_half_width
    else:
        fit_low, fit_high = fit_window

    if fit_low >= fit_high:
        raise ValueError("fit_window must satisfy fit_low < fit_high.")

    if not fit_low <= peak_guess <= fit_high:
        raise ValueError(
            "peak_guess must be inside the requested fit window."
        )

    if bin_width <= 0:
        raise ValueError("bin_width must be positive.")

    if sigma_guess <= 0:
        raise ValueError("sigma_guess must be positive.")

    # ----------------------------------------------------------
    # Get events that already pass your normal rve.py cuts
    # ----------------------------------------------------------
    plot_energy = np.asarray(energy[plt_mask], dtype=float)
    plot_length = np.asarray(lengths[plt_mask], dtype=float)

    finite_mask = (
        np.isfinite(plot_energy) &
        np.isfinite(plot_length)
    )

    plot_energy = plot_energy[finite_mask]
    plot_length = plot_length[finite_mask]

    if len(plot_energy) == 0:
        print("No events pass plt_mask.")
        return None

    if rve_xlim is None:
        rve_xlim = (
            max(0, fit_low - 0.25),
            fit_high + 0.25
        )

    points = np.column_stack(
        (plot_energy, plot_length)
    )

    final_result = {
        "success": False
    }

    # ----------------------------------------------------------
    # Create interactive figure
    # ----------------------------------------------------------
    fig, (ax_rve, ax_fit) = plt.subplots(
        1,
        2,
        figsize=(15, 6)
    )

    try:
        fig.canvas.manager.set_window_title(
            f"Fit peak near {peak_guess * 1000:.0f} keV"
        )
    except Exception:
        pass

    ax_rve.hist2d(
        plot_energy,
        plot_length,
        bins=rve_bins,
        norm=matplotlib.colors.LogNorm()
    )

    ax_rve.set_title(
        "Draw a polygon around the particle band"
    )
    ax_rve.set_xlabel("Energy (MeV)")
    ax_rve.set_ylabel("Range (mm)")
    ax_rve.set_xlim(*rve_xlim)
    ax_rve.set_ylim(*rve_ylim)

    ax_fit.set_title(
        "Complete the polygon to fit the peak"
    )
    ax_fit.set_xlabel("Energy (MeV)")
    ax_fit.set_ylabel(
        f"Counts / {bin_width * 1000:.1f} keV"
    )
    ax_fit.set_xlim(fit_low, fit_high)

    # ----------------------------------------------------------
    # Run whenever the polygon is completed or changed
    # ----------------------------------------------------------
    def onselect(vertices):

        selected_path = Path(vertices)
        inside_polygon = selected_path.contains_points(points)

        selected_energy = plot_energy[inside_polygon]

        in_fit_window = (
            (selected_energy >= fit_low) &
            (selected_energy <= fit_high)
        )

        fit_energy = selected_energy[in_fit_window]

        ax_fit.clear()
        ax_fit.set_xlabel("Energy (MeV)")
        ax_fit.set_ylabel(
            f"Counts / {bin_width * 1000:.1f} keV"
        )
        ax_fit.set_xlim(fit_low, fit_high)

        if len(fit_energy) < 20:
            ax_fit.set_title(
                "Not enough events in the polygon and fit window"
            )
            fig.canvas.draw_idle()
            return

        # ------------------------------------------------------
        # Create histogram with fixed-width bins
        # ------------------------------------------------------
        bin_edges = np.arange(
            fit_low,
            fit_high + bin_width,
            bin_width
        )

        if len(bin_edges) < 7:
            ax_fit.set_title(
                "Fit window has too few histogram bins"
            )
            fig.canvas.draw_idle()
            return

        counts, bin_edges = np.histogram(
            fit_energy,
            bins=bin_edges
        )

        bin_centers = (
            bin_edges[:-1] + bin_edges[1:]
        ) / 2

        actual_bin_width = bin_edges[1] - bin_edges[0]

        # ------------------------------------------------------
        # Gaussian area parameterization
        #
        # area is the total Gaussian integral in counts.
        #
        # bg_left and bg_right are background count densities
        # in counts/MeV at the edges of the fit window.
        # ------------------------------------------------------
        def model(
                x,
                area,
                centroid,
                sigma,
                bg_left,
                bg_right
            ):

            gaussian = (
                area *
                actual_bin_width /
                (np.sqrt(2 * np.pi) * sigma) *
                np.exp(
                    -0.5 *
                    ((x - centroid) / sigma) ** 2
                )
            )

            position = (
                (x - fit_low) /
                (fit_high - fit_low)
            )

            background_density = (
                bg_left * (1 - position) +
                bg_right * position
            )

            background = (
                background_density *
                actual_bin_width
            )

            return gaussian + background

        def gaussian_component(
                x,
                area,
                centroid,
                sigma
            ):

            return (
                area *
                actual_bin_width /
                (np.sqrt(2 * np.pi) * sigma) *
                np.exp(
                    -0.5 *
                    ((x - centroid) / sigma) ** 2
                )
            )

        def background_component(
                x,
                bg_left,
                bg_right
            ):

            position = (
                (x - fit_low) /
                (fit_high - fit_low)
            )

            background_density = (
                bg_left * (1 - position) +
                bg_right * position
            )

            return (
                background_density *
                actual_bin_width
            )

        # ------------------------------------------------------
        # Initial parameter estimates
        # ------------------------------------------------------
        number_of_edge_bins = max(
            2,
            len(counts) // 5
        )

        left_background_guess = (
            np.median(
                counts[:number_of_edge_bins]
            ) /
            actual_bin_width
        )

        right_background_guess = (
            np.median(
                counts[-number_of_edge_bins:]
            ) /
            actual_bin_width
        )

        background_guess_per_bin = np.linspace(
            left_background_guess,
            right_background_guess,
            len(counts)
        ) * actual_bin_width

        excess_counts = np.clip(
            counts - background_guess_per_bin,
            0,
            None
        )

        area_guess = max(
            float(np.sum(excess_counts)),
            1.0
        )

        initial_parameters = [
            area_guess,
            peak_guess,
            sigma_guess,
            max(left_background_guess, 0),
            max(right_background_guess, 0)
        ]

        lower_bounds = [
            0,
            fit_low,
            actual_bin_width / 3,
            0,
            0
        ]

        upper_bounds = [
            np.inf,
            fit_high,
            (fit_high - fit_low) / 2,
            np.inf,
            np.inf
        ]

        # Poisson uncertainties for histogram bins
        count_errors = np.sqrt(
            np.maximum(counts, 1)
        )

        try:
            parameters, covariance = optimize.curve_fit(
                model,
                bin_centers,
                counts,
                p0=initial_parameters,
                sigma=count_errors,
                absolute_sigma=True,
                bounds=(lower_bounds, upper_bounds),
                maxfev=100000
            )

        except Exception as error:
            ax_fit.step(
                bin_centers,
                counts,
                where="mid"
            )

            ax_fit.set_title(
                f"Fit failed: {error}"
            )

            print("Peak fit failed:", error)
            fig.canvas.draw_idle()
            return

        (
            fitted_area,
            fitted_centroid,
            fitted_sigma,
            fitted_bg_left,
            fitted_bg_right
        ) = parameters

        parameter_errors = np.sqrt(
            np.diag(covariance)
        )

        area_error = parameter_errors[0]
        centroid_error = parameter_errors[1]
        sigma_error = parameter_errors[2]

        fitted_fwhm = (
            2 * np.sqrt(2 * np.log(2)) *
            fitted_sigma
        )

        fwhm_error = (
            2 * np.sqrt(2 * np.log(2)) *
            sigma_error
        )

        # Fraction of the fitted Gaussian inside fit window
        lower_argument = (
            (fit_low - fitted_centroid) /
            (sqrt(2) * fitted_sigma)
        )

        upper_argument = (
            (fit_high - fitted_centroid) /
            (sqrt(2) * fitted_sigma)
        )

        fraction_inside_window = (
            0.5 *
            (
                erf(upper_argument) -
                erf(lower_argument)
            )
        )

        area_inside_window = (
            fitted_area *
            fraction_inside_window
        )

        background_counts = (
            0.5 *
            (fitted_bg_left + fitted_bg_right) *
            (fit_high - fit_low)
        )

        # Fit quality
        fitted_bin_counts = model(
            bin_centers,
            *parameters
        )

        chi_squared = np.sum(
            (
                (counts - fitted_bin_counts) /
                count_errors
            ) ** 2
        )

        degrees_of_freedom = (
            len(counts) - len(parameters)
        )

        if degrees_of_freedom > 0:
            reduced_chi_squared = (
                chi_squared /
                degrees_of_freedom
            )
        else:
            reduced_chi_squared = np.nan

        # ------------------------------------------------------
        # Plot fit results
        # ------------------------------------------------------
        smooth_energy = np.linspace(
            fit_low,
            fit_high,
            1000
        )

        ax_fit.step(
            bin_centers,
            counts,
            where="mid",
            label="Selected data"
        )

        ax_fit.plot(
            smooth_energy,
            model(
                smooth_energy,
                *parameters
            ),
            label="Gaussian + background"
        )

        ax_fit.plot(
            smooth_energy,
            background_component(
                smooth_energy,
                fitted_bg_left,
                fitted_bg_right
            ),
            linestyle="--",
            label="Background"
        )

        ax_fit.plot(
            smooth_energy,
            gaussian_component(
                smooth_energy,
                fitted_area,
                fitted_centroid,
                fitted_sigma
            ),
            linestyle=":",
            label="Gaussian peak"
        )

        ax_fit.axvline(
            fitted_centroid,
            linestyle=":"
        )

        ax_fit.set_xlabel("Energy (MeV)")
        ax_fit.set_ylabel(
            f"Counts / {actual_bin_width * 1000:.1f} keV"
        )

        ax_fit.set_xlim(fit_low, fit_high)

        ax_fit.set_title(
            f"Area = {fitted_area:.0f} ± "
            f"{area_error:.0f} {particle_name}"
        )

        ax_fit.legend()

        # ------------------------------------------------------
        # Save result from latest polygon
        # ------------------------------------------------------
        final_result.clear()

        final_result.update({
            "success": True,

            # Main result
            "area": fitted_area,
            "area_error": area_error,

            # Peak position and width
            "centroid_mev": fitted_centroid,
            "centroid_error_mev": centroid_error,
            "sigma_mev": fitted_sigma,
            "sigma_error_mev": sigma_error,
            "fwhm_mev": fitted_fwhm,
            "fwhm_error_mev": fwhm_error,

            # Additional count information
            "area_inside_fit_window": area_inside_window,
            "fraction_inside_fit_window": fraction_inside_window,
            "background_counts_in_window": background_counts,
            "selected_events": len(selected_energy),
            "raw_events_in_fit_window": len(fit_energy),

            # Fit quality
            "chi_squared": chi_squared,
            "degrees_of_freedom": degrees_of_freedom,
            "reduced_chi_squared": reduced_chi_squared,

            # Settings
            "peak_guess_mev": peak_guess,
            "fit_low_mev": fit_low,
            "fit_high_mev": fit_high,
            "bin_width_mev": actual_bin_width,
            "runs": np.asarray(get_runs).copy(),
            "cut_vertices": np.asarray(vertices).copy()
        })

        print("\n" + "=" * 60)
        print(
            f"Peak fit near "
            f"{peak_guess * 1000:.1f} keV"
        )
        print(f"Runs: {get_runs}")
        print(
            f"Events inside polygon: "
            f"{len(selected_energy)}"
        )
        print(
            f"Raw events in fit window: "
            f"{len(fit_energy)}"
        )
        print("-" * 60)
        print(
            f"Centroid: "
            f"{fitted_centroid * 1000:.2f} ± "
            f"{centroid_error * 1000:.2f} keV"
        )
        print(
            f"Sigma: "
            f"{fitted_sigma * 1000:.2f} ± "
            f"{sigma_error * 1000:.2f} keV"
        )
        print(
            f"FWHM: "
            f"{fitted_fwhm * 1000:.2f} ± "
            f"{fwhm_error * 1000:.2f} keV"
        )
        print(
            f"Background in fit window: "
            f"{background_counts:.1f} events"
        )
        print(
            f"Reduced chi-squared: "
            f"{reduced_chi_squared:.3f}"
        )
        print("-" * 60)
        print(
            f"BACKGROUND-SUBTRACTED AREA: "
            f"{fitted_area:.0f} ± "
            f"{area_error:.0f} {particle_name}"
        )
        print("=" * 60)

        fig.canvas.draw_idle()

    selector = PolygonSelector(
        ax_rve,
        onselect
    )

    print("\nInstructions:")
    print(
        "1. Draw a polygon following the particle band."
    )
    print(
        "2. Include the peak and some energy region "
        "on both sides for background fitting."
    )
    print(
        "3. Adjust the polygon until the fit looks reasonable."
    )
    print(
        "4. Close the figure to return the final result."
    )

    plt.tight_layout()
    plt.show(block=True)

    # Keep selector alive until the window closes
    _ = selector

    if final_result.get("success", False):
        return final_result

    print("No successful peak fit was completed.")
    return None

def plot_energy_spectrum_fixed_bin_width1(
        bin_width=0.01,
        title=" Proton Energy Spectrum",
        energy_range=(0.0, 8.0),
        xlim=None,
        font_scale=4.0,
        line_width=3.0,
        log_y=False,
        save_path=None
    ):
    global rve_cut_select_mask

    # Check that a Range-vs-Energy polygon cut exists
    if rve_cut_select_mask is None:
        print("Error: No cut has been defined.")
        print("Run define_cut_on_gui() and draw a polygon first.")
        return None

    # plt_mask is applied before the polygon selector in define_cut_on_gui()
    masked_energy = np.asarray(energy[plt_mask], dtype=float)
    cut_mask = np.asarray(rve_cut_select_mask, dtype=bool)

    # Make sure the polygon mask still matches the currently loaded data
    if len(cut_mask) != len(masked_energy):
        print("Error: The polygon cut does not match the current data.")
        print("Run define_cut_on_gui() again and redraw the cut.")
        return None

    # Energy values of events inside the polygon
    selected_energy = masked_energy[cut_mask]

    # Remove NaN and infinite values
    selected_energy = selected_energy[
        np.isfinite(selected_energy)
    ]

    if len(selected_energy) == 0:
        print("Warning: No events were found inside the current cut.")
        return None

    energy_min, energy_max = energy_range

    if energy_min >= energy_max:
        raise ValueError(
            "energy_range must satisfy minimum < maximum."
        )

    if bin_width <= 0:
        raise ValueError("bin_width must be greater than zero.")

    # Include the upper endpoint
    fixed_bins = np.arange(
        energy_min,
        energy_max + bin_width,
        bin_width
    )

    # Count events using the same bins that will be plotted
    counts, bin_edges = np.histogram(
        selected_energy,
        bins=fixed_bins
    )

    # Large slide-friendly font sizes
    title_font = 10.0 * font_scale
    label_font = 10.0 * font_scale
    tick_font = 8.0 * font_scale

    fig, ax = plt.subplots(figsize=(16, 10))

    ax.hist(
        selected_energy,
        bins=fixed_bins,
        histtype="step",
        linewidth=line_width,
        color="black"
    )

    ax.set_title(
        title,
        fontsize=title_font,
        pad=24
    )

    ax.set_xlabel(
        "Energy (MeV)",
        fontsize=label_font,
        labelpad=18
    )

    ax.set_ylabel(
        f"Counts / {bin_width * 1000:.0f} keV",
        fontsize=label_font,
        labelpad=18
    )

    # Enlarge both tick numbers and tick marks
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=tick_font,
        width=2.5,
        length=10,
        pad=10
    )

    ax.tick_params(
        axis="both",
        which="minor",
        width=2.0,
        length=6
    )

    # Enlarge scientific-notation multiplier, such as ×10^4
    ax.xaxis.get_offset_text().set_fontsize(tick_font)
    ax.yaxis.get_offset_text().set_fontsize(tick_font)

    # Use fewer major tick labels so the enlarged numbers do not overlap
    ax.xaxis.set_major_locator(MultipleLocator(0.25))
    ax.locator_params(axis="y", nbins=6)

    # Make the plot border more visible on a slide
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)

    ax.grid(
        True,
        alpha=0.25,
        linewidth=1.5
    )

    if log_y:
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom=0)

    # Automatically zoom around the selected spectrum
    if xlim is None:
        data_width = np.ptp(selected_energy)

        padding = max(
            0.05,
            0.08 * data_width
        )

        display_low = max(
            energy_min,
            np.min(selected_energy) - padding
        )

        display_high = min(
            energy_max,
            np.max(selected_energy) + padding
        )

        # Avoid identical limits for an extremely narrow selection
        if display_high <= display_low:
            display_low = max(
                energy_min,
                np.min(selected_energy) - 0.05
            )
            display_high = min(
                energy_max,
                np.max(selected_energy) + 0.05
            )

        ax.set_xlim(display_low, display_high)

    else:
        if xlim[0] >= xlim[1]:
            raise ValueError(
                "xlim must satisfy minimum < maximum."
            )

        ax.set_xlim(xlim)

    fig.tight_layout()

    if save_path is not None:
        save_directory = os.path.dirname(save_path)

        if save_directory:
            os.makedirs(
                save_directory,
                exist_ok=True
            )

        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        print(f"Figure saved to: {save_path}")

    plt.show()

    print(
        f"Projected {len(selected_energy)} selected events "
        f"onto the energy axis."
    )

    return {
        "selected_energy": selected_energy,
        "counts": counts,
        "bin_edges": bin_edges,
        "total_selected_events": len(selected_energy),
        "bin_width_mev": bin_width
    }