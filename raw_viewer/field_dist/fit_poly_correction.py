import copy
import pickle

from tqdm import tqdm
import numpy as np
import matplotlib.pylab as plt
import matplotlib.colors
import cupy as cp
from scipy import optimize

from raw_viewer.field_dist.polynomial_correction import PolynomialCorrection
from raw_viewer import ddas_interface
from raw_viewer import process_runs
from e23035_analysis import e23035_runs
from track_fitting import srim_interface, build_sim

experiment = 'e23035'
#get_runs = range(145, 150+1)
get_runs = [299, 300, 301]
ddas_runs = e23035_runs.get_DDAS_run_number(get_runs)

max_index = -1

#load stopping power table for protons and use it to get expected track lengths
#I assume ionization fraction ~1, which is a good assumption over 100 keV
material='P10'
stopping_power_path = 'track_fitting/stopping_powers/%s_in_%s.txt'%('1H', material)
proton_srim_table = srim_interface.SRIM_Table(stopping_power_path, build_sim.get_gas_density('e23035', get_runs[0]))
stopping_power_path = 'track_fitting/stopping_powers/%s_in_%s.txt'%('4He', material)
alpha_srim_table = srim_interface.SRIM_Table(stopping_power_path, build_sim.get_gas_density('e23035', get_runs[0]))

print('loading GET data')
uncorrected_lengths =e23035_runs.get_length_mm(get_runs)
uncorrected_widths = process_runs.get_quantity('charge_width', experiment, get_runs)
uncorrected_angles = process_runs.get_angle(experiment, get_runs)
energy_MeV = e23035_runs.get_energy_MeV(get_runs)
endpoints = process_runs.get_quantity('endpoints', experiment, get_runs)

print('loading times from DDAS')
times_since_beam_off = []
for ddas_run in tqdm(ddas_runs):
    #exclude last event since it seems to be dumped
    times_since_beam_off.append(ddas_interface.get_time_since_beam_off(experiment, ddas_run)[:-1])
times_since_beam_off = np.concatenate(times_since_beam_off)

if False:
    get_times = process_runs.get_quantity('timestamps', experiment, get_runs)
    ddas_es, ddas_ts, ddas_ms = ddas_interface.extract_get_event_data(experiment, ddas_runs)
    ddas_ts = ddas_ts[:, ddas_interface.CH_MAP.GET_TRIG_ACCEPTED]/1e9
    ddas_ts = ddas_ts[ddas_ts>0]
    plt.plot(get_times-get_times[0])
    plt.plot(ddas_ts-ddas_ts[0])
    plt.show()


#select events to use for fit
#event_mask = e23035_runs.get_proton_mask(get_runs)&(energy_MeV>2.2)
veto_mask = e23035_runs.get_veto_mask(get_runs)
expected_proton_length = proton_srim_table.get_stopping_distance(energy_MeV)
event_mask = veto_mask & (uncorrected_lengths < expected_proton_length+15) & (uncorrected_lengths > expected_proton_length-20) & (energy_MeV>1.5)

times_since_beam_off = times_since_beam_off[:max_index]
energy_MeV = energy_MeV[:max_index]
event_mask = event_mask[:max_index]
veto_mask = veto_mask[:max_index]
uncorrected_lengths = uncorrected_lengths[:max_index]
endpoints = endpoints[:max_index]
uncorrected_widths = uncorrected_widths[:max_index]
uncorrected_angles = uncorrected_angles[:max_index]


num_selected_events = len(energy_MeV[event_mask])
print('number of selected events', num_selected_events)

#setup a polynomial correction for the starting and ending points
gpu_id_to_use = 1
poly_correction1 = PolynomialCorrection(gpu_id_to_use)
poly_correction1.set_data(endpoints[event_mask,0,:], uncorrected_widths[event_mask], uncorrected_angles[event_mask], times_since_beam_off[event_mask])
poly_correction2 = PolynomialCorrection(gpu_id_to_use)
poly_correction2.set_data(endpoints[event_mask,1,:], uncorrected_widths[event_mask], uncorrected_angles[event_mask], times_since_beam_off[event_mask])

#width -> width + sum_ijk a_ijk*angle^i*width^j*time^k
poly_correction1.width_ijk = poly_correction2.width_ijk = [(1,0,0), (2,0,0), (0,0,1), (1,0,1), (1,1,0)]
#set up corrections for r, theta, and z
#r => r + sum_(ijklm) a_ijklmn r^i sin(theta)^j cos(theta)^k z^l corrected_width^m time^n
def build_ijklmn(r_power_max, sintheta_power_max, costheta_power_max, z_power_max, width_power_max, time_power_max, poly_order):
    to_return = []
    for i in range(r_power_max+1):
        for j in range(sintheta_power_max+1):
            for k in range(costheta_power_max+1):
                for l in range(z_power_max+1):
                    for m in range(width_power_max+1):
                        for n in range(time_power_max+1):
                            if (i,j,k,l,m,n) != (0,0,0,0,0,0) and i+j+k+l+m+n <= poly_order:
                                to_return.append((i,j,k,l,m,n))
    return to_return

poly_correction1.r_ijklmn = poly_correction2.r_ijklmn = build_ijklmn(2,2,0,0,2,2, 2)
poly_correction1.theta_ijklmn = poly_correction2.theta_ijklmn = build_ijklmn(2,2,0,0,2,2, 2)
poly_correction1.z_ijklmn = poly_correction2.z_ijklmn = build_ijklmn(2,2,0,0,2,2, 2)

def apply_params(params:np.ndarray, fit_beam_center:bool, fit_width:bool, fit_position:bool):
    index = 0
    if fit_beam_center:
        poly_correction1.beam_spot_center = poly_correction2.beam_spot_center = params[index:index+2]
        index += 2
        poly_correction1.convert_to_cylindrical_coords()
        poly_correction2.convert_to_cylindrical_coords()
    if fit_width:
        poly_correction1.width_parameters = poly_correction2.width_parameters = params[index:index+len(poly_correction1.width_ijk)]
        index += +len(poly_correction1.width_ijk)
        poly_correction1.apply_width_correction()
        poly_correction2.apply_width_correction()
    if fit_position:
        poly_correction1.r_parameters = poly_correction2.r_parameters = params[index:index+len(poly_correction1.r_ijklmn)]
        index += len(poly_correction1.r_ijklmn)
        poly_correction1.theta_parameters = poly_correction2.theta_parameters = params[index:index+len(poly_correction1.theta_ijklmn)]
        index += len(poly_correction1.theta_ijklmn)
        poly_correction1.z_parameters = poly_correction2.z_parameters = params[index:index+len(poly_correction1.z_ijklmn)]
        index += len(poly_correction1.z_ijklmn)
        poly_correction1.apply_field_correction()
        poly_correction2.apply_field_correction()

def get_init_guess(fit_beam_center:bool, fit_width:bool, fit_r_theta_z:bool):
    num_params = 1 #last parameter will be length offset
    if fit_beam_center:
        num_params += 2
    if fit_width:
        num_params += len(poly_correction1.width_ijk)
    if fit_r_theta_z:
        num_params += len(poly_correction1.z_ijklmn)
        num_params += len(poly_correction1.r_ijklmn)
        num_params += len(poly_correction1.theta_ijklmn)
    return np.zeros(num_params)


expected_lengths = proton_srim_table.get_stopping_distance(energy_MeV[event_mask])
with cp.cuda.Device(poly_correction1.gpu_id_to_use):
    expected_lengths_cp = cp.array(expected_lengths)

def get_lengths(params,fit_beam_center:bool, fit_width:bool, fit_position:bool, apply_width_correction=True):
    with cp.cuda.Device(gpu_id_to_use):
        apply_params(params, fit_beam_center, fit_width, fit_position)
        lengths_squared = cp.sum((poly_correction1.corrected_xyz - poly_correction2.corrected_xyz)**2, axis=1)
        #lengths_squared[lengths_squared<0] = 0
        if apply_width_correction:
            return params[-1]*poly_correction1.uncorrected_width + cp.sqrt(lengths_squared)#  + params[-2] 
        else:
            return cp.sqrt(lengths_squared)

def obj_func(params, fit_beam_center:bool, fit_width:bool, fit_position:bool):
    with cp.cuda.Device(gpu_id_to_use):
        delta_length = get_lengths(params, fit_beam_center, fit_width, fit_position) - expected_lengths_cp
        return np.sqrt(cp.asnumpy(cp.sum(delta_length**2))/num_selected_events)

def callback(res):
    print(res, )

mode_args = (False, True, True)
if not mode_args[0]:
    poly_correction1.convert_to_cylindrical_coords()
    poly_correction2.convert_to_cylindrical_coords()
def callback(res):
    print(res, obj_func(res, *mode_args))
save_fname = 'run299_300_301_field_dist.pkl'
if True:
    res = optimize.minimize(obj_func, get_init_guess(*mode_args), args=mode_args, callback=callback)
    with open(save_fname, 'wb') as f:
        pickle.dump(res, f)
else:
    with open(save_fname, 'rb') as f:
        res = pickle.load(f)
print(res)
print('used ', num_selected_events, ' to fit ', len(res.x), ' parameters')

plt_mask = veto_mask#e23035_runs.get_veto_mask(get_runs)
rve_bins = [plt.linspace(0,9, 200), plt.linspace(0,175, 200)]

x = np.linspace(0, 9)
y_proton = proton_srim_table.get_stopping_distance(x)
y_alpha = alpha_srim_table.get_stopping_distance(x)

plt.figure()
plt.title('uncorrected rve')
plt.hist2d(energy_MeV[plt_mask], uncorrected_lengths[plt_mask], bins=rve_bins, norm=matplotlib.colors.LogNorm())
plt.plot(x, y_proton)
plt.plot(x, y_alpha)

poly_correction1.set_data(endpoints[:,0,:], uncorrected_widths, uncorrected_angles, times_since_beam_off)
poly_correction2.set_data(endpoints[:,1,:], uncorrected_widths, uncorrected_angles, times_since_beam_off)
poly_correction1.convert_to_cylindrical_coords()
poly_correction2.convert_to_cylindrical_coords()

plt.figure()
plt.title('with correction')
lengths = cp.asnumpy(get_lengths(res.x, *mode_args))
plt.hist2d(energy_MeV[plt_mask], lengths[plt_mask], bins=rve_bins, norm=matplotlib.colors.LogNorm())
plt.plot(x, y_proton)
plt.plot(x, y_alpha)

# plt.figure()
# plt.scatter(cp.asnumpy(poly_correction1.uncorrected_angles), cp.asnumpy(poly_correction1.uncorrected_width), label='uncorrected width', marker='.')
# plt.scatter(cp.asnumpy(poly_correction1.uncorrected_angles), cp.asnumpy(poly_correction1.corrected_width), label='corrected width', marker='.', alpha=0.5)
# plt.legend()

plt.figure()
plt.title('uncorrected width vs angle')
plt.hist2d(cp.asnumpy(poly_correction1.uncorrected_angles), cp.asnumpy(poly_correction1.uncorrected_width), bins=200, norm=matplotlib.colors.LogNorm())
plt.figure()
plt.title('corrected width vs angle')
plt.hist2d(cp.asnumpy(poly_correction1.uncorrected_angles), cp.asnumpy(poly_correction1.corrected_width), bins=200, norm=matplotlib.colors.LogNorm())

plt.figure()
plt.title('events used for correction on uncorrected rve')
plt.hist2d(energy_MeV[plt_mask], uncorrected_lengths[plt_mask], bins=rve_bins, norm=matplotlib.colors.LogNorm())
plt.scatter(energy_MeV[event_mask], uncorrected_lengths[event_mask], marker='.', c='r',alpha=0.1)


plt.figure()
plt.title('events used for correction on corrected rve')
plt.hist2d(energy_MeV[plt_mask], lengths[plt_mask], bins=rve_bins, norm=matplotlib.colors.LogNorm())
plt.scatter(energy_MeV[event_mask], lengths[event_mask], marker='.', c='r',alpha=0.1)

plt.show(block=False)

#plt.scatter(cp.asnumpy(poly_correction1.uncorrected_angles), cp.asnumpy(poly_correction1.corrected_width), c=cp.asnumpy(poly_correction1.times), marker='.')

#run 145 events to try correcting: 5, 18, 36, 38, 39
def show_xyze(xs,ys,zs,es, title='',ax=None):
    if type(ax) == type(None):
        fig = plt.figure(figsize=(6,6))
        ax = plt.axes(projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    # ax.set_xlim3d(-200, 200)
    # ax.set_ylim3d(-200, 200)
    # ax.set_zlim3d(0, 400)

    ax.view_init(elev=45, azim=45)
    ax.scatter(xs, ys, zs, c=es)
    cbar = fig.colorbar(ax.get_children()[0])
    plt.title(title)

def show_corrected_event(evt_num, view_threshold=100):
    run = 145
    h5file = process_runs.get_h5_file(experiment=experiment, run_number=run)
    xyze = np.array(h5file.get_xyze(evt_num, threshold=view_threshold))
    points = xyze.T[:,:3]
    correction = copy.copy(poly_correction1)
    index = evt_num
    t = times_since_beam_off[index]
    width = uncorrected_widths[index]
    angle=uncorrected_angles[index]
    print("length (uncorrected, corrected)", uncorrected_lengths[evt_num], lengths[evt_num])
    print('evt num, width, angle, time')
    print(evt_num, width, angle, t)
    
    correction.set_data(points, widths=np.ones(len(points))*width, track_angles=angle, times=np.ones(len(points))*t)
    correction.convert_to_cylindrical_coords()
    correction.apply_width_correction()
    correction.apply_field_correction()
    corrected_points = cp.asnumpy(correction.corrected_xyz)

    fig = plt.figure( figsize=(13,6))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0)
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax1.scatter(xyze[0], xyze[1], xyze[2], c=xyze[3])
    ax2.scatter(corrected_points[:,0], corrected_points[:,1], corrected_points[:,2], c=xyze[3])

    ax1.set_xlim3d(-200, 200)
    ax1.set_ylim3d(-200, 200)
    ax1.set_zlim3d(0, 400)
    ax2.set_xlim3d(-200, 200)
    ax2.set_ylim3d(-200, 200)
    ax2.set_zlim3d(0, 400)

    def on_move(event):
        if event.inaxes == ax1:
            ax2.view_init(elev=ax1.elev, azim=ax1.azim)
        elif event.inaxes == ax2:
            ax1.view_init(elev=ax2.elev, azim=ax2.azim)
        else:
            return
        fig.canvas.draw_idle()

    c1 = fig.canvas.mpl_connect('motion_notify_event', on_move)

    # show_xyze(xyze[0], xyze[1], xyze[2], xyze[3], 'uncorrected points')
    # show_xyze(corrected_points[:,0], corrected_points[:,1], corrected_points[:,2], xyze[3], 'corrected points')

    