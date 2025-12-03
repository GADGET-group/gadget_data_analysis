import os
import re
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
import multiprocessing
import time

from track_fitting.SimulatedEvent import SimulatedEvent
from track_fitting. MultiParticleEvent import MultiParticleEvent
import raw_viewer.raw_h5_file as raw_h5_file
from track_fitting import SingleParticleEvent, build_sim

categorized_events_of_interest = pd.read_csv('./complete_categorized_events_of_interest.csv',\
    encoding='utf-8-sig', skip_blank_lines = False, nrows = 36164, header=None)

array_of_categorized_events_of_interest = categorized_events_of_interest[0].to_numpy()

mask = np.isin(array_of_categorized_events_of_interest, ['RnPo Chain', 'Accidental Coin', 'Double Alpha Candidate'])
# mask = np.isin(array_of_categorized_events_of_interest, ['Double Alpha Candidate'])
events = np.where(mask)[0]

# Get a list of events already fitted and add them to the list
# for log likelihood fits
# results_directory = os.fsencode("/egr/research-tpc/dopferjo/gadget_analysis/fit_results")
# pattern = re.compile(r"event_(\d+)_ll_fit_two_particle_decays_in_e24joe.dat$")

# for least sqaures fit
best_guess_results_directory = os.fsencode("/egr/research-tpc/dopferjo/gadget_analysis/fit_results/least_squares")

# for least sqaures fit (all directions)
ff_results_directory = os.fsencode("/egr/research-tpc/dopferjo/gadget_analysis/fit_results/least_squares/ff")
fb_results_directory = os.fsencode("/egr/research-tpc/dopferjo/gadget_analysis/fit_results/least_squares/fb")
bf_results_directory = os.fsencode("/egr/research-tpc/dopferjo/gadget_analysis/fit_results/least_squares/bf")
bb_results_directory = os.fsencode("/egr/research-tpc/dopferjo/gadget_analysis/fit_results/least_squares/bb")
pattern = re.compile(r"event_(\d+)_ls_fit_two_particle_decays_in_e24joe.dat$")

completed_fit_events = []
for filename in os.listdir(ff_results_directory):
    match = pattern.match(filename.decode('utf-8'))
    if match:
        event_num = int(match.group(1))
        completed_fit_events.append(event_num)
# events = [item for item in events if item in completed_fit_events]
# completed_fit_events = [34001]
evts, theta0,theta1, phi0,phi1, x0,y0,z0, x1,y1,z1, lls, cats, E0,E1, Erecs, nfev, sigma_xy, sigma_z, counts_per_mev = [], [],[], [],[], [],[],[], [],[],[], [], [], [],[], [], [], [],[], []
theta0_ff,theta1_ff, phi0_ff,phi1_ff, x0_ff,y0_ff,z0_ff, x1_ff,y1_ff,z1_ff, lls_ff, cats_ff, E0_ff,E1_ff, Erecs_ff, nfev_ff, sigma_xy0_ff, sigma_z0_ff, sigma_xy1_ff, sigma_z1_ff, counts_per_mev_ff = [],[], [],[], [],[],[], [],[],[], [], [], [],[], [], [], [],[], [],[], []
theta0_fb,theta1_fb, phi0_fb,phi1_fb, x0_fb,y0_fb,z0_fb, x1_fb,y1_fb,z1_fb, lls_fb, cats_fb, E0_fb,E1_fb, Erecs_fb, nfev_fb, sigma_xy0_fb, sigma_z0_fb, sigma_xy1_fb, sigma_z1_fb, counts_per_mev_fb = [],[], [],[], [],[],[], [],[],[], [], [], [],[], [], [], [],[], [],[], []
theta0_bf,theta1_bf, phi0_bf,phi1_bf, x0_bf,y0_bf,z0_bf, x1_bf,y1_bf,z1_bf, lls_bf, cats_bf, E0_bf,E1_bf, Erecs_bf, nfev_bf, sigma_xy0_bf, sigma_z0_bf, sigma_xy1_bf, sigma_z1_bf, counts_per_mev_bf = [],[], [],[], [],[],[], [],[],[], [], [], [],[], [], [], [],[], [],[], []
theta0_bb,theta1_bb, phi0_bb,phi1_bb, x0_bb,y0_bb,z0_bb, x1_bb,y1_bb,z1_bb, lls_bb, cats_bb, E0_bb,E1_bb, Erecs_bb, nfev_bb, sigma_xy0_bb, sigma_z0_bb, sigma_xy1_bb, sigma_z1_bb, counts_per_mev_bb = [],[], [],[], [],[],[], [],[],[], [], [], [],[], [], [], [],[], [],[], []
theta0_best,theta1_best, phi0_best,phi1_best, x0_best,y0_best,z0_best, x1_best,y1_best,z1_best, lls_best, cats_best, E0_best,E1_best, Erecs_best, nfev_best, sigma_xy0_best, sigma_z0_best, sigma_xy1_best, sigma_z1_best, counts_per_mev_best = [],[], [],[], [],[],[], [],[],[], [], [], [],[], [], [], [],[], [],[], []

message, success = [], []
message_ff, success_ff = [], []
message_fb, success_fb = [], []
message_bf, success_bf = [], []
message_bb, success_bb = [], []
message_best, success_best = [], []
residuals_ff = []
residuals_fb = []
residuals_bf = []
residuals_bb = []
residuals_kmeans = []

counts_per_mev_best = []
lls_best = []

residual_count_sum = []
observed_trace_sum = []

fit_results = {'event': [],
               'success_ff': [],
               'message_ff': [],
               'fun_ff': [],
               'theta0_ff': [],
               'theta1_ff': [],
               'phi0_ff': [],
               'phi1_ff': [],
               'x0_ff': [],
               'y0_ff': [],
               'z0_ff': [],
               'x1_ff': [],
               'y1_ff': [],
               'z1_ff': [],
               'e0_ff': [],
               'e1_ff': [],
               'sigma_xy0_ff': [],
               'sigma_z0_ff': [],
               'sigma_xy1_ff': [],
               'sigma_z1_ff': [],
               'counts_per_mev_ff': [],
               'nfev_ff': [],
               'residuals_ff': [],
               'success_fb': [],
               'message_fb': [],
               'fun_fb': [],
               'theta0_fb': [],
               'theta1_fb': [],
               'phi0_fb': [],
               'phi1_fb': [],
               'x0_fb': [],
               'y0_fb': [],
               'z0_fb': [],
               'x1_fb': [],
               'y1_fb': [],
               'z1_fb': [],
               'e0_fb': [],
               'e1_fb': [],
               'sigma_xy0_fb': [],
               'sigma_z0_fb': [],
               'sigma_xy1_fb': [],
               'sigma_z1_fb': [],
               'counts_per_mev_fb': [],
               'nfev_fb': [],
               'residuals_fb': [],
               'success_bf': [],
               'message_bf': [],
               'fun_bf': [],
               'theta0_bf': [],
               'theta1_bf': [],
               'phi0_bf': [],
               'phi1_bf': [],
               'x0_bf': [],
               'y0_bf': [],
               'z0_bf': [],
               'x1_bf': [],
               'y1_bf': [],
               'z1_bf': [],
               'e0_bf': [],
               'e1_bf': [],
               'sigma_xy0_bf': [],
               'sigma_z0_bf': [],
               'sigma_xy1_bf': [],
               'sigma_z1_bf': [],
               'counts_per_mev_bf': [],
               'nfev_bf': [],
               'residuals_bf': [],
               'success_bb': [],
               'message_bb': [],
               'fun_bb': [],
               'theta0_bb': [],
               'theta1_bb': [],
               'phi0_bb': [],
               'phi1_bb': [],
               'x0_bb': [],
               'y0_bb': [],
               'z0_bb': [],
               'x1_bb': [],
               'y1_bb': [],
               'z1_bb': [],
               'e0_bb': [],
               'e1_bb': [],
               'sigma_xy0_bb': [],
               'sigma_z0_bb': [],
               'sigma_xy1_bb': [],
               'sigma_z1_bb': [],
               'counts_per_mev_bb': [],
               'nfev_bb': [],
               'residuals_bb': [],
               'success_kmeans': [],
               'message_kmeans': [],
               'fun_kmeans': [],
               'theta0_kmeans': [],
               'theta1_kmeans': [],
               'phi0_kmeans': [],
               'phi1_kmeans': [],
               'x0_kmeans': [],
               'y0_kmeans': [],
               'z0_kmeans': [],
               'x1_kmeans': [],
               'y1_kmeans': [],
               'z1_kmeans': [],
               'e0_kmeans': [],
               'e1_kmeans': [],
               'sigma_xy_kmeans': [],
               'sigma_z_kmeans': [],
               'counts_per_mev_kmeans': [],
               'nfev_kmeans': [],
               'residuals_kmeans': []
               }
# completed_fit_events = [23510,1364]
failed_events = []
print("Number of Events to go through: ",len(completed_fit_events))
for i in tqdm(completed_fit_events):
    evts.append(i)
    fit_results["event"].append(i)
    # with open("/egr/research-tpc/dopferjo/gadget_analysis/fit_results/least_squares/ff/event_%05d_ls_fit_two_particle_decays_in_e24joe.dat"%i, 'rb') as file:
    with open("/egr/research-tpc/dopferjo/gadget_analysis/fit_results/least_squares/ff_best/event_%05d_ls_fit_two_particle_decays_in_e24joe_starting_from_previous_fit_params.dat"%i, 'rb') as file:
        data = pickle.load(file)
        # bnds = ((0,1),(0,1),(0,1),(0,1),(-1,1),(-1,1),(0,1),(-1,1),(-1,1),(0,1),(0.1,1),(0.1,1),(0,0.5),(0,0.5),(0,1))
        # if np.any(np.isclose(data[0].x, [b[0] for b in bnds], atol = 1e-6)) or np.any(np.isclose(data[0].x, [b[1] for b in bnds], atol = 1e-6)):
        #     print("Event %d"%i, data[0].message, data[0].x)

        message_ff.append(data[0].message)
        success_ff.append(data[0].success)
        lls_ff.append(data[0].fun)
        # if data[0].success:
        #     lls_ff.append(data[0].fun)
        # else:
        #     lls_ff.append(np.inf)
        # if data[0].message == "`callback` raised `StopIteration`." or data[0].message == "ABNORMAL_TERMINATION_IN_LNSRCH":
        #     continue
        # else:
        #     print(data[0].message)
        theta0_ff.append(data[0].x[0] * np.pi)
        theta1_ff.append(data[0].x[1] * np.pi)
        phi0_ff.append(data[0].x[2] * 2 * np.pi)
        phi1_ff.append(data[0].x[3] * 2 * np.pi)
        x0_ff.append(data[0].x[4]* 40)
        y0_ff.append(data[0].x[5] * 40)
        z0_ff.append(data[0].x[6] * 400)
        x1_ff.append(data[0].x[7]* 40)
        y1_ff.append(data[0].x[8] * 40)
        z1_ff.append(data[0].x[9] * 400)
        E0_ff.append(data[0].x[10] * 10)
        E1_ff.append(data[0].x[11] * 10)
        sigma_xy0_ff.append(data[0].x[12] * 10)
        sigma_z0_ff.append(data[0].x[13] * 10)
        sigma_xy1_ff.append(data[0].x[14] * 10)
        sigma_z1_ff.append(data[0].x[15] * 10)
        counts_per_mev_ff.append(data[0].x[16] * 1e6)
        # lls_ff.append(data[0].fun)
        nfev_ff.append(data[0].nfev)
        fit_results["success_ff"].append(data[0].success)
        fit_results["message_ff"].append(data[0].message)
        fit_results["fun_ff"].append(data[0].fun)
        fit_results["theta0_ff"].append(data[0].x[0] * np.pi)
        fit_results["theta1_ff"].append(data[0].x[1] * np.pi)
        fit_results["phi0_ff"].append(data[0].x[2] * 2 * np.pi)
        fit_results["phi1_ff"].append(data[0].x[3] * 2 * np.pi)
        fit_results["x0_ff"].append(data[0].x[4]* 40)
        fit_results["y0_ff"].append(data[0].x[5]* 40)
        fit_results["z0_ff"].append(data[0].x[6]* 400)
        fit_results["x1_ff"].append(data[0].x[7]* 40)
        fit_results["y1_ff"].append(data[0].x[8]* 40)
        fit_results["z1_ff"].append(data[0].x[9]* 400)
        fit_results["e0_ff"].append(data[0].x[10] * 10)
        fit_results["e1_ff"].append(data[0].x[11] * 10)
        fit_results["sigma_xy0_ff"].append(data[0].x[12] * 10)
        fit_results["sigma_z0_ff"].append(data[0].x[13] * 10)
        fit_results["sigma_xy1_ff"].append(data[0].x[14] * 10)
        fit_results["sigma_z1_ff"].append(data[0].x[15] * 10)
        fit_results["counts_per_mev_ff"].append(data[0].x[16] * 1e6)
        fit_results["nfev_ff"].append(data[0].nfev)
        trace_sim = build_sim.create_multi_particle_event('e24joe', 124, i, ['4He','4He'])
        trace_sim.per_particle_params = ['initial_energy', 'theta', 'phi', 'sigma_xy', 'sigma_z', 'num_stopping_power_points','initial_point'] 
        trace_sim.shared_params = ['gas_density']
        trace_sim.sims[0].adaptive_stopping_power = False
        trace_sim.sims[0].initial_energy = data[0].x[10] * 10
        trace_sim.sims[0].theta, trace_sim.sims[0].phi = data[0].x[0] * np.pi, data[0].x[2] * 2 * np.pi
        trace_sim.sims[0].num_stopping_power_points = trace_sim.sims[0].get_num_stopping_points_for_energy(data[0].x[10] * 10)
        trace_sim.sims[0].initial_point = (data[0].x[4]* 40,data[0].x[5]* 40,data[0].x[6]* 400)
        trace_sim.sims[0].sigma_xy = data[0].x[12] * 10
        trace_sim.sims[0].sigma_z = data[0].x[13] * 10
        trace_sim.sims[1].adaptive_stopping_power = False
        trace_sim.sims[1].initial_energy = data[0].x[11] * 10
        trace_sim.sims[1].theta, trace_sim.sims[1].phi = data[0].x[1] * np.pi, data[0].x[3] * 2 * np.pi
        trace_sim.sims[1].num_stopping_power_points = trace_sim.sims[1].get_num_stopping_points_for_energy(data[0].x[11] * 10)
        trace_sim.sims[1].initial_point = (data[0].x[7]* 40,data[0].x[8]* 40,data[0].x[9]* 400)
        trace_sim.sims[1].sigma_xy = data[0].x[14] * 10
        trace_sim.sims[1].sigma_z = data[0].x[15] * 10
        trace_sim.simulate_event()
        fit_results["residuals_ff"] = np.sum(trace_sim.get_residuals())
        residuals_ff.append(np.sum(trace_sim.get_residuals()))
        
    # with open("/egr/research-tpc/dopferjo/gadget_analysis/fit_results/least_squares/fb/event_%05d_ls_fit_two_particle_decays_in_e24joe.dat"%i, 'rb') as file:
    with open("/egr/research-tpc/dopferjo/gadget_analysis/fit_results/least_squares/fb_best/event_%05d_ls_fit_two_particle_decays_in_e24joe_starting_from_previous_fit_params.dat"%i, 'rb') as file:
        data = pickle.load(file)
        # bnds = ((0,1),(0,1),(0,1),(0,1),(-1,1),(-1,1),(0,1),(-1,1),(-1,1),(0,1),(0.1,1),(0.1,1),(0,0.5),(0,0.5),(0,1))
        # if np.any(np.isclose(data[0].x, [b[0] for b in bnds], atol = 1e-6)) or np.any(np.isclose(data[0].x, [b[1] for b in bnds], atol = 1e-6)):
        #     print("Event %d"%i, data[0].message, data[0].x)

        message_fb.append(data[0].message)
        success_fb.append(data[0].success)
        lls_fb.append(data[0].fun)
        # if data[0].success:
        #     lls_fb.append(data[0].fun)
        # else:
        #     lls_fb.append(np.inf)
        # if data[0].message == "`callback` raised `StopIteration`." or data[0].message == "ABNORMAL_TERMINATION_IN_LNSRCH":
        #     continue
        # else:
        #     print(data[0].message)
        theta0_fb.append(data[0].x[0] * np.pi)
        theta1_fb.append(data[0].x[1] * np.pi)
        phi0_fb.append(data[0].x[2] * 2 * np.pi)
        phi1_fb.append(data[0].x[3] * 2 * np.pi)
        x0_fb.append(data[0].x[4]* 40)
        y0_fb.append(data[0].x[5] * 40)
        z0_fb.append(data[0].x[6] * 400)
        x1_fb.append(data[0].x[7]* 40)
        y1_fb.append(data[0].x[8] * 40)
        z1_fb.append(data[0].x[9] * 400)
        E0_fb.append(data[0].x[10] * 10)
        E1_fb.append(data[0].x[11] * 10)
        sigma_xy0_fb.append(data[0].x[12] * 10)
        sigma_z0_fb.append(data[0].x[13] * 10)
        sigma_xy1_fb.append(data[0].x[14] * 10)
        sigma_z1_fb.append(data[0].x[15] * 10)
        counts_per_mev_fb.append(data[0].x[16] * 1e6)
        # lls_fb.append(data[0].fun)
        nfev_fb.append(data[0].nfev)
        fit_results["success_fb"].append(data[0].success)
        fit_results["message_fb"].append(data[0].message)
        fit_results["fun_fb"].append(data[0].fun)
        fit_results["theta0_fb"].append(data[0].x[0] * np.pi)
        fit_results["theta1_fb"].append(data[0].x[1] * np.pi)
        fit_results["phi0_fb"].append(data[0].x[2] * 2 * np.pi)
        fit_results["phi1_fb"].append(data[0].x[3] * 2 * np.pi)
        fit_results["x0_fb"].append(data[0].x[4]* 40)
        fit_results["y0_fb"].append(data[0].x[5]* 40)
        fit_results["z0_fb"].append(data[0].x[6]* 400)
        fit_results["x1_fb"].append(data[0].x[7]* 40)
        fit_results["y1_fb"].append(data[0].x[8]* 40)
        fit_results["z1_fb"].append(data[0].x[9]* 400)
        fit_results["e0_fb"].append(data[0].x[10] * 10)
        fit_results["e1_fb"].append(data[0].x[11] * 10)
        fit_results["sigma_xy0_fb"].append(data[0].x[12] * 10)
        fit_results["sigma_z0_fb"].append(data[0].x[13] * 10)
        fit_results["sigma_xy1_fb"].append(data[0].x[14] * 10)
        fit_results["sigma_z1_fb"].append(data[0].x[15] * 10)
        fit_results["counts_per_mev_fb"].append(data[0].x[16] * 1e6)
        fit_results["nfev_fb"].append(data[0].nfev)
        trace_sim = build_sim.create_multi_particle_event('e24joe', 124, i, ['4He','4He'])
        trace_sim.per_particle_params = ['initial_energy', 'theta', 'phi', 'sigma_xy', 'sigma_z', 'num_stopping_power_points','initial_point'] 
        trace_sim.shared_params = ['gas_density']
        trace_sim.sims[0].adaptive_stopping_power = False
        trace_sim.sims[0].initial_energy = data[0].x[10] * 10
        trace_sim.sims[0].theta, trace_sim.sims[0].phi = data[0].x[0] * np.pi, data[0].x[2] * 2 * np.pi
        trace_sim.sims[0].num_stopping_power_points = trace_sim.sims[0].get_num_stopping_points_for_energy(data[0].x[10] * 10)
        trace_sim.sims[0].initial_point = (data[0].x[4]* 40,data[0].x[5]* 40,data[0].x[6]* 400)
        trace_sim.sims[0].sigma_xy = data[0].x[12] * 10
        trace_sim.sims[0].sigma_z = data[0].x[13] * 10
        trace_sim.sims[1].adaptive_stopping_power = False
        trace_sim.sims[1].initial_energy = data[0].x[11] * 10
        trace_sim.sims[1].theta, trace_sim.sims[1].phi = data[0].x[1] * np.pi, data[0].x[3] * 2 * np.pi
        trace_sim.sims[1].num_stopping_power_points = trace_sim.sims[1].get_num_stopping_points_for_energy(data[0].x[11] * 10)
        trace_sim.sims[1].initial_point = (data[0].x[7]* 40,data[0].x[8]* 40,data[0].x[9]* 400)
        trace_sim.sims[1].sigma_xy = data[0].x[14] * 10
        trace_sim.sims[1].sigma_z = data[0].x[15] * 10
        trace_sim.simulate_event()
        fit_results["residuals_fb"] = np.sum(trace_sim.get_residuals())
        
    with open("/egr/research-tpc/dopferjo/gadget_analysis/fit_results/least_squares/bf_best/event_%05d_ls_fit_two_particle_decays_in_e24joe_starting_from_previous_fit_params.dat"%i, 'rb') as file:
    # with open("/egr/research-tpc/dopferjo/gadget_analysis/fit_results/least_squares/bf/event_%05d_ls_fit_two_particle_decays_in_e24joe.dat"%i, 'rb') as file:
        data = pickle.load(file)
        # bnds = ((0,1),(0,1),(0,1),(0,1),(-1,1),(-1,1),(0,1),(-1,1),(-1,1),(0,1),(0.1,1),(0.1,1),(0,0.5),(0,0.5),(0,1))
        # if np.any(np.isclose(data[0].x, [b[0] for b in bnds], atol = 1e-6)) or np.any(np.isclose(data[0].x, [b[1] for b in bnds], atol = 1e-6)):
        #     print("Event %d"%i, data[0].message, data[0].x)

        message_bf.append(data[0].message)
        success_bf.append(data[0].success)
        lls_bf.append(data[0].fun)
        # if data[0].success:
        #     lls_bf.append(data[0].fun)
        # else:
        #     lls_bf.append(np.inf)
        # if data[0].message == "`callback` raised `StopIteration`." or data[0].message == "ABNORMAL_TERMINATION_IN_LNSRCH":
        #     continue
        # else:
        #     print(data[0].message)
        theta0_bf.append(data[0].x[0] * np.pi)
        theta1_bf.append(data[0].x[1] * np.pi)
        phi0_bf.append(data[0].x[2] * 2 * np.pi)
        phi1_bf.append(data[0].x[3] * 2 * np.pi)
        x0_bf.append(data[0].x[4]* 40)
        y0_bf.append(data[0].x[5] * 40)
        z0_bf.append(data[0].x[6] * 400)
        x1_bf.append(data[0].x[7]* 40)
        y1_bf.append(data[0].x[8] * 40)
        z1_bf.append(data[0].x[9] * 400)
        E0_bf.append(data[0].x[10] * 10)
        E1_bf.append(data[0].x[11] * 10)
        sigma_xy0_bf.append(data[0].x[12] * 10)
        sigma_z0_bf.append(data[0].x[13] * 10)
        sigma_xy1_bf.append(data[0].x[14] * 10)
        sigma_z1_bf.append(data[0].x[15] * 10)
        counts_per_mev_bf.append(data[0].x[16] * 1e6)
        # lls_bf.append(data[0].fun)
        nfev_bf.append(data[0].nfev)
        fit_results["success_bf"].append(data[0].success)
        fit_results["message_bf"].append(data[0].message)
        fit_results["fun_bf"].append(data[0].fun)
        fit_results["theta0_bf"].append(data[0].x[0] * np.pi)
        fit_results["theta1_bf"].append(data[0].x[1] * np.pi)
        fit_results["phi0_bf"].append(data[0].x[2] * 2 * np.pi)
        fit_results["phi1_bf"].append(data[0].x[3] * 2 * np.pi)
        fit_results["x0_bf"].append(data[0].x[4]* 40)
        fit_results["y0_bf"].append(data[0].x[5]* 40)
        fit_results["z0_bf"].append(data[0].x[6]* 400)
        fit_results["x1_bf"].append(data[0].x[7]* 40)
        fit_results["y1_bf"].append(data[0].x[8]* 40)
        fit_results["z1_bf"].append(data[0].x[9]* 400)
        fit_results["e0_bf"].append(data[0].x[10] * 10)
        fit_results["e1_bf"].append(data[0].x[11] * 10)
        fit_results["sigma_xy0_bf"].append(data[0].x[12] * 10)
        fit_results["sigma_z0_bf"].append(data[0].x[13] * 10)
        fit_results["sigma_xy1_bf"].append(data[0].x[14] * 10)
        fit_results["sigma_z1_bf"].append(data[0].x[15] * 10)
        fit_results["counts_per_mev_bf"].append(data[0].x[16] * 1e6)
        fit_results["nfev_bf"].append(data[0].nfev)
        trace_sim = build_sim.create_multi_particle_event('e24joe', 124, i, ['4He','4He'])
        trace_sim.per_particle_params = ['initial_energy', 'theta', 'phi', 'sigma_xy', 'sigma_z', 'num_stopping_power_points','initial_point'] 
        trace_sim.shared_params = ['gas_density']
        trace_sim.sims[0].adaptive_stopping_power = False
        trace_sim.sims[0].initial_energy = data[0].x[10] * 10
        trace_sim.sims[0].theta, trace_sim.sims[0].phi = data[0].x[0] * np.pi, data[0].x[2] * 2 * np.pi
        trace_sim.sims[0].num_stopping_power_points = trace_sim.sims[0].get_num_stopping_points_for_energy(data[0].x[10] * 10)
        trace_sim.sims[0].initial_point = (data[0].x[4]* 40,data[0].x[5]* 40,data[0].x[6]* 400)
        trace_sim.sims[0].sigma_xy = data[0].x[12] * 10
        trace_sim.sims[0].sigma_z = data[0].x[13] * 10
        trace_sim.sims[1].adaptive_stopping_power = False
        trace_sim.sims[1].initial_energy = data[0].x[11] * 10
        trace_sim.sims[1].theta, trace_sim.sims[1].phi = data[0].x[1] * np.pi, data[0].x[3] * 2 * np.pi
        trace_sim.sims[1].num_stopping_power_points = trace_sim.sims[1].get_num_stopping_points_for_energy(data[0].x[11] * 10)
        trace_sim.sims[1].initial_point = (data[0].x[7]* 40,data[0].x[8]* 40,data[0].x[9]* 400)
        trace_sim.sims[1].sigma_xy = data[0].x[14] * 10
        trace_sim.sims[1].sigma_z = data[0].x[15] * 10
        trace_sim.simulate_event()
        fit_results["residuals_bf"] = np.sum(trace_sim.get_residuals())
        
    with open("/egr/research-tpc/dopferjo/gadget_analysis/fit_results/least_squares/bb_best/event_%05d_ls_fit_two_particle_decays_in_e24joe_starting_from_previous_fit_params.dat"%i, 'rb') as file:
    # with open("/egr/research-tpc/dopferjo/gadget_analysis/fit_results/least_squares/bb/event_%05d_ls_fit_two_particle_decays_in_e24joe.dat"%i, 'rb') as file:
        data = pickle.load(file)
        # bnds = ((0,1),(0,1),(0,1),(0,1),(-1,1),(-1,1),(0,1),(-1,1),(-1,1),(0,1),(0.1,1),(0.1,1),(0,0.5),(0,0.5),(0,1))
        # if np.any(np.isclose(data[0].x, [b[0] for b in bnds], atol = 1e-6)) or np.any(np.isclose(data[0].x, [b[1] for b in bnds], atol = 1e-6)):
        #     print("Event %d"%i, data[0].message, data[0].x)

        message_bb.append(data[0].message)
        success_bb.append(data[0].success)
        lls_bb.append(data[0].fun)
        # if data[0].success:
        #     lls_bb.append(data[0].fun)
        # else:
        #     lls_bb.append(np.inf)
        # if data[0].message == "`callback` raised `StopIteration`." or data[0].message == "ABNORMAL_TERMINATION_IN_LNSRCH":
        #     continue
        # else:
        #     print(data[0].message)
        theta0_bb.append(data[0].x[0] * np.pi)
        theta1_bb.append(data[0].x[1] * np.pi)
        phi0_bb.append(data[0].x[2] * 2 * np.pi)
        phi1_bb.append(data[0].x[3] * 2 * np.pi)
        x0_bb.append(data[0].x[4]* 40)
        y0_bb.append(data[0].x[5] * 40)
        z0_bb.append(data[0].x[6] * 400)
        x1_bb.append(data[0].x[7]* 40)
        y1_bb.append(data[0].x[8] * 40)
        z1_bb.append(data[0].x[9] * 400)
        E0_bb.append(data[0].x[10] * 10)
        E1_bb.append(data[0].x[11] * 10)
        sigma_xy0_bb.append(data[0].x[12] * 10)
        sigma_z0_bb.append(data[0].x[13] * 10)
        sigma_xy1_bb.append(data[0].x[14] * 10)
        sigma_z1_bb.append(data[0].x[15] * 10)
        counts_per_mev_bb.append(data[0].x[16] * 1e6)
        # lls_bb.append(data[0].fun)
        nfev_bb.append(data[0].nfev)
        fit_results["success_bb"].append(data[0].success)
        fit_results["message_bb"].append(data[0].message)
        fit_results["fun_bb"].append(data[0].fun)
        fit_results["theta0_bb"].append(data[0].x[0] * np.pi)
        fit_results["theta1_bb"].append(data[0].x[1] * np.pi)
        fit_results["phi0_bb"].append(data[0].x[2] * 2 * np.pi)
        fit_results["phi1_bb"].append(data[0].x[3] * 2 * np.pi)
        fit_results["x0_bb"].append(data[0].x[4]* 40)
        fit_results["y0_bb"].append(data[0].x[5]* 40)
        fit_results["z0_bb"].append(data[0].x[6]* 400)
        fit_results["x1_bb"].append(data[0].x[7]* 40)
        fit_results["y1_bb"].append(data[0].x[8]* 40)
        fit_results["z1_bb"].append(data[0].x[9]* 400)
        fit_results["e0_bb"].append(data[0].x[10] * 10)
        fit_results["e1_bb"].append(data[0].x[11] * 10)
        fit_results["sigma_xy0_bb"].append(data[0].x[12] * 10)
        fit_results["sigma_z0_bb"].append(data[0].x[13] * 10)
        fit_results["sigma_xy1_bb"].append(data[0].x[14] * 10)
        fit_results["sigma_z1_bb"].append(data[0].x[15] * 10)
        fit_results["counts_per_mev_bb"].append(data[0].x[16] * 1e6)
        fit_results["nfev_bb"].append(data[0].nfev)
        trace_sim = build_sim.create_multi_particle_event('e24joe', 124, i, ['4He','4He'])
        trace_sim.per_particle_params = ['initial_energy', 'theta', 'phi', 'sigma_xy', 'sigma_z', 'num_stopping_power_points','initial_point'] 
        trace_sim.shared_params = ['gas_density']
        trace_sim.sims[0].adaptive_stopping_power = False
        trace_sim.sims[0].initial_energy = data[0].x[10] * 10
        trace_sim.sims[0].theta, trace_sim.sims[0].phi = data[0].x[0] * np.pi, data[0].x[2] * 2 * np.pi
        trace_sim.sims[0].num_stopping_power_points = trace_sim.sims[0].get_num_stopping_points_for_energy(data[0].x[10] * 10)
        trace_sim.sims[0].initial_point = (data[0].x[4]* 40,data[0].x[5]* 40,data[0].x[6]* 400)
        trace_sim.sims[0].sigma_xy = data[0].x[12] * 10
        trace_sim.sims[0].sigma_z = data[0].x[13] * 10
        trace_sim.sims[1].adaptive_stopping_power = False
        trace_sim.sims[1].initial_energy = data[0].x[11] * 10
        trace_sim.sims[1].theta, trace_sim.sims[1].phi = data[0].x[1] * np.pi, data[0].x[3] * 2 * np.pi
        trace_sim.sims[1].num_stopping_power_points = trace_sim.sims[1].get_num_stopping_points_for_energy(data[0].x[11] * 10)
        trace_sim.sims[1].initial_point = (data[0].x[7]* 40,data[0].x[8]* 40,data[0].x[9]* 400)
        trace_sim.sims[1].sigma_xy = data[0].x[14] * 10
        trace_sim.sims[1].sigma_z = data[0].x[15] * 10
        trace_sim.simulate_event()
        fit_results["residuals_bb"] = np.sum(trace_sim.get_residuals())
        
    with open("/egr/research-tpc/dopferjo/gadget_analysis/fit_results/least_squares/cluster_best_initial_guess_start/event_%05d_ls_fit_two_particle_decays_in_e24joe.dat"%i, 'rb') as file:
        data = pickle.load(file)
        # bnds = ((0,1),(0,1),(0,1),(0,1),(-1,1),(-1,1),(0,1),(-1,1),(-1,1),(0,1),(0.1,1),(0.1,1),(0,0.5),(0,0.5),(0,1))
        # if np.any(np.isclose(data[0].x, [b[0] for b in bnds], atol = 1e-6)) or np.any(np.isclose(data[0].x, [b[1] for b in bnds], atol = 1e-6)):
        #     print("Event %d"%i, data[0].message, data[0].x)

        message.append(data[0].message)
        success.append(data[0].success)
        lls.append(data[0].fun)
        # if data[0].success:
        #     lls.append(data[0].fun)
        # else:
        #     lls.append(np.inf)
        
        # if data[0].message == "`callback` raised `StopIteration`." or data[0].message == "ABNORMAL_TERMINATION_IN_LNSRCH":
        #     continue
        # else:
        #     print(data[0].message)
        theta0.append(data[0].x[0] * np.pi)
        theta1.append(data[0].x[1] * np.pi)
        phi0.append(data[0].x[2] * 2 * np.pi)
        phi1.append(data[0].x[3] * 2 * np.pi)
        x0.append(data[0].x[4]* 40)
        y0.append(data[0].x[5] * 40)
        z0.append(data[0].x[6] * 400)
        x1.append(data[0].x[7]* 40)
        y1.append(data[0].x[8] * 40)
        z1.append(data[0].x[9] * 400)
        E0.append(data[0].x[10] * 10)
        E1.append(data[0].x[11] * 10)
        sigma_xy.append(data[0].x[12] * 10)
        sigma_z.append(data[0].x[13] * 10)
        # counts_per_mev.append(data[0].x[14] * 1e6)
        # lls.append(data[0].fun)
        nfev.append(data[0].nfev)
        fit_results["success_kmeans"].append(data[0].success)
        fit_results["message_kmeans"].append(data[0].message)
        fit_results["fun_kmeans"].append(data[0].fun)
        fit_results["theta0_kmeans"].append(data[0].x[0] * np.pi)
        fit_results["theta1_kmeans"].append(data[0].x[1] * np.pi)
        fit_results["phi0_kmeans"].append(data[0].x[2] * 2 * np.pi)
        fit_results["phi1_kmeans"].append(data[0].x[3] * 2 * np.pi)
        fit_results["x0_kmeans"].append(data[0].x[4]* 40)
        fit_results["y0_kmeans"].append(data[0].x[5]* 40)
        fit_results["z0_kmeans"].append(data[0].x[6]* 400)
        fit_results["x1_kmeans"].append(data[0].x[7]* 40)
        fit_results["y1_kmeans"].append(data[0].x[8]* 40)
        fit_results["z1_kmeans"].append(data[0].x[9]* 400)
        fit_results["e0_kmeans"].append(data[0].x[10] * 10)
        fit_results["e1_kmeans"].append(data[0].x[11] * 10)
        fit_results["sigma_xy_kmeans"].append(data[0].x[12] * 10)
        fit_results["sigma_z_kmeans"].append(data[0].x[13] * 10)
        fit_results["counts_per_mev_kmeans"].append(-1)
        fit_results["nfev_kmeans"].append(data[0].nfev)
    
        # best_direction_from_all_direction_fits = np.argmin([lls_ff[-1],lls_fb[-1],lls_bf[-1],lls_bb[-1]]) # should be 0,1,2, or 3
        # print(best_direction_from_all_direction_fits)
        # distance_between_bg_and_ff_00 = np.sqrt( (x0[-1] - x0_ff[-1])**2 + (y0[-1] - y0_ff[-1])**2 + (z0[-1] - z0_ff[-1])**2)
        # distance_between_bg_and_ff_01 = np.sqrt( (x0[-1] - x1_ff[-1])**2 + (y0[-1] - y1_ff[-1])**2 + (z0[-1] - z1_ff[-1])**2)
        
        # distance_between_bg_and_fb_00 = np.sqrt( (x0[-1] - x0_fb[-1])**2 + (y0[-1] - y0_fb[-1])**2 + (z0[-1] - z0_fb[-1])**2)
        # distance_between_bg_and_fb_01 = np.sqrt( (x0[-1] - x1_fb[-1])**2 + (y0[-1] - y1_fb[-1])**2 + (z0[-1] - z1_fb[-1])**2)
        
        # distance_between_bg_and_bf_00 = np.sqrt( (x0[-1] - x0_bf[-1])**2 + (y0[-1] - y0_bf[-1])**2 + (z0[-1] - z0_bf[-1])**2)
        # distance_between_bg_and_bf_01 = np.sqrt( (x0[-1] - x1_bf[-1])**2 + (y0[-1] - y1_bf[-1])**2 + (z0[-1] - z1_bf[-1])**2)
        
        # distance_between_bg_and_bb_00 = np.sqrt( (x0[-1] - x0_bb[-1])**2 + (y0[-1] - y0_bb[-1])**2 + (z0[-1] - z0_bb[-1])**2)
        # distance_between_bg_and_bb_01 = np.sqrt( (x0[-1] - x1_bb[-1])**2 + (y0[-1] - y1_bb[-1])**2 + (z0[-1] - z1_bb[-1])**2)
        
        # if distance_between_bg_and_ff_00 > distance_between_bg_and_ff_01:
        #     distance_between_bg_and_ff_1 = np.sqrt( (x1[-1] - x0_ff[-1])**2 + (y1[-1] - y0_ff[-1])**2 + (z1[-1] - z0_ff[-1])**2)
        # else:
        #     distance_between_bg_and_ff_1 = np.sqrt( (x1[-1] - x1_ff[-1])**2 + (y1[-1] - y1_ff[-1])**2 + (z1[-1] - z1_ff[-1])**2)
            
        # if distance_between_bg_and_fb_00 > distance_between_bg_and_fb_01:
        #     distance_between_bg_and_fb_1 = np.sqrt( (x1[-1] - x0_fb[-1])**2 + (y1[-1] - y0_fb[-1])**2 + (z1[-1] - z0_fb[-1])**2)
        # else:
        #     distance_between_bg_and_fb_1 = np.sqrt( (x1[-1] - x1_fb[-1])**2 + (y1[-1] - y1_fb[-1])**2 + (z1[-1] - z1_fb[-1])**2)
        
        # if distance_between_bg_and_bf_00 > distance_between_bg_and_bf_01:
        #     distance_between_bg_and_bf_1 = np.sqrt( (x1[-1] - x0_bf[-1])**2 + (y1[-1] - y0_bf[-1])**2 + (z1[-1] - z0_bf[-1])**2)
        # else:
        #     distance_between_bg_and_bf_1 = np.sqrt( (x1[-1] - x1_bf[-1])**2 + (y1[-1] - y1_bf[-1])**2 + (z1[-1] - z1_bf[-1])**2)
            
        # if distance_between_bg_and_bb_00 > distance_between_bg_and_bb_01:
        #     distance_between_bg_and_bb_1 = np.sqrt( (x1[-1] - x0_bb[-1])**2 + (y1[-1] - y0_bb[-1])**2 + (z1[-1] - z0_bb[-1])**2)
        # else:
        #     distance_between_bg_and_bb_1 = np.sqrt( (x1[-1] - x1_bb[-1])**2 + (y1[-1] - y1_bb[-1])**2 + (z1[-1] - z1_bb[-1])**2) 
        
        # if np.argmin([distance_between_bg_and_ff_1, distance_between_bg_and_fb_1, distance_between_bg_and_bf_1, distance_between_bg_and_bb_1]) == np.argmin([lls_ff[-1],lls_fb[-1],lls_bf[-1],lls_bb[-1]]):
        #     print("Both fit approaches match!")
        # else:
        #     print("Wrong")
        #     print("Closest Origin to Best Guess: ",np.argmin([distance_between_bg_and_ff_1, distance_between_bg_and_fb_1, distance_between_bg_and_bf_1, distance_between_bg_and_bb_1]), "Smallest least squares: ",np.argmin([lls_ff[-1],lls_fb[-1],lls_bf[-1],lls_bb[-1]]))
        # print(success_ff, success_fb, success_bf, success_bb, success, success_best)
        # print(lls_ff[-1],lls_fb[-1],lls_bf[-1],lls_bb[-1],lls[-1],lls_best)
    if (not success_ff) and (not success_fb) and (not success_bf) and (not success_bb):
        failed_events.append(i)
    if (lls_ff[-1] <= lls_fb[-1] and lls_ff[-1] <= lls_bf[-1] and lls_ff[-1] <= lls_bb[-1]) and lls_ff[-1] <= lls[-1]:
    # if False:
        counts_per_mev_best.append(counts_per_mev_ff[-1])
        lls_best.append(lls_ff[-1])
        message_best.append(message_ff[-1])
        success_best.append(success_ff[-1])
        theta0_best.append(theta0_ff[-1])
        theta1_best.append(theta1_ff[-1])
        phi0_best.append(phi0_ff[-1])
        phi1_best.append(phi1_ff[-1])
        x0_best.append(x0_ff[-1])
        y0_best.append(y0_ff[-1])
        z0_best.append(z0_ff[-1])
        x1_best.append(x1_ff[-1])
        y1_best.append(y1_ff[-1])
        z1_best.append(z1_ff[-1])
        E0_best.append(E0_ff[-1])
        E1_best.append(E1_ff[-1])
        sigma_xy0_best.append(sigma_xy0_ff[-1])
        sigma_z0_best.append(sigma_z0_ff[-1])
        sigma_xy1_best.append(sigma_xy1_ff[-1])
        sigma_z1_best.append(sigma_z1_ff[-1])
        nfev_best.append(nfev_ff[-1])
    elif (lls_fb[-1] <= lls_ff[-1] and lls_fb[-1] <= lls_bf[-1] and lls_fb[-1] <= lls_bb[-1]) and lls_fb[-1] <= lls[-1]:
    # if False:
        counts_per_mev_best.append(counts_per_mev_fb[-1])
        lls_best.append(lls_fb[-1])
        message_best.append(message_fb[-1])
        success_best.append(success_fb[-1])
        theta0_best.append(theta0_fb[-1])
        theta1_best.append(theta1_fb[-1])
        phi0_best.append(phi0_fb[-1])
        phi1_best.append(phi1_fb[-1])
        x0_best.append(x0_fb[-1])
        y0_best.append(y0_fb[-1])
        z0_best.append(z0_fb[-1])
        x1_best.append(x1_fb[-1])
        y1_best.append(y1_fb[-1])
        z1_best.append(z1_fb[-1])
        E0_best.append(E0_fb[-1])
        E1_best.append(E1_fb[-1])
        sigma_xy0_best.append(sigma_xy0_fb[-1])
        sigma_z0_best.append(sigma_z0_fb[-1])
        sigma_xy1_best.append(sigma_xy1_fb[-1])
        sigma_z1_best.append(sigma_z1_fb[-1])
        nfev_best.append(nfev_fb[-1])
    elif (lls_bf[-1] <= lls_ff[-1] and lls_bf[-1] <= lls_fb[-1] and lls_bf[-1] <= lls_bb[-1]) and lls_bf[-1] <= lls[-1]:
    # if False:
        counts_per_mev_best.append(counts_per_mev_bf[-1])
        lls_best.append(lls_bf[-1])
        message_best.append(message_bf[-1])
        success_best.append(success_bf[-1])
        theta0_best.append(theta0_bf[-1])
        theta1_best.append(theta1_bf[-1])
        phi0_best.append(phi0_bf[-1])
        phi1_best.append(phi1_bf[-1])
        x0_best.append(x0_bf[-1])
        y0_best.append(y0_bf[-1])
        z0_best.append(z0_bf[-1])
        x1_best.append(x1_bf[-1])
        y1_best.append(y1_bf[-1])
        z1_best.append(z1_bf[-1])
        E0_best.append(E0_bf[-1])
        E1_best.append(E1_bf[-1])
        sigma_xy0_best.append(sigma_xy0_bf[-1])
        sigma_z0_best.append(sigma_z0_bf[-1])
        sigma_xy1_best.append(sigma_xy1_bf[-1])
        sigma_z1_best.append(sigma_z1_bf[-1])
        nfev_best.append(nfev_bf[-1])
    elif (lls_bb[-1] <= lls_ff[-1] and lls_bb[-1] <= lls_fb[-1] and lls_bb[-1] <= lls_bf[-1]) and lls_bb[-1] <= lls[-1]:
    # if False:
        counts_per_mev_best.append(counts_per_mev_bb[-1])
        lls_best.append(lls_bb[-1])
        message_best.append(message_bb[-1])
        success_best.append(success_bb[-1])
        theta0_best.append(theta0_bb[-1])
        theta1_best.append(theta1_bb[-1])
        phi0_best.append(phi0_bb[-1])
        phi1_best.append(phi1_bb[-1])
        x0_best.append(x0_bb[-1])
        y0_best.append(y0_bb[-1])
        z0_best.append(z0_bb[-1])
        x1_best.append(x1_bb[-1])
        y1_best.append(y1_bb[-1])
        z1_best.append(z1_bb[-1])
        E0_best.append(E0_bb[-1])
        E1_best.append(E1_bb[-1])
        sigma_xy0_best.append(sigma_xy0_bb[-1])
        sigma_z0_best.append(sigma_z0_bb[-1])
        sigma_xy1_best.append(sigma_xy1_bb[-1])
        sigma_z1_best.append(sigma_z1_bb[-1])
        nfev_best.append(nfev_bb[-1])
    # if lls_ff[-1] == lls_fb[-1] == lls_bf[-1] == lls_bb[-1] or (lls[-1] < lls_ff[-1] and lls[-1] < lls_fb[-1] and lls[-1] < lls_bf[-1] and lls[-1] < lls_bb[-1]):
    elif lls[-1] <= lls_ff[-1] and lls[-1] <= lls_fb[-1] and lls[-1] <= lls_bf[-1] and lls[-1] <= lls_bb[-1]:
        counts_per_mev_best.append(-1)
        lls_best.append(lls[-1])
        message_best.append(message[-1])
        success_best.append(success[-1])
        theta0_best.append(theta0[-1])
        theta1_best.append(theta1[-1])
        phi0_best.append(phi0[-1])
        phi1_best.append(phi1[-1])
        x0_best.append(x0[-1])
        y0_best.append(y0[-1])
        z0_best.append(z0[-1])
        x1_best.append(x1[-1])
        y1_best.append(y1[-1])
        z1_best.append(z1[-1])
        E0_best.append(E0[-1])
        E1_best.append(E1[-1])
        sigma_xy0_best.append(sigma_xy[-1])
        sigma_z0_best.append(sigma_z[-1])
        sigma_xy1_best.append(sigma_xy[-1])
        sigma_z1_best.append(sigma_z[-1])
        nfev_best.append(nfev[-1])
        

    # if the fit terminated abnormally, view the fit particle by particle
    # if not success_best[-1]:
    if False:
        # print("Event",i,"failed: ",success_best[-1])
        # print(message_best[-1])
    # if data[0].message != "`callback` raised `StopIteration`.":
        trace_sim = build_sim.create_multi_particle_event('e24joe', 124, i, ['4He','4He'])
        trace_sim.per_particle_params = ['initial_energy', 'theta', 'phi', 'sigma_xy', 'sigma_z', 'num_stopping_power_points','initial_point'] 
        trace_sim.shared_params = ['gas_density']
        trace_sim.sims[0].adaptive_stopping_power = False
        trace_sim.sims[0].initial_energy = E0_best[-1]
        trace_sim.sims[0].theta, trace_sim.sims[0].phi = theta0_best[-1], phi0_best[-1]
        trace_sim.sims[0].num_stopping_power_points = trace_sim.sims[0].get_num_stopping_points_for_energy(E0_best[-1])
        trace_sim.sims[0].initial_point = (x0_best[-1],y0_best[-1],z0_best[-1])
        trace_sim.sims[0].sigma_xy = sigma_xy_best[-1]
        trace_sim.sims[0].sigma_z = sigma_z_best[-1]
        trace_sim.sims[1].adaptive_stopping_power = False
        trace_sim.sims[1].initial_energy = E1_best[-1]
        trace_sim.sims[1].theta, trace_sim.sims[1].phi = theta1_best[-1], phi1_best[-1]
        trace_sim.sims[1].num_stopping_power_points = trace_sim.sims[1].get_num_stopping_points_for_energy(E1_best[-1])
        trace_sim.sims[1].initial_point = (x1_best[-1],y1_best[-1],z1_best[-1])
        trace_sim.sims[1].sigma_xy = sigma_xy_best[-1]
        trace_sim.sims[1].sigma_z = sigma_z_best[-1]
        trace_sim.simulate_event()
        # trace_sim.plot_residuals()
        # trace_sim.plot_residuals_3d(threshold=25)
        # trace_sim.plot_real_data_3d(threshold=50)
        # trace_sim.plot_simulated_3d_data(threshold=25)
        x,y,z,e = trace_sim.get_xyze(traces=trace_sim.get_residuals())
        # print('residual count sum:', np.sum(e))
        residual_count_sum.append(np.sum(e))
        x,y,z,e = trace_sim.get_xyze(traces=trace_sim.traces_to_fit)
        # print('observed data sum:', np.sum(e))
        observed_trace_sum.append(np.sum(e))
        # plt.show()
        
# print("Parallel processing:")
# start_time = time.time()
# manager = multiprocessing.Manager()
# fit_results = manager.dict()  # shared dictionary

# with multiprocessing.Pool(processes=100) as pool:
#     for result in tqdm(pool.imap_unordered(add_event_to_dict,completed_fit_events)):
#         pass

# end_time = time.time()
# print(f"Parallel time: {end_time - start_time:.4f} seconds")
# fit_results['event'] = completed_fit_events
# fit_results['success_ff'] = success_ff
# fit_results['message_ff'] = message_ff
# fit_results['fun_ff'] = lls_ff
# fit_results['theta0_ff'] = theta0_ff
# fit_results['phi0_ff'] = phi0_ff
# fit_results['x0_ff'] = x0_ff
# fit_results['y0_ff'] = y0_ff
# fit_results['z0_ff'] = z0_ff
# fit_results['x1_ff'] = x1_ff
# fit_results['y1_ff'] = y1_ff
# fit_results['z1_ff'] = z1_ff
# fit_results['e0_ff'] = E0_ff
# fit_results['e1_ff'] = E1_ff
# fit_results['sigma_xy0_ff'] = sigma_xy0_ff
# fit_results['sigma_z0_ff'] = sigma_z0_ff
# fit_results['sigma_xy1_ff'] = sigma_xy1_ff
# fit_results['sigma_z1_ff'] = sigma_z1_ff
# fit_results['counts_per_mev_ff'] = counts_per_mev_ff
# fit_results['nfev_ff'] = nfev_ff
# fit_results['residuals_ff'] = 

# print("Percentage of events terminated as expected: ",message.count("`callback` raised `StopIteration`.")/(len(message)))
# print("Percentage of events ABNORMAL: ",message.count("ABNORMAL_TERMINATION_IN_LNSRCH")/(len(message)))
# print("Percentage of events other: ",(message.count("ABNORMAL_TERMINATION_IN_LNSRCH") + message.count("`callback` raised `StopIteration`."))/(len(message)))
# for i in range(len(success)):
#     print("ff: ",success_ff[i], message_ff[i], lls_ff[i],"fb: ",success_fb[i], message_fb[i], lls_fb[i],"bf: ",success_bf[i], message_bf[i], lls_bf[i],"bb: ",success_bb[i], message_bb[i], lls_bb[i],)
print("Number of events fitted based on lls appended: ",len(lls_best))
print("Number of events looked up in ff: ",len(lls_ff))
print("Number of events looked up in fb: ",len(lls_fb))
print("Number of events looked up in bf: ",len(lls_bf))
print("Number of events looked up in bb: ",len(lls_bb))
print("Number of events looked up in cluster: ",len(lls))
print("Failed Events: ",failed_events)
print(set(message_best))

with open("/egr/research-tpc/dopferjo/gadget_analysis/fit_results/least_squares/diff_sigmas_prev_results_fit_dict_all_dir_and_kmeans.pkl", 'wb') as f:
    pickle.dump(fit_results, f)

# np.savez("/egr/research-tpc/dopferjo/gadget_analysis/fit_results/least_squares/best_fit_arrays.npz",evts=evts,
#          counts_per_mev=counts_per_mev_best,
#          fun=lls_best,
#          message=message_best,
#          success=success_best,
#          theta0=theta0_best,
#          theta1=theta1_best,
#          phi0=phi0_best,
#          phi1=phi1_best,
#          x0=x0_best,
#          y0=y0_best,
#          z0=z0_best,
#          x1=x1_best,
#          y1=y1_best,
#          z1=z1_best,
#          E0=E0_best,
#          E1=E1_best,
#          sigma_xy = sigma_xy_best,
#          sigma_z = sigma_z_best,
#          nfev = nfev_best)

print("Percentage of events terminated with success: ",success_best.count(True)/(len(success_best)))
print(success_best.count(True))
print(len(success_best))

bad_residuals = [residual for residual, mess in zip(residual_count_sum,success_best) if not mess] 
good_residuals = [residual for residual, mess in zip(residual_count_sum,success_best) if mess]
print("This should be the range of the residual plot: ",np.min(residual_count_sum), np.max(residual_count_sum))
# 1D Histogram of counts_per_mev
plt.hist(residual_count_sum,color="blue",bins=500,zorder=1,range=(-5.1e6,5.1e6))
plt.hist(bad_residuals, color="red",bins=500,range=(-5.1e6,5.1e6),zorder=3, label="Minimization Exited with Success = False")
plt.hist(good_residuals, color="green",bins=500,range=(-5.1e6,5.1e6),zorder=2, label = "Minimization Exited with Success = True")
plt.title("Trace Residuals Dist\n mean, std = %d,%d"%(np.mean(residual_count_sum),np.std(residual_count_sum)))
plt.xlabel('Summed Residuals')
plt.ylabel('Counts')
# plt.yscale('log')
plt.show()

bad_CpM = [CpM for CpM, mess in zip(counts_per_mev_best,success_best) if not mess] 
good_CpM = [CpM for CpM, mess in zip(counts_per_mev_best,success_best) if mess]
print("This should be the range of the CpM plot: ",np.min(counts_per_mev_best), np.max(counts_per_mev_best))
# 1D Histogram of counts_per_mev
plt.hist(counts_per_mev_best,color="blue",bins=500,zorder=1,range=(-2,4e5))
plt.hist(bad_CpM, color="red",bins=500,range=(-2,4e5),zorder=3, label="Minimization Exited with Success = False")
plt.hist(good_CpM, color="green",bins=500,range=(-2,4e5),zorder=2, label = "Minimization Exited with Success = True")
plt.title("Counts Per MeV Dist\n mean, std = %d,%d"%(np.mean(counts_per_mev_best),np.std(counts_per_mev_best)))
plt.xlabel('Counts per MeV')
plt.ylabel('Counts')
# plt.yscale('log')
plt.show()

# 1D Histogram of lls
plt.hist(lls_best,bins=200,range=(0,0.5e10),zorder=1, label="Best of All Four Direction Fits (mean, std.: %.4e,%.4e)"%(np.mean(lls_best),np.std(lls_best)),fill=False, histtype='step',color="green")
plt.xlabel('Minimized Least Squares')
plt.hist(lls,   bins=200,range=(0,0.5e10),zorder=3, label="Best Initial Guess (mean, std.: %.4e,%.4e)"%(np.mean(lls),np.std(lls)),fill=False, histtype='step', color="red")
plt.hist(lls_ff,bins=200,range=(0,0.5e10),zorder=2, label = "ff_direction (mean, std.: %.4e,%.4e)"%(np.mean(lls_ff),np.std(lls_ff)),fill=False, histtype='step', color="blue")
plt.hist(lls_fb,bins=200,range=(0,0.5e10),zorder=2, label = "fb_direction (mean, std.: %.4e,%.4e)"%(np.mean(lls_fb),np.std(lls_fb)),fill=False, histtype='step', color="orange")
plt.hist(lls_bf,bins=200,range=(0,0.5e10),zorder=2, label = "bf_direction (mean, std.: %.4e,%.4e)"%(np.mean(lls_bf),np.std(lls_bf)),fill=False, histtype='step', color="yellow")
plt.hist(lls_bb,bins=200,range=(0,0.5e10),zorder=2, label = "bb_direction (mean, std.: %.4e,%.4e)"%(np.mean(lls_bb),np.std(lls_bb)),fill=False, histtype='step', color="violet")
plt.legend(fontsize=14)
plt.ylabel('Counts')
# plt.yscale('log')
plt.show()

# bad_lls = [ll for ll, mess in zip(lls,message) if mess == "ABNORMAL_TERMINATION_IN_LNSRCH"] 
# good_lls = [ll for ll, mess in zip(lls,message) if mess == "`callback` raised `StopIteration`."]
bad_lls = [ll for ll, mess in zip(lls_best,success_best) if not mess] 
good_lls = [ll for ll, mess in zip(lls_best,success_best) if mess]
# 1D Histogram of lls
# plt.hist(lls,color="blue",bins=100,range=(0,1e10),zorder=1, label="all fits")
plt.hist(lls_best,color="blue",bins=100,range=(0,1e10),zorder=1, label="all fits")
plt.xlabel('Minimized Log Likelihood')
plt.hist(bad_lls, color="red",bins=100,range=(0,1e10),zorder=3, label="Minimization Exited with Success = False")
plt.hist(good_lls, color="green",bins=100,range=(0,1e10),zorder=2, label = "Minimization Exited with Success = True")
plt.legend()
plt.ylabel('Counts')
plt.yscale('log')
plt.show()

# angles = []

# for i in range(len(theta0)):
#     angles.append(np.arccos(np.sin(theta0_best[i]) * np.sin(theta1_best[i]) * np.cos(phi0_best[i]) * np.cos(phi1_best[i]) + np.sin(theta0_best[i]) * np.sin(theta1_best[i]) * np.sin(phi0_best[i]) * np.sin(phi1_best[i]) + np.cos(theta0_best[i]) * np.cos(theta1_best[i])))

dxy = np.zeros_like(x0)
dzs = np.zeros_like(z0)
dxy_gate = np.array([])
dz_gate = np.array([])
events_to_check = []

angles = []
angles_gate = []

categorized_events_of_interest = pd.read_csv('./complete_categorized_events_of_interest.csv',\
    encoding='utf-8-sig', skip_blank_lines = False, nrows = 36164, header=None)

array_of_categorized_events_of_interest = categorized_events_of_interest[0].to_numpy()
mask = np.isin(array_of_categorized_events_of_interest, ['Double Alpha Candidate'])# ['RnPo Chain', 'Accidental Coin', 'Double Alpha Candidate'])
events = np.where(mask)[0]
print("Events manually assigned Double Alpha Candidate: ")
print(events)


for i in range(len(x0)):
    dxy[i] = np.sqrt((x0_best[i] - x1_best[i])**2 + (y0_best[i] - y1_best[i])**2)
    dzs[i] = np.abs(z0_best[i]-z1_best[i])
    angles.append(np.arccos(np.sin(theta0[i]) * np.sin(theta1[i]) * np.cos(phi0[i]) * np.cos(phi1[i]) + np.sin(theta0[i]) * np.sin(theta1[i]) * np.sin(phi0[i]) * np.sin(phi1[i]) + np.cos(theta0[i]) * np.cos(theta1[i])))
    if dxy[i] < 10 and dzs[i] < 10:
        angles_gate.append(angles[i])
    if angles[i] > 2.96706: # greater than 170 deg
        dxy_gate = np.append(dxy_gate,dxy[i])
        dz_gate = np.append(dz_gate,dzs[i])
        if dxy[i] < 20 and dzs[i] < 20:
            events_to_check.append(evts[i])
            # print(success[i])
            
    # both of these angle calculations are the same
    # print("old - new angle calc: ", (np.arccos(np.sin(theta0[i]) * np.sin(theta1[i]) * np.cos(phi0[i] - phi1[i]) + np.cos(theta0[i]) * np.cos(theta1[i])) - \
    #     np.arccos(np.sin(theta0[i]) * np.sin(theta1[i]) * np.cos(phi0[i]) * np.cos(phi1[i]) + np.sin(theta0[i]) * np.sin(theta1[i]) * np.sin(phi0[i]) * np.sin(phi1[i]) + np.cos(theta0[i]) * np.cos(theta1[i]))))

print("Events that looks like Double Alpha Candidate after fitting: ")
print(events_to_check)
# print(events[:20])
# print(events_to_check)
# for index in events_to_check:
#     print(index, events[index], angles[index], dxy[index], dzs[index])
#     print("Individual Angle Calc.: ", theta0[index],phi0[index], theta1[index],phi1[index])

# print(len(dxy_gate))
# print(len(dz_gate))


# Create 2D histogram (counts in bins)
bins = 20 # Number of bins for both x and y
hist, xedges, yedges = np.histogram2d(dxy, dzs, bins=bins) # change this to dxy, dzs for not gate, and dxy_gate and dz_gate for gate

# Prepare data for bar3d
# Construct arrays for the anchor positions of the bars
xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0 # All bars start at z=0

# Calculate bar dimensions
dx = xedges[1] - xedges[0] # Width of each bar in x-direction
dy = yedges[1] - yedges[0] # Width of each bar in y-direction
dz = hist.ravel() # Height of each bar (counts)

# Create the 3D plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

cmap = cm.get_cmap('jet') # Get desired colormap - you can change this!
max_height = np.max(dz)   # get range of colorbars so we can normalize
min_height = np.min(dz)
# scale each z to [0,1], and get their rgb values
rgba = [cmap((k-min_height)/max_height) for k in dz] 

# Plot the 3D bars
ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color = rgba, zsort='average')

# Set labels
ax.set_xlabel('dxy',labelpad=20)
ax.set_ylabel('dz',labelpad=20)
ax.set_zlabel('Counts',labelpad=20)
ax.set_title('Distance Between Origins')
plt.show()

# 2D Histogram of Origins
plt.hist2d(dxy,dzs, bins=(20,20))# , cmap='viridis')
plt.xlabel('dxy')
plt.ylabel('dz')
# plt.plot(z1, color = 'red')
# plt.plot(z0, color = 'blue')
plt.title('Distance Between Two Particles',pad=10)
plt.colorbar()
plt.show()

# 1D Histogram of Angles
plt.hist(angles, bins=50)# , cmap='viridis')
plt.xlabel('Angle Between Alphas (radians)')
plt.ylabel('Counts')
# plt.plot(z1, color = 'red')
# plt.plot(z0, color = 'blue')
# plt.colorbar()
plt.show()

# 1D Histogram of Angles
plt.hist(angles_gate, bins=50)# , cmap='viridis')
plt.xlabel('Angle Between Alphas (dxy < 10, dz < 10)')
plt.ylabel('Counts')
# plt.plot(z1, color = 'red')
# plt.plot(z0, color = 'blue')
# plt.colorbar()
plt.show()