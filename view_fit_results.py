import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from track_fitting.SimulatedEvent import SimulatedEvent
from track_fitting. MultiParticleEvent import MultiParticleEvent
import raw_viewer.raw_h5_file as raw_h5_file
from track_fitting import SingleParticleEvent, build_sim

fitted = False

evts, theta0,theta1, phi0,phi1, x0,y0,z0, x1,y1,z1, lls, cats, E0,E1, Erecs, nfev, sigma_xy, sigma_z = [], [],[], [],[], [],[],[], [],[],[], [], [], [],[], [], [], [],[]
trace_sims = []

file_path = '/egr/research-tpc/dopferjo/gadget_analysis/two_particle_decays_in_e24joe_test.dat' #TODO: double-check this is the right file name
file_path = '/egr/research-tpc/dopferjo/gadget_analysis/two_particle_decays_in_e24joe_best_direction_20_events.dat'
file_path = '/egr/research-tpc/dopferjo/gadget_analysis/two_particle_decays_in_e24joe_no_fit.dat'
with open(file_path, 'rb') as file:
    fit_results_dict = pickle.load(file)
    for key in fit_results_dict:
        # if key < 10:
        #     print("Event ", key)
        #     print(fit_results_dict[key])
        if fitted:
            res, sim = fit_results_dict[key]
            if res.fun == np.inf:
                print("Event %d exited with inf ll!"%key)
            # if res.success == True:
            # print(res.message)
            # elif res.success == True:
            theta0.append(res.x[0] * np.pi)
            theta1.append(res.x[1] * np.pi)
            phi0.append(res.x[2] * 2 * np.pi)
            phi1.append(res.x[3] * 2 * np.pi)
            x0.append(res.x[4]* 40)
            y0.append(res.x[5] * 40)
            z0.append(res.x[6] * 400)
            x1.append(res.x[7]* 40)
            y1.append(res.x[8] * 40)
            z1.append(res.x[9] * 400)
            E0.append(res.x[10] * 10)
            E1.append(res.x[11] * 10)
            sigma_xy.append(res.x[12] * 10)
            sigma_z.append(res.x[13] * 10)
            lls.append(res.fun)
            evts.append(key)
            nfev.append(res.nfev)
            trace_sim = build_sim.create_multi_particle_event('e24joe', 124, key, ['4He','4He'])
            trace_sim.per_particle_params = ['initial_energy', 'theta', 'phi', 'sigma_xy', 'sigma_z', 'num_stopping_power_points','initial_point'] 
            trace_sim.shared_params = ['gas_density']
            trace_sim.sims[0].adaptive_stopping_power = False
            trace_sim.sims[1].adaptive_stopping_power = False
            trace_sim.sims[0].initial_energy = E0[-1]
            trace_sim.sims[1].initial_energy = E1[-1]
            trace_sim.sims[0].theta, trace_sim.sims[0].phi, trace_sim.sims[1].theta, trace_sim.sims[1].phi = theta0[-1], phi0[-1], theta1[-1], phi1[-1]

            trace_sim.sims[0].num_stopping_power_points = trace_sim.sims[0].get_num_stopping_points_for_energy(E0[-1])
            trace_sim.sims[1].num_stopping_power_points = trace_sim.sims[1].get_num_stopping_points_for_energy(E1[-1])
            trace_sim.sims[0].initial_point = (x0[-1],y0[-1],z0[-1])
            trace_sim.sims[1].initial_point = (x1[-1],y1[-1],z1[-1])
            trace_sim.sims[0].sigma_xy, trace_sim.sims[1].sigma_xy = sigma_xy[-1], sigma_xy[-1]
            trace_sim.sims[0].sigma_z, trace_sim.sims[1].sigma_z = sigma_z[-1], sigma_z[-1]
            print(trace_sim.sims[0].sigma_xy)
            print(trace_sim.sims[1].sigma_xy) 
            # trace_sim.plot_residuals()
            # trace_sim.plot_residuals_3d(threshold=25)
            # trace_sim.plot_simulated_3d_data(threshold=25)
            plt.show(block=True)
        
        elif not fitted:
            theta0.append(fit_results_dict[key][0])
            theta1.append(fit_results_dict[key][1])
            phi0.append(fit_results_dict[key][2])
            phi1.append(fit_results_dict[key][3])
            x0.append(fit_results_dict[key][4])
            y0.append(fit_results_dict[key][5])
            z0.append(fit_results_dict[key][6])
            x1.append(fit_results_dict[key][7])
            y1.append(fit_results_dict[key][8])
            z1.append(fit_results_dict[key][9])
            E0.append(fit_results_dict[key][10])
            E1.append(fit_results_dict[key][11])
            sigma_xy.append(fit_results_dict[key][12])
            sigma_z.append(fit_results_dict[key][13])
    # thetas.append(res.x[0])
    # phis.append(res.x[1])
    # xs.append(res.x[2])
    # ys.append(res.x[3])
    # zs.append(res.x[4])
    # Es.append(res.x[5])
    # sigma_xys.append(res.x[6])
    # sigma_zs.append(res.x[7])
    # lls.append(res.fun)
    # cats.append(cat)
    # evts.append(evt)
    # nfev.append(res.nfev)
        # print(key)
        # print(res.x)
        # print(theta0)
        # print(theta1)
        # print(phi0)
        # print(phi1)
        # print(x0)
        # print(y0)
        # print(z0)
        # print(x1)
        # print(y1)
        # print(z1)
        # print(E0)
        # print(E1)
dxy = np.zeros_like(x0)
dzs = np.zeros_like(z0)
print(len(dxy))
print(len(dzs))
for i in range(len(x0)):
    dxy[i] = np.sqrt((x0[i] - x1[i])**2 + (y0[i] - y1[i])**2)
    dzs[i] = np.abs(z0[i]-z1[i])
angles = []
for i in range(len(theta0)):
    angles.append(np.arccos(np.sin(theta0[i]) * np.sin(theta1[i]) * np.cos(phi0[i] - phi1[i]) + np.cos(theta0[i]) * np.cos(theta1[i])))

# Create 2D histogram (counts in bins)
bins = 20 # Number of bins for both x and y
hist, xedges, yedges = np.histogram2d(dxy, dzs, bins=bins)

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
# ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color = rgba, zsort='average')

# Set labels
# ax.set_xlabel('dxy')
# ax.set_ylabel('dz')
# ax.set_zlabel('Counts')
# ax.set_title('Distance Between Origins')
# plt.show()

print(len(dxy))
print(len(dzs))
plt.hist2d(dxy,dzs, bins=(10,20))# , cmap='viridis')
plt.xlabel('dxy')
plt.ylabel('dz')
# plt.plot(z1, color = 'red')
# plt.plot(z0, color = 'blue')
plt.colorbar()
plt.show()