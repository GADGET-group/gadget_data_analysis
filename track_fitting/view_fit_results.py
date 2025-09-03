import pickle
import matplotlib.pyplot as plt
import numpy as np

evts, theta0,theta1, phi0,phi1, x0,y0,z0, x1,y1,z1, lls, cats, E0,E1, Erecs, nfev, sigma_xy, sigma_z = [], [],[], [],[], [],[],[], [],[],[], [], [], [],[], [], [], [],[]
trace_sims = []

file_path = '/egr/research-tpc/dopferjo/gadget_analysis/two_particle_decays_in_e24joe_test.dat' #TODO: double-check this is the right file name
with open(file_path, 'rb') as file:
    fit_results_dict = pickle.load(file)
    for key in fit_results_dict:
        res, sim = fit_results_dict[key]
        if res.fun == np.inf:
            print("Event %d exited with inf ll!"%key)
        if res.success == False:
            print(res.message)
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

