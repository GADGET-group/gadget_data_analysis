# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 15:09:07 2025

@author: alexn
"""

import numpy as np
import scipy.optimize as opt
 

def get_efficiency(implant_time, half_life, dead_time_start, dead_time_end,decay_time):
    total_cycle_time = implant_time+decay_time
    measure_time = total_cycle_time - dead_time_start - dead_time_end - implant_time
    decay_constant = np.log(2)/half_life
    rate = 1
    def number_after(time, N0, R):
        return (N0 - R/decay_constant)*np.exp(-time*decay_constant) + R/decay_constant
    stop_time = 10000*half_life
    t = 0
    N = 0
    decays_measured = 0
    total_implant_time = 0
    while t < stop_time:
        #implant_cycle
        N = number_after(implant_time, N, rate)
        t += implant_time
        total_implant_time += implant_time
        #dead_time
        N = number_after(dead_time_start, N, 0)
        t += dead_time_start
        #measurment_time
        N_old = N
        N = number_after(measure_time, N, 0)
        t += measure_time
        decays_measured += N_old - N
        #dead_time
        N = number_after(dead_time_end, N, 0)
        t += dead_time_end
    return decays_measured/(rate*total_implant_time)

   

def get_best_settings(half_life, dead_time_start, dead_time_end):
    to_min = lambda x: -get_efficiency(x[0], half_life, dead_time_start, dead_time_end, x[1])
    x0 = (half_life, half_life)
    return opt.fmin(to_min, x0)

def get_best_settings_fixed_decay_time(half_life,  dead_time_start, dead_time_end, decay_time):
    to_min = lambda x: -get_efficiency(x, half_life, dead_time_start, dead_time_end, decay_time)
    x0 = half_life
    return opt.fmin(to_min, x0)

def get_best_settings_equal_on_off(half_life, dead_time_start, dead_time_end):
    to_min = lambda x: -get_efficiency(x[0], half_life, dead_time_start, dead_time_end, 0.5)
    x0 = (half_life,)
    return opt.fmin(to_min, x0)

Ga60_half_life = 69.4e-3
Ge61_half_life = 44e-3
Zn59_half_life = 182e-3
Mg20_half_life = 90e-3
Mg21_half_life = 120e-3
dead_time_start = 12e-3
dead_time_end = 2e-3
 
if False:
    implant_time, decay_time = get_best_settings(Mg20_half_life, dead_time_start, dead_time_end)
    print('optimum implant time and decay time = %f s, %f s'%(implant_time, decay_time))
    print('efficiencies with optimum times')
    print('61Ge efficiency = ', get_efficiency(implant_time, Ge61_half_life, dead_time_start, dead_time_end, decay_time))
    print('Zn59 efficiency = ', get_efficiency(implant_time, Zn59_half_life, dead_time_start, dead_time_end, decay_time))
    print('60Ga efficiency = ', get_efficiency(implant_time, Ga60_half_life, dead_time_start, dead_time_end, decay_time))
    print('20Mg efficiency = ', get_efficiency(implant_time, Mg20_half_life, dead_time_start, dead_time_end, decay_time))

implant_time = 100e-3
decay_time = implant_time

('60Ga efficiency = ', get_efficiency(implant_time, Ga60_half_life, dead_time_start, dead_time_end, decay_time))

print('efficiencies with %f s implant time and %f s decay time'%(implant_time, decay_time))
print('61Ge efficiency = ', get_efficiency(implant_time, Ge61_half_life, dead_time_start, dead_time_end, decay_time))
print('Zn59 efficiency = ', get_efficiency(implant_time, Zn59_half_life, dead_time_start, dead_time_end, decay_time))
print('60Ga efficiency = ', get_efficiency(implant_time, Ga60_half_life, dead_time_start, dead_time_end, decay_time))
print('20Mg efficiency = ', get_efficiency(implant_time, Mg20_half_life, dead_time_start, dead_time_end, decay_time))

if False:
    decay_time = 300e-3
    half_life = Mg20_half_life
    implant_time = get_best_settings_fixed_decay_time(half_life, dead_time_start, dead_time_end, decay_time)
    print('optimum implant time for %f s decay time is %f s'%(decay_time, implant_time))
    print('efficiencies with these settings:')

    print('20Mg efficiency = ', get_efficiency(implant_time, Mg20_half_life, dead_time_start, dead_time_end, decay_time))