import sys
import importlib
import multiprocessing
from raw_viewer import process_runs

def import_or_reload(module_name):
    if module_name in sys.modules:
        module_obj = sys.modules[module_name]
        importlib.reload(module_obj)
    else:
        importlib.import_module(module_name)

def rve():
    import_or_reload('raw_viewer.plots.rve')

def rates():
    import_or_reload('raw_viewer.plots.rates')
    
def rate_summary():
    import_or_reload('raw_viewer.plots.rate_summary')

def xy_centering():
    import_or_reload('raw_viewer.plots.xy_centering')

experiment = 'e25058'

def process_run(run):
    try:
        process_runs.get_processed_run(experiment, run, True)
        print('processed run ', run)
    except Exception as e:
        print('error processing run ', run)
        print(e)

if __name__ == '__main__':
    #runs = [49]#range(61,74)#[74,75,76,77]
    from e23035_analysis.e23035_runs import run_df
    import numpy as np
    #runs = run_df['GET'][np.isfinite(run_df['GET'])]
    runs=[70,71,72,76,77,78,79,80,81,82,83]#49, 61,62,63,64,65]
    with multiprocessing.Pool(50) as pool:
        pool.map(process_run, runs)