import sys
import importlib
import multiprocessing
import os

from raw_viewer import process_runs, ddas_interface

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

experiment = 'e23035'

def process_run(run):
    try:
        process_runs.get_processed_run(experiment, run, True)
        print('processed run ', run)
    except Exception as e:
        print('error processing run ', run)
        print(e)

def merge_ddas(ddas_run, remerge=True):
    try:
        if os.path.exists(ddas_interface.get_merged_root_file_path(ddas_run)) and not remerge:
            print('ddas run %d already merged\n')
        else:
            print('merging ddas run %d'%ddas_run)
            ddas_interface.make_merged_root_file(ddas_run)
            print('finished merging ddas run %d'%ddas_run)
    except Exception as e:
        print('error merging ddas run ', ddas_run)
        print(e)
if __name__ == '__main__':
    #runs = [49]#range(61,74)#[74,75,76,77]
    from e23035_analysis.e23035_runs import run_df
    import numpy as np
    if True:
        ddas_runs = range(0, 287)
        with multiprocessing.Pool(200) as pool:
            pool.map(merge_ddas, ddas_runs)
    if False:
        #runs = run_df['GET'][np.isfinite(run_df['GET'])]
        runs=[131,177,178,180,214,215,221,229,234,235,257,266,271,274]#49, 61,62,63,64,65]
        with multiprocessing.Pool(50) as pool:
            pool.map(process_run, runs)