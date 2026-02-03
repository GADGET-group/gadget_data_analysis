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

experiment = 'e23035_prep_vault'

def process_run(run):
    process_runs.get_processed_run(experiment, run, True)

if __name__ == '__main__':
    runs = [49]#range(61,74)#[74,75,76,77]
    with multiprocessing.Pool(10) as pool:
        pool.map(process_run, runs)