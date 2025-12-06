import sys
import importlib
import multiprocessing
from raw_viewer import process_runs

experiment = 'e25058'

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

def process_run(run):
    process_runs.get_processed_run(experiment, run)

if __name__ == '__main__':
    runs = range(157, 300)#[74,75,76,77]
    with multiprocessing.Pool(10) as pool:
        pool.map(process_run, runs)