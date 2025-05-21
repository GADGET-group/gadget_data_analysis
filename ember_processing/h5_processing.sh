#!/usr/bin/env bash

# sends user email at end of job
#SBATCH --mail-type=END

# memory is counted in MB
#SBATCH --mem=1024

# number of cores to reserve
#SBATCH --ntasks-per-node=4

# output the environment of running job
printenv

echo 'Testing Run 555'

# python processing_script.py /mnt/daqtesting/protondetector/h5/run_0555.h5 /mnt/projects/e21072/OfflineAnalysis/analysis_scripts/joe/gadget_analysis/raw_viewer/channel_mappings/flatlookup2cobos.csv /mnt/projects/e21072/OfflineAnalysis/analysis_scripts/joe/gadget_analysis/raw_viewer/gui_configs/p10_2000torr.gui_ini

echo 'End Test'
# sleep so job stays active for human scale of time
sleep 3
