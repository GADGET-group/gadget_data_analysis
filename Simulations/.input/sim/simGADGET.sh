#!/bin/bash

# set paths for ATTPCROOT and Automation scripts
automation_dir=$(dirname "$(readlink -f "$0")") # get parent directory of script
automation_dir=$(readlink -f "$automation_dir" | sed 's:\([^/]\)$:\1/:') # add trailing slash if not present

# define functions
status_file () {
    # remove all tmp files and create new one
    rm -f *.tmp
    touch $1
}

status_file "STARTING.tmp"

# check for ATTPCROOT installed in same directory, download if not
if [ ! -f $automation_dir"ATTPCROOTv2/env_fishtank.sh" ]; then
    status_file "CLONING_ATTPCROOT.tmp"
    git clone https://github.com/ATTPC/ATTPCROOTv2
    cd ATTPCROOTv2
    git checkout 40699a2
    rm -rf .git
    cd $automation_dir
    attpcroot_dir=$automation_dir"ATTPCROOTv2/"
else # ATTPCROOT already installed
    attpcroot_dir=$automation_dir"ATTPCROOTv2/"
fi

source $attpcroot_dir"env_fishtank.sh"

if [ ! -f $attpcroot_dir"build/Makefile" ]; then
    # steps pulled from ATTPCROOTv2 installation wiki page
    status_file "CMAKING_ATTPCROOT.tmp"
    cd $attpcroot_dir
    source env_fishtank.sh
    mkdir build
    cd build
    cmake -DCMAKE_PREFIX_PATH=/mnt/simulations/attpcroot/fair_install_18.6/ ../
    cd $automation_dir
    status_file "MAKING_ATTPCROOT.tmp"
    cd $attpcroot_dir"build/"
    make install -j4
    source config.sh
    cd $automation_dir
    status_file "GENERATING_GEOMETRY.tmp"
    cd $attpcroot_dir"geometry/"
    nohup root -b -l GADGET_II.C
    pid1=$!
    wait $pid1
    cd $automation_dir
    mkdir -p $attpcroot_dir"macro/Simulation/Charge_Dispersion/data"
fi

# load prerequisites for ATTPCROOT
source $attpcroot_dir"env_fishtank.sh"
module load fairroot/18.6.3

# create output directory if it doesn't exist
mkdir -p $automation_dir"out"

touch $automation_dir"geo.temp" # always update geometry on first run

# SIMULATION LOOP 
while true; do
	# queue new simulation parameters or break loop
	python3 $automation_dir"queue.py" $automation_dir $attpcroot_dir

    if [ -f $automation_dir"STOP.tmp" ]; then
        status_file "STOPPED.tmp"
        exit 0
    fi
    
    if [ -f ../STOP.tmp ]; then
        status_file "STOPPED.tmp"
        exit 0
    fi
    
    # build ATTPCROOT and run simulation
    status_file "REBUILDING.tmp"
    make -C $attpcroot_dir"build/" -j8
    
    if [ -f "geo.temp" ]; then
        # update geometry if geo.temp exists
        rm "geo.temp"
        status_file "UPDATING_GEOMETRY.tmp"
        cd $attpcroot_dir"geometry/"
        nohup root -b -l GADGET_II.C
        pid1=$!
        wait $pid1
        cd $automation_dir
    fi
    
    status_file "SIMULATING.tmp"
    cd $attpcroot_dir"macro/Simulation/Charge_Dispersion/"
    echo "Running simulation (Mg20_test_sim_pag.C)"
    nohup root -b -l Mg20_test_sim_pag.C
    pid1=$!
    wait $pid1
    cd $automation_dir
    status_file "DIGITIZATION.tmp"
    cd $attpcroot_dir"macro/Simulation/Charge_Dispersion/"
    nohup root -b -l rundigi_sim_CD.C
    pid2=$!
    wait $pid2
    cd $automation_dir
    mv $attpcroot_dir"macro/Simulation/Charge_Dispersion/data/output.h5" $automation_dir"out/output.h5"
    
    python3 $automation_dir"process_h5.py" $automation_dir $attpcroot_dir
done