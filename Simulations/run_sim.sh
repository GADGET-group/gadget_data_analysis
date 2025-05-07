#!/bin/bash

### GADGET2 ATTPCROOT Automation Script ###
automation_dir=$(dirname "$(readlink -f "$0")") # get parent directory of script
automation_dir=$(readlink -f "$automation_dir" | sed 's:\([^/]\)$:\1/:') # add trailing slash if not present

#####################################
#### READ COMMAND LINE ARGUMENTS ####
num_simulators=4
tuning=0
reset_params=0
var_params=0
premade=0
zip_output=0

while getopts "htvdcakm:p" option; do
    case ${option} in
        h ) # display help
            if [ -f "$automation_dir/README.md" ]; then
                cat "$automation_dir/README.md"
            else
                echo "README.md file not found in $automation_dir"
            fi
            exit 0
            ;;
        t ) # run simulation in tuning mode
            echo "tuning mode enabled"
            tuning="y"
            ;;
        v ) # generate parameters with variation script
            var_params="y"
            ;;
        d ) # parameter debug mode
            echo "parameters.csv will be reset at end of loop and sim0 log file will be printed for debugging"
            reset_params="y"
            ;;
        c ) # clean output before running
            echo "Cleaning Output Directory..."
            rm -rf $automation_dir"Output/"
            ;;
        a ) # force reset of ATTPCROOT
            echo "Resetting Simulators..."
            rm -rf $automation_dir".sims/"
            ;;
        k ) # stop all running simulators
            echo "Stopping all running simulators..."
            touch $automation_dir".sims/STOP.tmp"
            exit 0
            ;;
        m ) # specify number of simulators
            num_simulators="${OPTARG}"
            ;;
        p ) # premade tuning / var file
            premade="y"
            ;;
        z ) # zip output folder
            zip_output="y"
            ;;
        \? ) # invalid option
            echo "Invalid option: -$OPTARG" 1>&2
            exit 1
            ;;
    esac
done

#####################################
### CHECK FOR CONFLICTING OPTIONS ###

if [ $tuning == "y" ] && [ $reset_params == "y" ]; then
    echo "tuning and resetting parameters options conflict"
    exit 1
fi
if [ $tuning == "y" ] && [ $var_params == "y" ]; then
    echo "tuning mode and parameter variation options conflict"
    exit 1
fi
if [ $reset_params == "y" ] && [ $var_params == "y" ]; then
    echo "resetting parameters and parameter variation options conflict"
    exit 1
fi

#####################################
### SET UP SIMULATION ENVIRONMENT ###

if [ $var_params == "y" ]; then # run create-params.py if needed"
    python3 $automation_dir".input/nb2py.py" $automation_dir".input/create-params.ipynb"
    python3 $automation_dir".input/create-params.py" $automation_dir $premade
    if [ $? -ne 0 ]; then # if create-params.py fails, exit script
        echo "Parameter Variation Cancelled"
        rm -f $automation_dir".input/create-params.py"
        exit 1
    fi
    rm -f $automation_dir".input/create-params.py" 
fi

if [ ! -f $automation_dir"parameters.csv" ]; then # test for parameters.csv
    if [ $tuning != "y" ]; then
        echo "parameters.csv not found, creating new one"
        echo "If this is your first time, run with -h flag for help"
        cp $automation_dir".input/sim/param.csv" $automation_dir"parameters.csv"
        exit 1
    fi
fi

if [ $reset_params == "y" ]; then # backup parameters.csv for debugging
    cp $automation_dir"parameters.csv" $automation_dir".input/parameters.bak"
fi

if [ ! -d $automation_dir".sims/" ]; then # create Sims directory if needed
    mkdir -p $automation_dir".sims/"
fi

rm -f $automation_dir".sims/STOP.tmp" # remove master STOP.tmp if present

# setup Output directories
mkdir -p $automation_dir"Output/"
mkdir -p $automation_dir"Analysis/Models/"

if [ $tuning == "y" ]; then
    python3 $automation_dir".input/nb2py.py" $automation_dir".input/tuning.ipynb"
fi
#####################################
######### RUN SIMULATIONS ###########

start=`date +%s`
echo "Starting simulations at `date`"

# run simulations managed by python scripts
python3 $automation_dir".input/nb2py.py" $automation_dir".input/simManager.ipynb"
python3 $automation_dir".input/simManager.py" $automation_dir $num_simulators $tuning $premade

#####################################
######### CLEAN UP SIMULATIONS ######

rm -f $automation_dir".input/simManager.py"
rm -f $automation_dir"status.csv"
rm -f $automation_dir".input/tuning.py"
cp -f $automation_dir"parameters.csv" $automation_dir"Output/parameters.csv"

if [ $reset_params == "y" ]; then # restore parameters.csv
    mv $automation_dir".input/parameters.bak" $automation_dir"parameters.csv"

    # read off sim0 log file for debugging
    if [ -f $automation_dir".sims/0/log.log" ]; then
        echo "sim0 log file:"
        cat $automation_dir".sims/0/log.log"
    fi
fi

touch $automation_dir".sims/STOP.tmp" # create master STOP.tmp

# zip simOutput folder
if [ $zip_output == "y" ]; then
    cd $automation_dir"Output/"
    zip -r output.zip *
    cd $automation_dir
fi

end=`date +%s`
runtime=$((end-start))
echo "Finished simulations at `date` "
echo "Total runtime: $runtime seconds"