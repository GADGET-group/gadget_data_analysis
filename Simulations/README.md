# GADGET2 ATTPCROOT Parameters Automation
Developed by Adam Jaros as part of the FRIB's E21072 under Dr. Chris Wrede

## Simulation Instructions and Requirements
Usage: run_sim.sh [-flags]
  -t   run tuning mode
  -v   generate parameters with variation script
  -m#  specify number of simulators (1 - 10)
  -p   premade tuning / var file
  -c   clean Output before running
  -a   force reset of Simulators
  -d   reset parameter file for debugging
  -k   kill all running simulators
  -z   zip output folder
  -h   display this help message

- Will automatically download and install [ATTPCROOTv2](https://github.com/ATTPC/ATTPCROOTv2) to run simulations
- Meant to run on FRIB's fishtank, tested to work on all servers

