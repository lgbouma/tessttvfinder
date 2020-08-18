#!/usr/bin/env bash

##########################################
# USAGE: ./measure_wasp4b_times_ast.sh &> ../logs/wasp4b.log &
# (note the extra &'s needed to pipe stdout and stderr)
##########################################

n_sector=2
lcdir='/home/luke/local/tess_alert_lightcurves/sector-'${n_sector}'/'
chain_savdir='/home/luke/local/emcee_chains/'
n_mcmc=2000 # 2000, 2 mins per individual transit
n_phase_mcmc=1000 #  at 1000, 15 minutes per phase transit. (30 total), 
n_workers=16 # number of workers on brik!
n_transit_durations=2

ticid=100100827 # WASP-18b
python -u measure_transit_times_from_lightcurve.py \
  --ticid $ticid --sectornum $n_sector --n_mcmc_steps $n_mcmc \
  --n_phase_mcmc_steps $n_phase_mcmc \
  --no-getspocparams --read_literature_params \
  --overwritesamples --no-mcmcprogressbar \
  --nworkers $n_workers --chain_savdir $chain_savdir --lcdir $lcdir \
  --no-verify-times \
  --n_transit_durations $n_transit_durations
