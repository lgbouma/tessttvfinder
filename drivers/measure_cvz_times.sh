#!/usr/bin/env bash

##########################################
# USAGE: ./measure_times.sh &> ../logs/measure_times.log &
##########################################

cd ../src

chain_savdir='/home/luke/local/emcee_chains/'
n_mcmc=1000
n_phase_mcmc=1000 #  at 1000, 15 minutes per phase transit. (30 total), 
n_workers=16 # number of workers on ast!
n_transit_durations=2

# WASP-100 = 38846515
# WASP-62 = 149603524

ticids=( 38846515 149603524 )

for ticid in "${ticids[@]}"
do
  lcdir='/home/luke/local/tess_mast_lightcurves/tic_'${ticid}'/'
  python -u measure_transit_times_from_lightcurve.py \
    --ticid $ticid --n_mcmc_steps $n_mcmc \
    --n_phase_mcmc_steps $n_phase_mcmc \
    --no-getspocparams --read_literature_params \
    --overwritesamples --no-mcmcprogressbar \
    --nworkers $n_workers --chain_savdir $chain_savdir --lcdir $lcdir \
    --no-verify-times \
    --n_transit_durations $n_transit_durations
done
