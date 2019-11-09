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

# KELT-9 = 16740101  # worked, after s14 tuning for epoch
# KELT-16 = 236445129 # worked, after tuning for epoch
# HATS-70 = 98545929 # fails b/c only FFI data available. sector 7 though! (have LC)
# Kepler-91 = 352011875 # fails in MCMC, b/c depth 400ppm (R*=6Rsun). T=12.5 -> s=800 ppm hr^{1/2}. period = 6.2 days
# OGLE-TR-56 = 1425441246 # fails b/c only FFI data available.

ticids=( 236445129 98545929 352011875 1425441246 16740101 )

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
