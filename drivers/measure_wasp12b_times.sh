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

#  # 9385460 # HATS-31 -- not observed!
# 38846515 # WASP-100  -- seen in MANY SECTORS
# 66818296 # WASP-017
# 149603524 # WASP-062
# 398943781 # WASP-041
# 231670397 # WASP-073
# 466840711 # OGLE2-TR-L9
# 130150682 # OGLE-TR-10
# 390001646 # OGLE-TR-211

ticids=( 86396382 )

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
