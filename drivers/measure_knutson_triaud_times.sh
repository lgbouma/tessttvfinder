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

# WASP-53 = 268766053 : worked!
# WASP-22 = 257567854 : worked!
# HAT-P-7 = Kepler-2 = 424865156 : worked!
# Kepler-13 = 1717079071 : worked!

# HAT-P-11 = 28230919 : error! getting zero points in transit

# WASP-81 = 399402994 : no SC data
# HAT-P-2 = 39903405 : no SC data
# HAT-P-4 = 138294130 : no SC data
# HAT-P-10 = 85593751 : no SC data
# HAT-P-13 = 20096620 : no SC data
# HAT-P-22 = 252479260 : no SC data
# HAT-P-29 = 250707118 : no SC data
# HAT-P-32 = 292152376 : no SC data
# WASP-10 = 431701493 : no SC data
# XO-2 = 356473034 : no SC data
# Kepler-1 = TrES-2 = 399860444 : no SC data. wtf why?

# ticids=( 268766053 399402994 39903405 138294130 424865156 85593751 28230919 20096620 252479260 250707118 292152376 431701493 257567854 356473034 1717079071 )
ticids=( 399860444 ) #28230919 )

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
