#!/usr/bin/env bash

##########################################
# USAGE: ./measure_wasp4b_times_ast.sh &> ../logs/wasp4b.log &
# (note the extra &'s needed to pipe stdout and stderr)
##########################################

cd ../src/

n_sector=2
lcdir='/home/luke/local/tess_alert_lightcurves/sector-'${n_sector}'/'
chain_savdir='/home/luke/local/emcee_chains/'
n_mcmc=1000
n_workers=16 # number of workers on ast!

ticid=120610833 # WASP-6b
python measure_transit_times_from_lightcurve.py \
  --ticid $ticid --sectornum $n_sector --n_mcmc_steps $n_mcmc \
  --no-getspocparams --read_literature_params \
  --overwritesamples --no-mcmcprogressbar \
  --nworkers $n_workers --chain_savdir $chain_savdir --lcdir $lcdir \
  --no-verify-times

# ticid=100100827 # WASP-18b
# python measure_transit_times_from_lightcurve.py \
#   --ticid $ticid --sectornum $n_sector --n_mcmc_steps $n_mcmc \
#   --no-getspocparams --read_literature_params \
#   --no-overwritesamples --no-mcmcprogressbar \
#   --nworkers $n_workers --chain_savdir $chain_savdir --lcdir $lcdir \
#   --verify-times

# ticid=184240683 # WASP-5b
# python measure_transit_times_from_lightcurve.py \
#   --ticid $ticid --sectornum $n_sector --n_mcmc_steps $n_mcmc \
#   --no-getspocparams --read_literature_params \
#   --no-overwritesamples --no-mcmcprogressbar \
#   --nworkers $n_workers --chain_savdir $chain_savdir --lcdir $lcdir \
#   --verify-times
