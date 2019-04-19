#!/usr/bin/env bash

##########################################
# USAGE: ./measure_sector_2_times_brik.sh &> ../logs/measure_times_sector_2.log &
# (note the extra &'s needed to pipe stdout and stderr)
##########################################

n_sector=2
lcdir='/home/luke/local/tess_alert_lightcurves/sector-'${n_sector}'/'
chain_savdir='/home/luke/local/emcee_chains/'
n_mcmc=1000
n_workers=16 # number of workers on brik!

ticid=402026209 # WASP-4b
python measure_transit_times_from_lightcurve.py \
  --ticid $ticid --sectornum $n_sector --n_mcmc_steps $n_mcmc \
  --getspocparams \
  --no-overwritesamples --no-mcmcprogressbar \
  --nworkers $n_workers --chain_savdir $chain_savdir --lcdir $lcdir

# ticid=184240683 # WASP-5b
# python measure_transit_times_from_lightcurve.py \
#   --ticid $ticid --sectornum $n_sector --n_mcmc_steps $n_mcmc \
#   --getspocparams \
#   --no-overwritesamples --no-mcmcprogressbar \
#   --nworkers $n_workers --chain_savdir $chain_savdir --lcdir $lcdir

# ticid=100100827 # WASP-18b
# python measure_transit_times_from_lightcurve.py \
#   --ticid $ticid --sectornum $n_sector --n_mcmc_steps $n_mcmc \
#   --getspocparams \
#   --no-overwritesamples --no-mcmcprogressbar \
#   --nworkers $n_workers --chain_savdir $chain_savdir --lcdir $lcdir
