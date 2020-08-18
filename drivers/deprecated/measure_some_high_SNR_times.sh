#!/usr/bin/env bash

##########################################
# USAGE: ./measure_some_high_SNR_times.sh &> ../logs/measure_times_sector_1.log &
# (note the extra &'s needed to pipe stdout and stderr)
##########################################

lcdir='/home/luke/local/tess_alert_lightcurves/'
chain_savdir='/home/luke/local/emcee_chains/'
n_mcmc=100
n_workers=10

# WASP-46
python measure_transit_times_from_lightcurve.py \
  --ticid 231663901 --n_mcmc_steps $n_mcmc \
  --getspocparams \
  --overwritesamples --mcmcprogressbar \
  --nworkers $n_workers --chain_savdir $chain_savdir --lcdir $lcdir &

# # WASP-62
# python measure_transit_times_from_lightcurve.py \
#   --ticid 149603524 --n_mcmc_steps $n_mcmc \
#   --getspocparams \
#   --no-overwritesamples --mcmcprogressbar \
#   --nworkers $n_workers --chain_savdir $chain_savdir --lcdir $lcdir &
# 
# # WASP-94A
# python measure_transit_times_from_lightcurve.py \
#   --ticid 92352620 --n_mcmc_steps $n_mcmc \
#   --getspocparams \
#   --no-overwritesamples --mcmcprogressbar \
#   --nworkers $n_workers --chain_savdir $chain_savdir --lcdir $lcdir &

# # struggles to find TIC xmatch. why?
# # WASP 95
# python measure_transit_times_from_lightcurve.py \
#   --ticid 144065872 --n_mcmc_steps $n_mcmc \
#   --getspocparams \
#   --no-overwritesamples --mcmcprogressbar \
#   --nworkers $n_workers --chain_savdir $chain_savdir --lcdir $lcdir &
