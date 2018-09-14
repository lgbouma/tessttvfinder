#!/usr/bin/env bash

##########################################
# USAGE: ./measure_WASP-46b_times.sh &> ../logs/wasp_46b.log &
# (note the extra &'s needed to pipe stdout and stderr)
##########################################

# ticid = 231663901 # WASP-46b's TICID
# rest read off manually from DV reports

python measure_transit_times_from_lightcurve.py \
  --ticid 231663901 --n_mcmc_steps 20000 \
  --spoc_rp 0.14 --spoc_sma 6.17 --spoc_t0 1326.0089 --spoc_b 0.68 \
  --no-overwriteexistingsamples --mcmcprogressbar \
  --nworkers 16
