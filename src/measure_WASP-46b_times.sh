#!/usr/bin/env bash

# ticid = 231663901 # WASP-46b's TICID
# rest read off manually from DV reports

python measure_transit_times_from_lightcurve.py --ticid 231663901 -n_mcmc_steps 100\
  --spoc_rp 0.14 --spoc_sma 6.17 --spoc_t0 1326.0089 --spoc_b 0.68
