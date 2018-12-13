# -*- coding: utf-8 -*-
"""
how does the observed noise compare with a prediction?
"""
from __future__ import division, print_function

import os, argparse, pickle, h5py
from glob import glob

import numpy as np, matplotlib.pyplot as plt, pandas as pd
from astropy.io import fits
from astropy import units as u, constants as const

from astrobase.lcmath import time_bin_magseries

from numpy import array as nparr
from statsmodels import robust

def check_noise_performance(
    ticid, pickledir='../results/tess_lightcurve_fit_parameters/',
    sampledir='/home/luke/local/emcee_chains/',
    fittype='mandelagol_and_line',
    ndim=4):

    pickledir += str(ticid)

    # empirical errors -> believable error bars!
    fpattern = (
        '{:s}_{:s}_fit_empiricalerrs_t???.pickle'.
        format(str(ticid), fittype)
    )
    fnames = np.sort(glob(os.path.join(pickledir,fpattern)))

    samplepattern = (
        '{:s}_{:s}_fit_samples_{:d}d_t???_empiricalerrs.h5'.
        format(str(ticid), fittype, ndim)
    )
    samplenames = np.sort(glob(sampledir+samplepattern))

    transit_ix = 0
    residnoise = []
    for fname, samplename in zip(fnames, samplenames):

        transit_ix += 1

        try:

            d = pickle.load(open(fname, 'rb'))

            ms = d['magseries']
            fi = d['fitinfo']

            resid = fi['fitmags'] - ms['mags']

            times = ms['times']

            bin_resid = time_bin_magseries(times, resid, binsize=3600.,
                                           minbinelems=10)

            print('{:d}/{:d}: 1hr residual RMS: {:.6f} = {:.3f} ppm hr^(1/2)'.
                  format(transit_ix, len(fnames),
                         np.std(bin_resid['binnedmags']),
                         np.std(bin_resid['binnedmags'])*1e6)
                 )

            residnoise.append(np.std(bin_resid['binnedmags'])*1e6)


        except Exception as e:
            print(e)
            print('transit {:d} failed, continue'.format(transit_ix))
            continue

    residnoise = np.array(residnoise)

    print('mean RMS: {:.3f} ppm hr^(1/2)'.format(np.mean(residnoise)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=('Given a lightcurve with transits (e.g., alerted '
                     'from TESS Science Office), measure the times that they '
                     'fall at by fitting models.'))

    parser.add_argument('--ticid', type=int, default=None,
        help=('integer TIC ID for object. Pickle paramfile assumed to be at '
              '../results/tess_lightcurve_fit_parameters/'
             ))

    args = parser.parse_args()

    if not args.ticid:
        raise AssertionError('gotta give a ticid')

    check_noise_performance(args.ticid)
