# -*- coding: utf-8 -*-
'''
Given samples from measure_transit_times_from_lightcurve, get summary
statistics. Save them to csv.
'''
from __future__ import division, print_function
import numpy as np, pandas as pd

from astrobase.varbase import lcfit
from astrobase import astrotess as at
from astrobase.periodbase import kbls
from astrobase.varbase.trends import smooth_magseries_ndimage_medfilt
from astrobase import lcmath

from measure_transit_times_from_lightcurve import get_transit_times, \
        get_limb_darkening_initial_guesses, get_a_over_Rstar_guess

from glob import glob
from numpy import array as nparr

import os, argparse, pickle, h5py

np.random.seed(42)

def retrieve_measured_times_pickle(
    ticid, pickledir='../results/tess_lightcurve_fit_parameters/',
    sampledir='/home/luke/local/emcee_chains/'):

    # empirical errors -> believable error bars!
    fpattern = (
        '{:s}_mandelagol_fit_empiricalerrs_t???.pickle'.
        format(str(ticid))
    )
    fnames = np.sort(glob(pickledir+fpattern))

    samplepattern = (
        '{:s}_mandelagol_fit_samples_4d_t???_empiricalerrs.h5'.
        format(str(ticid))
    )
    samplenames = np.sort(glob(sampledir+samplepattern))

    t0_list, t0_merrs, t0_perrs, t0_bigerrs, samplepaths, picklepaths = (
        [],[],[],[],[],[]
    )

    transit_ix = 0
    for fname, samplename in zip(fnames, samplenames):
        transit_ix += 1

        try:
            d = pickle.load(open(fname, 'rb'))

            fitparams = d['fitinfo']['finalparams']
            fiterrs = d['fitinfo']['finalparamerrs']

            t0_list.append(fitparams['t0'])
            t0_merrs.append(fiterrs['std_merrs']['t0'])
            t0_perrs.append(fiterrs['std_perrs']['t0'])
            t0_bigerrs.append(max(
                fiterrs['std_merrs']['t0'],fiterrs['std_perrs']['t0']))
            samplepaths.append(samplename)
            picklepaths.append(fname)

        except Exception as e:
            print(e)
            print('transit {:d} failed, continue'.format(transit_ix))
            continue


    t0, t0_merr, t0_perr, t0_bigerr = (
        nparr(t0_list),nparr(t0_merrs),nparr(t0_perrs),nparr(t0_bigerrs)
    )

    df = pd.DataFrame({
        't0_BTJD':t0, 't0_merr':t0_merr, 't0_perr':t0_perr,
        't0_bigerr':t0_bigerr, 'BJD_TDB':t0+2457000, 'samplepath':samplepaths,
        'picklepath':picklepaths
    })
    outdir = '../data/'
    outname = (
        str(ticid)+'_measured_TESS_times_{:d}_transits.csv'.format(transit_ix)
    )
    df.to_csv(outdir+outname, index=False)
    print('saved to {:s}'.format(outdir+outname))


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

    retrieve_measured_times_pickle(args.ticid)
