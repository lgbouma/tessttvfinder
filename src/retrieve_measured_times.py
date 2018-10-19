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

def retrieve_measured_times_mcmc(ticid):
    # NOTE: outdated. phase out soon!

    sampledir = '/Users/luke/local/emcee_chains/'
    fpattern = '{:s}_mandelagol_fit_samples_6d_t???.h5'.format(str(ticid))
    fnames = glob(sampledir+fpattern)

    ##########################################
    # detrending parameters. mingap: minimum gap to determine time group size.
    # smooth_window_day: window for median filtering.
    mingap = 240./60./24.
    smooth_window_day = 2.
    cadence_min = 2
    make_diagnostic_plots = True

    cadence_day = cadence_min / 60. / 24.
    windowsize = int(smooth_window_day/cadence_day)
    if windowsize % 2 == 0:
        windowsize += 1

    # paths for reading and writing plots
    lcdir = '../data/tess_lightcurves/'
    lcname = 'tess2018206045859-s0001-{:s}-111-s_llc.fits.gz'.format(
                str(ticid).zfill(16))
    lcfile = lcdir + lcname
    ##########################################

    time, flux, err_flux = at.get_time_flux_errs_from_Ames_lightcurve(
                                lcfile, 'PDCSAP')

    # get time groups, and median filter each one
    ngroups, groups = lcmath.find_lc_timegroups(time, mingap=mingap)

    tg_smooth_flux = []
    for group in groups:
        tg_flux = flux[group]
        tg_smooth_flux.append(
            smooth_magseries_ndimage_medfilt(tg_flux, windowsize)
        )

    smooth_flux = np.concatenate(tg_smooth_flux)
    whitened_flux = flux/smooth_flux

    # run bls to get initial parameters.
    endp = 1.05*(np.nanmax(time) - np.nanmin(time))/2
    blsdict = kbls.bls_parallel_pfind(time, flux, err_flux, magsarefluxes=True,
                                      startp=0.1, endp=endp,
                                      maxtransitduration=0.3, nworkers=8,
                                      sigclip=10.)
    fitd = kbls.bls_stats_singleperiod(time, flux, err_flux,
                                       blsdict['bestperiod'],
                                       magsarefluxes=True, sigclip=10.,
                                       perioddeltapercent=5)

    # fit Mandel & Agol model to each single transit, +/- 5 transit durations
    tmids, t_starts, t_ends = get_transit_times(fitd, time, 5)

    t0_list, t0_merrs, t0_perrs, t0_bigerrs, samplepaths = [],[],[],[],[]

    for transit_ix, t_start, t_end in list(
        zip(range(len(t_starts)), t_starts, t_ends)
    ):

        try:
            sel = (time < t_end) & (time > t_start)
            sel_time = time[sel]
            sel_whitened_flux = whitened_flux[sel]
            sel_err_flux = err_flux[sel]

            u_linear, u_quad = get_limb_darkening_initial_guesses(lcfile)
            a_guess = get_a_over_Rstar_guess(lcfile, fitd['period'])

            rp = np.sqrt(fitd['transitdepth'])

            initfitparams = {'t0':t_start + (t_end-t_start)/2., 'rp':rp,
                             'sma':a_guess, 'incl':85, 'u':[u_linear,u_quad] }

            fixedparams = {'ecc':0., 'omega':90., 'limb_dark':'quadratic',
                           'period':fitd['period'] }

            priorbounds = {'rp':(rp-0.01, rp+0.01),
                           'u_linear':(u_linear-1, u_linear+1),
                           'u_quad':(u_quad-1, u_quad+1),
                           't0':(np.min(sel_time), np.max(sel_time)),
                           'sma':(0.7*a_guess,1.3*a_guess), 'incl':(75,90) }

            t_num = str(transit_ix).zfill(3)
            sample_plotname = (
                str(ticid)+'_mandelagol_fit_samples_6d_t{:s}.h5'.format(t_num)
            )

            chain_savdir = '/Users/luke/local/emcee_chains/'
            samplesavpath = chain_savdir + sample_plotname

            print('beginning {:s}'.format(samplesavpath))

            mandelagolfit = lcfit.mandelagol_fit_magseries(
                            sel_time, sel_whitened_flux, sel_err_flux,
                            initfitparams, priorbounds, fixedparams,
                            trueparams=None, magsarefluxes=True,
                            sigclip=10., plotfit=None,
                            plotcorner=None,
                            samplesavpath=samplesavpath, nworkers=8,
                            n_mcmc_steps=42, eps=1e-1, n_walkers=500,
                            skipsampling=True, overwriteexistingsamples=False)

            fitparams = mandelagolfit['fitinfo']['finalparams']
            fiterrs = mandelagolfit['fitinfo']['finalparamerrs']

            t0_list.append(fitparams['t0'])
            t0_merrs.append(fiterrs['std_merrs']['t0'])
            t0_perrs.append(fiterrs['std_perrs']['t0'])
            t0_bigerrs.append(max(
                fiterrs['std_merrs']['t0'],fiterrs['std_perrs']['t0']))
            samplepaths.append(samplesavpath)

        except Exception as e:
            print(e)
            print('transit {:d} failed, continue'.format(transit_ix))
            continue

    t0, t0_merr, t0_perr, t0_bigerr = (
        nparr(t0_list),nparr(t0_merrs),nparr(t0_perrs),nparr(t0_bigerrs)
    )

    df = pd.DataFrame({
        't0_BTJD':t0, 't0_merr':t0_merr, 't0_perr':t0_perr,
        't0_bigerr':t0_bigerr, 'BJD_TDB':t0+2457000, 'samplepath':samplepaths
    })
    outdir = '../data/'
    outname = (
        str(ticid)+'_measured_TESS_times_{:d}_transits.csv'.format(transit_ix)
    )
    df.to_csv(outdir+outname, index=False)
    print('saved to {:s}'.format(outdir+outname))


def retrieve_measured_times_pickle(
    ticid, pickledir='../results/tess_lightcurve_fit_parameters/',
    sampledir='/home/luke/local/emcee_chains/'
):

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

    retrieve_measured_times_pickle(args.ticid)
