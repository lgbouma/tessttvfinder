# -*- coding: utf-8 -*-
'''
usage: measure_transit_times_from_lightcurve.py [-h] [--ticid TICID]
                                                [--n_mcmc_steps N_MCMC_STEPS]
                                                [--nworkers NWORKERS]
                                                [--mcmcprogressbar]
                                                [--no-mcmcprogressbar]
                                                [--overwritesamples]
                                                [--no-overwritesamples]
                                                [--spoc_rp SPOC_RP]
                                                [--spoc_sma SPOC_SMA]
                                                [--spoc_b SPOC_B]
                                                [--spoc_t0 SPOC_T0]

Given a lightcurve with transits (e.g., alerted from TESS Science Office),
measure the times that they fall at by fitting models.

optional arguments:
  -h, --help            show this help message and exit
  --ticid TICID         integer TIC ID for object. Lightcurve assumed to be at
                        ../data/tess_lightcurves/tess2018206045859-s0001-{tici
                        d}-111-s_llc.fits.gz
  --n_mcmc_steps N_MCMC_STEPS
                        steps to run in MCMC
  --nworkers NWORKERS   how many workers?
  --mcmcprogressbar
  --no-mcmcprogressbar
  --overwritesamples
  --no-overwritesamples
  --spoc_rp SPOC_RP     spoc rp/rstar
  --spoc_sma SPOC_SMA   spoc a/rstar
  --spoc_b SPOC_B       spoc impact param
  --spoc_t0 SPOC_T0     spoc epoch
'''
from __future__ import division, print_function

import os, argparse

import matplotlib as mpl
mpl.use('Agg')
import numpy as np, matplotlib.pyplot as plt
from astropy.io import fits
from mast_utils import tic_single_object_crossmatch
from astropy import units as u, constants as const

from astrobase.varbase import lcfit
from astrobase import astrotess as at
from astrobase.periodbase import kbls
from astrobase.varbase.trends import smooth_magseries_ndimage_medfilt
from astrobase import lcmath

np.random.seed(42)

def get_a_over_Rstar_guess(lcfile, period):
    # xmatch TIC. get Mstar, and Rstar.
    # with period and Mstar, you can guess the semimajor axis.
    # then calculate a/Rstar.

    # get object RA/dec, so that you can get the Teff/logg by x-matching TIC,
    # so that you can get the theoretical limb-darkening coefficients.
    hdulist = fits.open(lcfile)
    main_hdr = hdulist[0].header
    lc_hdr = hdulist[1].header
    lc = hdulist[1].data
    ra, dec = lc_hdr['RA_OBJ'], lc_hdr['DEC_OBJ']
    sep = 0.1*u.arcsec
    obj = tic_single_object_crossmatch(ra,dec,sep.to(u.deg).value)
    if len(obj['data'])==1:
        rad = obj['data'][0]['rad']
        mass = obj['data'][0]['mass']
    else:
        raise NotImplementedError
    if not isinstance(rad ,float) and not isinstance(mass, float):
        raise NotImplementedError

    # P^2 / a^3   = 4pi^2 / (GM)
    # a^3 = P^2 * GM / (4pi^2)
    # a = ( P^2 * GM / (4pi^2) )^(1/3)

    P = period*u.day
    M = mass*u.Msun
    a = ( P**2 * const.G*M / (4*np.pi**2) )**(1/3)
    R = rad*u.Rsun
    a_by_Rstar = (a.cgs/R.cgs).value

    return a_by_Rstar


def get_limb_darkening_initial_guesses(lcfile):
    '''
    CITE: Claret 2017, whose coefficients we're parsing
    '''

    # get object RA/dec, so that you can get the Teff/logg by x-matching TIC,
    # so that you can get the theoretical limb-darkening coefficients.
    hdulist = fits.open(lcfile)
    main_hdr = hdulist[0].header
    lc_hdr = hdulist[1].header
    lc = hdulist[1].data
    ra, dec = lc_hdr['RA_OBJ'], lc_hdr['DEC_OBJ']
    sep = 0.1*u.arcsec
    obj = tic_single_object_crossmatch(ra,dec,sep.to(u.deg).value)
    if len(obj['data'])==1:
        teff = obj['data'][0]['Teff']
        logg = obj['data'][0]['logg']
        metallicity = obj['data'][0]['MH'] # often None
        if not isinstance(metallicity,float):
            metallicity = 0 # solar
    else:
        raise NotImplementedError

    # get the Claret quadratic priors for TESS bandpass
    # the selected table below is good from Teff = 1500 - 12000K, logg = 2.5 to
    # 6. We choose values computed with the "r method", see
    # http://vizier.u-strasbg.fr/viz-bin/VizieR-n?-source=METAnot&catid=36000030&notid=1&-out=text
    assert 2300 < teff < 12000
    assert 2.5 < logg < 6

    from astroquery.vizier import Vizier
    Vizier.ROW_LIMIT = -1
    catalog_list = Vizier.find_catalogs('J/A+A/600/A30')
    catalogs = Vizier.get_catalogs(catalog_list.keys())
    t = catalogs[1]
    sel = (t['Type'] == 'r')
    df = t[sel].to_pandas()

    # since we're using these as starting guesses, not even worth
    # interpolating. just use the closest match!
    # each Teff gets 8 logg values. first, find the best teff match.
    foo = df.iloc[(df['Teff']-teff).abs().argsort()[:8]]
    # then, among those best 8, get the best logg match.
    bar = foo.iloc[(foo['logg']-logg).abs().argsort()].iloc[0]

    u_linear = bar['aLSM']
    u_quad = bar['bLSM']

    return float(u_linear), float(u_quad)


def get_transit_times(fitd, time, N):
    '''
    Given a BLS period, epoch, and transit ingress/egress points, compute
    the times within ~N transit durations of each transit.  This is useful
    for fitting & inspecting individual transits.
    '''

    tmids = [fitd['epoch'] + ix*fitd['period'] for ix in range(-1000,1000)]
    sel = (tmids > np.nanmin(time)) & (tmids < np.nanmax(time))
    tmids_obsd = np.array(tmids)[sel]
    if not fitd['transegressbin'] > fitd['transingressbin']:
        raise AssertionError('careful of the width...')
    tdur = (
        fitd['period']*
        (fitd['transegressbin']-fitd['transingressbin'])/fitd['nphasebins']
    )

    t_Is = tmids_obsd - tdur/2
    t_IVs = tmids_obsd + tdur/2

    # focus on the times around transit
    t_starts = t_Is - 5*tdur
    t_ends = t_Is + 5*tdur

    return tmids, t_starts, t_ends

def single_whitening_plot(time, flux, smooth_flux, whitened_flux, ticid):
    f, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8,6))
    axs[0].scatter(time, flux, c='k', alpha=0.5, label='PDCSAP', zorder=1,
                   s=1.5, rasterized=True, linewidths=0)
    axs[0].plot(time, smooth_flux, 'b-', alpha=0.9, label='median filter',
                zorder=2)
    axs[1].scatter(time, whitened_flux, c='k', alpha=0.5,
                   label='PDCSAP/median filtered',
                   zorder=1, s=1.5, rasterized=True, linewidths=0)

    for ax in axs:
        ax.legend(loc='best')

    axs[0].set(ylabel='relative flux')
    axs[1].set(xlabel='time [days]', ylabel='relative flux')
    f.tight_layout(h_pad=0, w_pad=0)
    savdir='../results/lc_analysis/'
    savname = str(ticid)+'_whitening_PDCSAP_thru_medianfilter.png'
    f.savefig(savdir+savname, dpi=400, bbox_inches='tight')


def measure_transit_times_from_lightcurve(ticid, n_mcmc_steps, spoc_rp=None,
                                          spoc_t0=None, spoc_sma=None,
                                          spoc_b=None,
                                          overwriteexistingsamples=False,
                                          mcmcprogressbar=False,
                                          nworkers=4):

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

    fit_savdir = '../results/lc_analysis/'
    blsfit_plotname = str(ticid)+'_bls_fit.png'
    trapfit_plotname = str(ticid)+'_trapezoid_fit.png'
    mandelagolfit_plotname = str(ticid)+'_mandelagol_fit_6d.png'
    corner_plotname = str(ticid)+'_corner_mandelagol_fit_6d.png'
    sample_plotname = str(ticid)+'_mandelagol_fit_samples_6d.h5'

    blsfit_savfile = fit_savdir + blsfit_plotname
    trapfit_savfile = fit_savdir + trapfit_plotname
    mandelagolfit_savfile = fit_savdir + mandelagolfit_plotname
    corner_savfile = fit_savdir + corner_plotname
    chain_savdir = '/Users/luke/local/emcee_chains/'
    samplesavpath = chain_savdir + sample_plotname
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

    if make_diagnostic_plots:
        single_whitening_plot(time, flux, smooth_flux, whitened_flux, ticid)

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

    #  plot the BLS model.
    lcfit._make_fit_plot(fitd['phases'], fitd['phasedmags'], None,
                         fitd['blsmodel'], fitd['period'], fitd['epoch'],
                         fitd['epoch'], blsfit_savfile, magsarefluxes=True)

    ingduration_guess = fitd['transitduration']*0.2
    transitparams = [fitd['period'], fitd['epoch'], fitd['transitdepth'],
                     fitd['transitduration'], ingduration_guess
                    ]

    # fit a trapezoidal transit model; plot the resulting phased LC.
    trapfit = lcfit.traptransit_fit_magseries(time, flux, err_flux,
                                              transitparams,
                                              magsarefluxes=True,
                                              sigclip=10.,
                                              plotfit=trapfit_savfile)

    # fit Mandel & Agol model to each single transit, +/- 5 transit durations
    tmids, t_starts, t_ends = get_transit_times(fitd, time, 5)

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

            initfitparams = {'t0':t_start + (t_end-t_start)/2.,
                             'rp':rp,
                             'sma':a_guess,
                             'incl':85,
                             'u':[u_linear,u_quad] }

            fixedparams = {'ecc':0.,
                           'omega':90.,
                           'limb_dark':'quadratic',
                           'period':fitd['period'] }

            priorbounds = {'rp':(rp-0.01, rp+0.01),
                           'u_linear':(u_linear-1, u_linear+1),
                           'u_quad':(u_quad-1, u_quad+1),
                           't0':(np.min(sel_time), np.max(sel_time)),
                           'sma':(0.7*a_guess,1.3*a_guess),
                           'incl':(75,90) }

            spoc_incl = None
            if isinstance(spoc_b,float) and isinstance(spoc_sma, float):
                # b = a/Rstar * cosi
                cosi = spoc_b / spoc_sma
                spoc_incl = np.degrees(np.arccos(cosi))

            spocparams = {'rp':spoc_rp,
                          't0':spoc_t0,
                          'u_linear':u_linear,
                          'u_quad':u_quad,
                          'sma':spoc_sma,
                          'incl':spoc_incl }

            t_num = str(transit_ix).zfill(3)
            mandelagolfit_plotname = (
                str(ticid)+'_mandelagol_fit_6d_t{:s}.png'.format(t_num)
            )
            corner_plotname = (
                str(ticid)+'_corner_mandelagol_fit_6d_t{:s}.png'.format(t_num)
            )
            sample_plotname = (
                str(ticid)+'_mandelagol_fit_samples_6d_t{:s}.h5'.format(t_num)
            )

            mandelagolfit_savfile = fit_savdir + mandelagolfit_plotname
            corner_savfile = fit_savdir + corner_plotname
            chain_savdir = '/Users/luke/local/emcee_chains/'
            if not os.path.exists(chain_savdir):
                chain_savdir = '/home/luke/local/emcee_chains/'
                if not os.path.exists(chain_savdir):
                    raise AssertionError('you need to save chains')
            samplesavpath = chain_savdir + sample_plotname

            print('beginning {:s}'.format(samplesavpath))

            plt.close('all')
            mandelagolfit = lcfit.mandelagol_fit_magseries(
                            sel_time, sel_whitened_flux, sel_err_flux,
                            initfitparams, priorbounds, fixedparams,
                            trueparams=spocparams, magsarefluxes=True,
                            sigclip=10., plotfit=mandelagolfit_savfile,
                            plotcorner=corner_savfile,
                            samplesavpath=samplesavpath, nworkers=nworkers,
                            n_mcmc_steps=n_mcmc_steps, eps=1e-1, n_walkers=500,
                            skipsampling=False,
                            overwriteexistingsamples=overwriteexistingsamples,
                            mcmcprogressbar=mcmcprogressbar)
        except Exception as e:
            print(e)
            print('transit {:d} failed, continue'.format(transit_ix))
            continue


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description=('Given a lightcurve with transits (e.g., alerted '
                     'from TESS Science Office), measure the times that they '
                     'fall at by fitting models.'))

    parser.add_argument('--ticid', type=int, default=None,
        help=('integer TIC ID for object. Lightcurve assumed to be at '
              '../data/tess_lightcurves/'
              'tess2018206045859-s0001-{ticid}-111-s_llc.fits.gz'
             ))

    parser.add_argument('--n_mcmc_steps', type=int, default=None,
        help=('steps to run in MCMC'))
    parser.add_argument('--nworkers', type=int, default=4,
        help=('how many workers?'))

    parser.add_argument('--mcmcprogressbar', dest='progressbar',
        action='store_true')
    parser.add_argument('--no-mcmcprogressbar', dest='progressbar',
        action='store_false')
    parser.set_defaults(progressbar=True)

    parser.add_argument('--overwritesamples', dest='overwrite',
        action='store_true')
    parser.add_argument('--no-overwritesamples', dest='overwrite',
        action='store_false')
    parser.set_defaults(overwrite=False)

    parser.add_argument('--spoc_rp', type=float, default=None,
        help=('spoc rp/rstar'))
    parser.add_argument('--spoc_sma', type=float, default=None,
        help=('spoc a/rstar'))
    parser.add_argument('--spoc_b', type=float, default=None,
        help=('spoc impact param'))
    parser.add_argument('--spoc_t0', type=float, default=None,
        help=('spoc epoch'))

    args = parser.parse_args()

    measure_transit_times_from_lightcurve(
        args.ticid, args.n_mcmc_steps, spoc_rp=args.spoc_rp,
        spoc_t0=args.spoc_t0, spoc_sma=args.spoc_sma, spoc_b=args.spoc_b,
        overwriteexistingsamples=args.overwrite,
        mcmcprogressbar=args.progressbar,nworkers=args.nworkers)
