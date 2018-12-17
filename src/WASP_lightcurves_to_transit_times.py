# -*- coding: utf-8 -*-
'''
Tools for working with WASP lightcurves.
'''
from __future__ import division, print_function

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import numpy as np, pandas as pd
from numpy import array as nparr

from astropy.io import ascii
from astropy.io import fits
from astropy import units as u, constants as const
from astropy.table import Table
from astropy.coordinates import SkyCoord

from astrobase.periodbase import kbls
from astrobase.plotbase import plot_phased_mag_series
from astrobase import periodbase, checkplot
from astrobase.lcmath import phase_magseries, sigclip_magseries
from astrobase.varbase import lcfit
from astrobase.periodbase import get_snr_of_dip
from astrobase.varbase.transits import estimate_achievable_tmid_precision

from glob import glob
from parse import parse, search
import os, pickle


def read_WASP_lightcurve(lcfile):
    """
    querying the nasaexoplanet archive gives WASP lightcurves in a format. this
    function reads them.
    """

def read_WASP_lightcurve(lcfile):
    '''
    querying the WASP All Star Catalog at
        http://www.astrouw.edu.pl/wasp/?page=aasc
    gives lightcurves in a particular format. This is the function that reads
    them.
    '''

    with open(lcfile, 'r') as f:
        lines = [l.rstrip('\n').strip() for l in f.readlines() if
                 (not l.startswith('\\')) and (not l.startswith('|'))
                ]

    hjd = [l.split()[0] for l in lines]
    relmag = [l.split()[1] for l in lines]
    err = [l.split()[2] for l in lines]
    accepted = [l.split()[3] for l in lines]

    df = pd.DataFrame({'HJD': hjd, 'RELMAG': relmag, 'ERR':err,
                       'ACCEPTED':accepted})

    return df

def HJD_UTC_to_BJD_TDB_astropy(hjd_arr, ra, dec):
    # assume you get decimal ra, decimal dec.

    raise NotImplementedError

    from astropy import time, coordinates, units as u

    c = coordinates.SkyCoord(ra*u.deg, dec*u.deg, frame='icrs')

    location = coordinates.EarthLocation.from_geocentric(0*u.m,0*u.m,0*u.m)

    hjdtimes = time.Time(hjd_arr, format='jd', scale='utc', location=location)

    ltt_helio = times.light_travel_time(c, 'heliocentric')

    jd_utc_times = hjdtimes

    import IPython; IPython.embed()

    ltt_bary = times.light_travel_time(c)

    time_barycentre = times.tdb + ltt_bary



def HJD_UTC_to_BJD_TDB_eastman(hjd_arr, ra, dec):
    '''
    Given HJD_UTC, compute BJD_TDB by querying the Eastman et al (2010) API.

    CITE: Eastman et al. (2010)

    example query:
    'http://astroutils.astronomy.ohio-state.edu/time/convert.php?'+
    'JDS=2450000.123,2450000,2451234.123&RA=219.90206&DEC=-60.837528&'+
    'FUNCTION=hjd2bjd'

    As written, I hackily split the query into twos. NOTE: will need to
    generalize some day, maybe.  max querystr length is something like 8000
    characters, which is why the split is needed.
    '''
    import urllib.request

    hjd_longstr = ','.join(list(map(str,hjd_arr)))

    n_times = len(hjd_arr)
    n_characters_per_time = int(np.mean(list(map(len,hjd_longstr.split(',')))))

    import IPython; IPython.embed()
    if 2 < n_characters_per_time*n_times < 25000:
        hjd_strs = [ ','.join(list(map(str,hjd_arr[0:int(n_times/4) ]))),
                     ','.join(list(map(str,hjd_arr[int(n_times/4):int(2*n_times/4)]))),
                     ','.join(list(map(str,hjd_arr[int(2*n_times/4):int(3*n_times/4)]))),
                     ','.join(list(map(str,hjd_arr[int(3*n_times/4):]))),
                   ]
        #hjd_strs = [ ','.join(list(map(str,hjd_arr[0:int(n_times/2) ]))),
        #             ','.join(list(map(str,hjd_arr[int(n_times/2):])))
        #           ]
    else:
        raise NotImplementedError('need to generalize code for N queries')

    querystrs=[
        'http://astroutils.astronomy.ohio-state.edu/time/convert.php?'+
        'JDS={:s}'.format(hjd_str)+
        '&RA={:f}&DEC={:f}&'.format(ra, dec)+
        'FUNCTION=hjd2bjd'
        for hjd_str in hjd_strs
    ]

    reqs = [urllib.request.Request(querystr) for querystr in querystrs]
    responses = [urllib.request.urlopen(req) for req in reqs]
    bjds = [response.read().decode('ascii').rstrip('\n') for response in
            responses]

    bjds = [bjd.split('\n') for bjd in bjds]
    bjds_flattened = [v for sublist in bjds for v in sublist]

    bjds = nparr(list(map(float, bjds_flattened)))

    return bjds


def wrangle_WASP_lightcurve(df, ra, dec, plname):
    '''
    do the following janitorial tasks:

    * convert values to floats

    * convert HJDs to BJDs (currently, manually with big nubmer of times)

    return df with the columns:
        Index(['BJD_TDB', ...])
    '''

    # convert HJDs to BJDs with Eastman's applet.. (MANUALLY is current
    # implementation)
    hjds = nparr(df['HJD'])

    # FIXME wrapping this time conversion / asking joel might be smarter. what
    # if you did it in astropy?
    bjd_tdb = np.genfromtxt(
        '../data/WASP_lightcurves/{:s}_BJD_TDB_eastman.txt'.format(plname))

    df['BJD_TDB'] = bjd_tdb

    df['ERR'] = pd.to_numeric(df['ERR'])
    df['RELMAG'] = pd.to_numeric(df['RELMAG'])
    df['HJD'] = pd.to_numeric(df['HJD'])
    df['BJD_TDB'] = pd.to_numeric(df['BJD_TDB'])

    return df


def plot_old_lcs(times, mags, stimes, smags, phasedict, period, epoch, sfluxs,
                 plname, savdir='../results/WASP_lightcurves/',
                 telescope='WASP'):

    f,ax=plt.subplots(figsize=(12,6))
    ax.scatter(times, mags)
    ax.set_xlabel('BJD TDB')
    ax.set_ylabel('{:s} mag (only ap)'.format(telescope))
    ax.set_ylim([max(ax.get_ylim()), min(ax.get_ylim())])
    f.savefig(savdir+'{:s}_onlyap.png'.format(plname), dpi=400)
    print('made {:s}'.format(savdir+'{:s}_onlyap.png'.format(plname)))
    plt.close('all')

    f,ax=plt.subplots(figsize=(12,6))
    ax.scatter(stimes, smags)
    ax.set_xlabel('BJD TDB')
    ax.set_ylabel('sigclipped {:s} mag (only ap)'.format(telescope))
    ax.set_ylim([max(ax.get_ylim()), min(ax.get_ylim())])
    f.savefig(savdir+'{:s}_sigclipped_onlyap.png'.format(plname), dpi=400)
    print('made {:s}'.format(savdir+'{:s}_sigclipped_onlyap.png'.format(plname)))
    plt.close('all')

    f,ax=plt.subplots(figsize=(12,6))
    ax.scatter(phasedict['phase'], phasedict['mags'])
    ax.set_xlabel('phase')
    ax.set_ylabel('sigclipped {:s} mag (only ap)'.format(telescope))
    ax.set_ylim([max(ax.get_ylim()), min(ax.get_ylim())])
    ax.set_xlim([-.6,.6])
    f.savefig(savdir+'{:s}_phased_on_discovery_params.png'.format(plname),
              dpi=400)
    print('made {:s}'.format(savdir+'{:s}_phased_on_discovery_params.png'.format(plname)))
    plt.close('all')

    n_obs = [148,296,592,1184]
    phasebin = [0.08,0.04,0.02,0.01]
    from scipy.interpolate import interp1d
    pfn = (
        interp1d(n_obs, phasebin, bounds_error=False, fill_value='extrapolate')
    )

    pb = float(pfn(len(stimes)))

    outfile = (
        savdir+'{:s}_phased_on_discovery_params_binned.png'.format(plname)
    )
    plot_phased_mag_series(stimes, smags, period, magsarefluxes=False,
                           errs=None, normto=False, epoch=epoch,
                           outfile=outfile, sigclip=False, phasebin=pb,
                           plotphaselim=[-.6,.6], plotdpi=400)
    print('made {:s}'.format(outfile))

    outfile = (
        savdir+'{:s}_fluxs_phased_on_discovery_params_binned.png'.format(plname)
    )
    plot_phased_mag_series(stimes, sfluxs, period, magsarefluxes=True,
                           errs=None, normto=False, epoch=epoch,
                           outfile=outfile, sigclip=False, phasebin=pb,
                           plotphaselim=[-.6,.6], plotdpi=400)
    print('made {:s}'.format(outfile))

def run_wasp_periodograms(times, mags, errs,
                          outdir='../results/WASP_lightcurves/',
                          outname='WASP-18b_BLS_GLS.png'):

    blsdict = kbls.bls_parallel_pfind(times, mags, errs,
                                      magsarefluxes=False, startp=0.5,
                                      endp=1.5, maxtransitduration=0.3,
                                      nworkers=8, sigclip=[15,3])
    gls = periodbase.pgen_lsp(times, mags, errs, magsarefluxes=False,
                              startp=0.5, endp=1.5, nworkers=8,
                              sigclip=[15,3])
    outpath = outdir+outname
    cpf = checkplot.twolsp_checkplot_png(blsdict, gls, times, mags, errs,
                                         outfile=outpath, objectinfo=None)


def fit_lightcurve_get_transit_time(stimes, sfluxs, serrs, savstr,
                                    plname, period, epoch,
                                    n_mcmc_steps=100,
                                    overwriteexistingsamples=False):
    '''
    fit for the epoch, fix all other transit parameters.

    args:
        stimes, sluxs, serrs (np.ndarray): sigma-clipped times, fluxes, and
        errors.  (Errors can be either empirical, or from WASP).

        savstr (str): used as identifier in chains, plots, etc.

        plname (str): used to prepend in chains, plots, etc.

        for example,
            mandelagolfit_plotname = (
                str(plname)+'_mandelagol_fit_{:s}_fixperiod.png'.format(savstr)
            )

        period, epoch (float, units of days): used to fix the period, and get
        initial epoch guess.
    '''

    fit_savdir = '../results/WASP_lightcurves/'
    chain_savdir = '/home/luke/local/emcee_chains/'
    savdir='../results/WASP_lightcurves/'

    # numbers for initial guess, from Winn+ 2009.
    true_b, true_sma, true_t0, true_rp = (
        0.143, 5.473, epoch, 0.15375 )

    # ClarHa03 Claret & Hauschildt (2003A+A...412..241C), V band, via JKTLD
    # note this WASP data is ~V band (Pollacco et al 2006), so this should be
    # fine, unless the transit is seriously chromatic.
    # $ jktld 5500 4.4813 0 2 q 5 VJ
    u_linear, u_quad = 0.5882, 0.1448

    true_incl = None
    if isinstance(true_b,float) and isinstance(true_sma, float):
        # b = a/Rstar * cosi
        cosi = true_b / true_sma
        true_incl = np.degrees(np.arccos(cosi))

    ## # approach #1: use the discovery epoch as initial guess, and narrow priors
    ## # around it
    ## initfitparams = {'t0':epoch, 'sma':true_sma}
    ## priorbounds = {'t0':(epoch-0.03, epoch+0.03),
    ##                'sma':(0.7*true_sma,1.3*true_sma)}
    ## discoveryparams = {'t0':true_t0, 'sma':true_sma}

    ## # approach #2: take epoch closest mean time as initial guess (weights
    ## # appropriately for the timeseries), and enforce in prior that the time is
    ## # within +/- period/2 of that. fit for duration and t0.
    ## init_ix = int(np.floor( (np.nanmean(stimes) - epoch)/period ))
    ## init_t0 = true_t0 + init_ix*period

    ## initfitparams = {'t0': init_t0, 'sma':true_sma}
    ## priorbounds = {'t0':(init_t0-0.03, init_t0+0.03),
    ##                'sma':(0.7*true_sma,1.3*true_sma) }
    ## discoveryparams = {'t0':init_t0, 'sma':true_sma}

    ## fixedparams = {'ecc':0., 'omega':90., 'limb_dark':'quadratic',
    ##                'period':period, 'incl': 86.0,
    ##                'u':[u_linear,u_quad], 'rp':true_rp}

    # approach #3: take epoch closest mean time as initial guess (weights
    # appropriately for the timeseries), and enforce in prior that the time is
    # within +/- (small window) of that. fit for only t0.
    init_ix = int(np.floor( (np.nanmean(stimes) - epoch)/period ))
    init_t0 = true_t0 + init_ix*period

    initfitparams = {'t0': init_t0}
    priorbounds = {'t0':(init_t0-0.03, init_t0+0.03),
                  }
    discoveryparams = {'t0':init_t0}

    fixedparams = {'ecc':0., 'omega':90., 'limb_dark':'quadratic',
                   'period':period, 'incl': true_incl,
                   'u':[u_linear,u_quad], 'rp':true_rp, 'sma':true_sma}

    mandelagolfit_plotname = (
        str(plname)+'_mandelagol_fit_{:s}_fixperiod.png'.format(savstr)
    )
    corner_plotname = (
        str(plname)+'_corner_mandelagol_fit_{:s}_fixperiod.png'.format(savstr)
    )
    sample_plotname = (
        str(plname)+'_mandelagol_fit_samples_{:s}_fixperiod.h5'.format(savstr)
    )

    mandelagolfit_savfile = fit_savdir + mandelagolfit_plotname
    corner_savfile = fit_savdir + corner_plotname
    if not os.path.exists(chain_savdir):
        try:
            os.mkdir(chain_savdir)
        except:
            raise AssertionError('you need to save chains')
    samplesavpath = chain_savdir + sample_plotname

    print('beginning {:s}'.format(samplesavpath))

    plt.close('all')

    mandelagolfit = lcfit.mandelagol_fit_magseries(
                    stimes, sfluxs, serrs,
                    initfitparams, priorbounds, fixedparams,
                    trueparams=discoveryparams, magsarefluxes=True,
                    sigclip=None, plotfit=mandelagolfit_savfile,
                    plotcorner=corner_savfile,
                    samplesavpath=samplesavpath, nworkers=16,
                    n_mcmc_steps=n_mcmc_steps, eps=1e-2, n_walkers=500,
                    skipsampling=False,
                    overwriteexistingsamples=overwriteexistingsamples,
                    mcmcprogressbar=True)

    maf_savpath = (
        "../data/WASP_pickles/"+str(plname)+
        "_mandelagol_fit_{:s}_fixperiod.pickle".format(savstr)
    )
    with open(maf_savpath, 'wb') as f:
        pickle.dump(mandelagolfit, f, pickle.HIGHEST_PROTOCOL)
        print('PKLSAV saved {:s}'.format(maf_savpath))

    fitfluxs = mandelagolfit['fitinfo']['fitmags']
    initfluxs = mandelagolfit['fitinfo']['initialmags']

    outfile = (
        savdir+
        '{:s}_phased_initialguess_{:s}_fit.png'.format(plname, savstr)
    )
    plot_phased_mag_series(stimes, sfluxs, period, magsarefluxes=True,
                           errs=None, normto=False, epoch=epoch,
                           outfile=outfile, sigclip=False, phasebin=0.02,
                           plotphaselim=[-.6,.6], plotdpi=400,
                           modelmags=initfluxs, modeltimes=stimes)

    fitepoch = mandelagolfit['fitinfo']['fitepoch']
    fiterrors = mandelagolfit['fitinfo']['finalparamerrs']
    fitepoch_perr = fiterrors['std_perrs']['t0']
    fitepoch_merr = fiterrors['std_merrs']['t0']

    outfile = (
        savdir+
        '{:s}_phased_{:s}_fitfluxs.png'.format(plname, savstr)
    )
    plot_phased_mag_series(stimes, sfluxs, period, magsarefluxes=True,
                           errs=None, normto=False, epoch=fitepoch,
                           outfile=outfile, sigclip=False, phasebin=0.02,
                           plotphaselim=[-.6,.6], plotdpi=400,
                           modelmags=fitfluxs, modeltimes=stimes)

    print('fitepoch : {:.8f}'.format(fitepoch))
    print('fitepoch_perr: {:.8f}'.format(fitepoch_perr))
    print('fitepoch_merr: {:.8f}'.format(fitepoch_merr))

    snr, _, empirical_noise = get_snr_of_dip(
        stimes, sfluxs, stimes, fitfluxs, magsarefluxes=True)

    sigma_tc_theory = estimate_achievable_tmid_precision(snr)

    print(
        'mean fitepoch err: {:.2f}'.format(
        np.mean([fitepoch_merr, fitepoch_perr]))
    )

    print('mean fitepoch err / theory err = {:.2f}'.format(
        np.mean([fitepoch_merr, fitepoch_perr]) / sigma_tc_theory
    ))

    print('mean data error from WASP = {:.2e}'.format(np.mean(serrs))+
          '\nempirical RMS = {:.2e}'.format(empirical_noise)
    )

    return empirical_noise


def reduce_WASP_4b(plname='WASP-4b_dataset0'):

    # options when running
    try_to_recover_periodograms = False
    make_lc_plots = True

    # table 1 of Hellier et al 2009 discovery paper
    period, epoch = 1.3382282, 2454365.91464
    # decimal ra, dec of target used only for BJD conversion
    ra, dec = 353.56283333, -42.06141667

    # file parsing
    lcdir = '../data/WASP_lightcurves/'
    wasp_lcs = [f for f in glob(lcdir+'*.tbl') if plname in f]
    if not len(wasp_lcs)==1:
        raise AssertionError
    lcfile = wasp_lcs[0]

    fit_savdir = '../results/WASP_lightcurves/'
    chain_savdir = '/home/luke/local/emcee_chains/'
    savdir='../results/WASP_lightcurves/'
    for sdir in [fit_savdir, chain_savdir, savdir]:
        if not os.path.exists(sdir):
            os.mkdir(sdir)

    #########
    # begin #
    #########
    tempdf = read_WASP_lightcurve(lcfile)
    df = wrangle_WASP_lightcurve(tempdf, ra, dec, plname)

    times, mags, errs = (nparr(df['BJD_TDB']), nparr(df['RELMAG']),
                         nparr(df['ERR']))

    stimes, smags, serrs = sigclip_magseries(times, mags, errs, sigclip=[5,5],
                                             magsarefluxes=False)

    phzd = phase_magseries(stimes, smags, period, epoch, wrap=True, sort=True)

    # convert from mags to relative fluxes for fitting
    # m_x - m_x0 = -5/2 log10( f_x / f_x0 )
    # so
    # f_x = f_x0 * 10 ** ( -2/5 (m_x - m_x0) )
    m_x0, f_x0 = 10, 1e3 # arbitrary
    sfluxs = f_x0 * 10**( -0.4 * (smags - m_x0) )
    sfluxs /= np.nanmedian(sfluxs)

    if try_to_recover_periodograms:
        run_wasp_periodograms(stimes, smags, serrs)

    if make_lc_plots:
        plot_old_lcs(times, mags, stimes, smags, phzd, period, epoch, sfluxs,
                     plname, savdir=savdir, telescope='WASP')

    ####################################################################
    # fit the lightcurve, show the phased result, get the transit time #
    ####################################################################

    n_mcmc_steps = 500
    overwrite=1

    savstr = 'wasp_errs_1d'
    # use physical parameters from Wilson+ 2008 as fixed parameters
    empirical_errs = fit_lightcurve_get_transit_time(stimes, sfluxs, serrs,
                                                     savstr, plname, period, epoch,
                                                     n_mcmc_steps=n_mcmc_steps,
                                                     overwriteexistingsamples=overwrite)


    # the WASP errors are good for fitting an initial model to the data, but
    # they may be over/under-estimates. instead use the "empirical errors",
    # which are the measured 1-sigma standard deviations of the residual.

    savstr = 'wasp_empirical_errs_1d'
    eerrs = np.ones_like(serrs)*empirical_errs

    _ = fit_lightcurve_get_transit_time(stimes, sfluxs, eerrs,
                                        savstr, plname, period, epoch,
                                        n_mcmc_steps=n_mcmc_steps,
                                        overwriteexistingsamples=overwrite)


if __name__ == "__main__":

    only_WASP_4b = True

    if only_WASP_4b:
        reduce_WASP_4b(plname='WASP-4b_dataset0')
        reduce_WASP_4b(plname='WASP-4b_dataset1')
