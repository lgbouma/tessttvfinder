# -*- coding: utf-8 -*-
'''
usage: python measure_transit_times_from_lightcurve.py --help

Given a lightcurve with transits (e.g., alerted from TESS Science Office),
measure the times that they fall at by fitting models.
'''

###########
# imports #
###########
import os, argparse, pickle, h5py, json
from glob import glob
from parse import search
from copy import deepcopy

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import rc
rc('text', usetex=False)

import numpy as np, matplotlib.pyplot as plt, pandas as pd

from numpy.polynomial.legendre import Legendre

from astropy.io import fits
from astropy import units as u, constants as const
from astropy.coordinates import SkyCoord

from astrobase.varbase import lcfit
from astrobase import astrotess as at
from astrobase.periodbase import kbls
from astrobase.varbase.trends import smooth_magseries_ndimage_medfilt
from astrobase import lcmath
from astrobase.services.mast import tic_xmatch
from astrobase.services.limbdarkening import get_tess_limb_darkening_guesses
from astrobase.varbase.transits import (
    get_snr_of_dip, estimate_achievable_tmid_precision, get_transit_times
)
from astrobase.plotbase import plot_phased_magseries
from astrobase.services.mast import tic_objectsearch
from astrobase.lcfit.utils import make_fit_plot

from lightkurve.search import search_lightcurvefile

from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive

##########
# config #
##########
many_gaps_expected = [
    16740101, # KELT-16. cam1,ccd1 in sectors 14+15
    257567854 # WASP-22. sector 4 instrument anomaly.
]

GAPPY_SECTORS = [
    4,8
]

# dictionary of sectors for which each ticid has astrobase's single bls
# "refinement" getting the secondary, not primary
half_epoch_off = {
    16740101: [14],
    236445129: [15]
}

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
    sep = 1*u.arcsec
    obj = tic_xmatch(ra, dec, radius_arcsec=sep.to(u.arcsec).value)
    if len(obj['data'])==1:
        rad = obj['data'][0]['rad']
        mass = obj['data'][0]['mass']
    elif len(obj['data'])>1:
        raise NotImplementedError('more than one object')
    else:
        raise NotImplementedError('could not find TIC match within 1arcsec')
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


def get_limb_darkening_initial_guesses(lcfile, lit_logg):
    '''
    CITE: Claret 2017, whose coefficients we're parsing
    '''

    # get object RA/dec, so that you can get the Teff/logg by x-matching TIC,
    # so that you can get the theoretical limb-darkening coefficients.
    hdulist = fits.open(lcfile)
    main_hdr = hdulist[0].header
    lc_hdr = hdulist[1].header
    lc = hdulist[1].data

    teff = main_hdr['TEFF']
    logg = main_hdr['LOGG']
    metallicity = main_hdr['MH']

    if not isinstance(metallicity,float):
        metallicity = 0 # solar
    if not isinstance(logg, float):
        logg = lit_logg
        if not isinstance(logg, float):
            raise AssertionError('did not get float logg')

    u_linear, u_quad = get_tess_limb_darkening_guesses(teff, logg)

    return float(u_linear), float(u_quad)


def get_alerted_params(ticid):
    df = pd.read_csv('../data/toi-plus-2018-09-14.csv', delimiter=',')
    ap = df[df['tic_id']==ticid]
    if not len(ap) == 1:
        print('should be exactly one ticid match')
        import IPython; IPython.embed()
        raise AssertionError

    spoc_rp = float(np.sqrt(ap['Transit Depth']))
    spoc_t0 = float(ap['Epoc'])

    Tdur = float(ap['Duration'])*u.hour
    period = float(ap['Period'])*u.day

    rstar = float(ap['R_s'])*u.Rsun
    g = 10**(float(ap['logg']))*u.cm/u.s**2 # GMstar/Rstar^2
    mstar = g * rstar**2 / const.G

    a = ( period**2 * const.G*mstar / (4*np.pi**2) )**(1/3)

    spoc_sma = (a.cgs/rstar.cgs).value # a/Rstar

    T0 = period * rstar / (np.pi * a)
    b = np.sqrt(1 - (Tdur.cgs/T0.cgs)**2) # Winn 2010, eq 18

    spoc_b = b.value

    return spoc_b, spoc_sma, spoc_t0, spoc_rp


def retrieve_no_whitening(lcfile, sectornum, make_diagnostic_plots=True,
                          orbitgap=1., orbitpadding=60/(60*24),
                          expected_norbits=2, dump_interval=2.1,
                          expected_ndumps=10, dumppadding=10/(60*24)):
    '''
    Retrieve the lightcurve file and perform basic cleaning.

    args:
        lcfile (str): path to TESS lc.fits.gz lightcurve

    kwargs:
        make_diagnostic_plots (bool): True

        orbitgap (float): units of days, required to determine first and last
        minutes of each orbit.

        orbitpadding (float): amount of time to clip near TESS perigee to
        remove ramp signals. Typically 30 minutes. (Must give in units of
        days).

        expected_norbits (int): expected number of spacecraft orbits. Used to
        ensure the group splitting works.

    returns:
        time, flux, err_flux, lightcurvedictionary

    Step 0. Retrieve the lightcurve. Do nothing else to it.

    Step 1. Filter out non-zero quality flags.  Filter out points with
    non-finite times, PDC fluxes, or PDC errors on fluxes.

    Step 2.  Clip out the first and last 30 minutes of each orbit.  To do this,
    get time groups for spacecraft orbits using the `orbitgap` kwarg. Then
    iterate over spacecraft orbits to make the time masks.

    Step 3. Clip out over ~10 minute windows around momentum dumps. Quality
    flags are supposed to purge this, but it never hurts to be cautious.
    '''

    # Step 0. Retrieve the lightcurve. Do nothing else to it.
    if lcfile.endswith('.fits'):
        lcd = at.read_tess_fitslc(lcfile, normalize=False)
    else:
        raise NotImplementedError('expected .fits files, got {:s}'.
                                  format(lcfile))

    step0_time = lcd['time']
    step0_flux = lcd['pdc']['pdcsap_flux']
    step0_err_flux = lcd['pdc']['pdcsap_flux_err']

    # note which times have data quality flags with "bit 5 set (Reaction Wheel
    # desaturation Event) and bit 7 set (Manual Exclude)---see the SDPDD section
    # 9". according to
    # https://archive.stsci.edu/hlsps/tess-data-alerts/hlsp_tess-data-alerts_tess_phot_s01_tess_v1_release-notes.pdf
    # this is the combination we must worry about for momentum wheel dumps.
    # This combination is actually bit 6 and bit 8, which sum to give 160.
    step0_quality = lcd['quality']
    momentumdump_time = step0_time[step0_quality==160]
    momentumdump_flux = step0_flux[step0_quality==160]

    # Step 1. Filter out non-zero quality flags.  Filter out points with
    # non-finite times, PDC fluxes, or PDC errors on fluxes.
    lcd = at.filter_tess_lcdict(lcd, filterqualityflags=True,
                                nanfilter='pdc,time')

    step1_time = lcd['time']
    step1_flux = lcd['pdc']['pdcsap_flux']
    step1_err_flux = lcd['pdc']['pdcsap_flux_err']

    # Step 2.  Clip out the first and last 30 minutes of each orbit.  To do
    # this, get time groups for spacecraft orbits. Then iterate over spacecraft
    # orbits to make the time masks.
    norbits, groups = lcmath.find_lc_timegroups(step1_time, mingap=orbitgap)

    if norbits != expected_norbits:
        raise_error = True
        errmsg = (
            'expected {} orbits. got {}.'.format(expected_norbits, norbits)
        )
        if norbits == 1:
            # default gap was too wide.
            norbits, groups = lcmath.find_lc_timegroups(step1_time, mingap=0.3)
            raise_error = False

        ticid = np.int64(search('{}/tic_{}/{}',lcfile)[1])
        if ticid in many_gaps_expected or sectornum in GAPPY_SECTORS:
            raise_error = False

        if raise_error:
            plt.scatter(step1_time, step1_flux)
            plt.savefig('temp_{}.png'.format(ticid))
            raise AssertionError(errmsg)

    masked_times = []
    for group in groups:
        tg_time = step1_time[group]
        start_mask = (np.min(tg_time), np.min(tg_time) + orbitpadding)
        end_mask = (np.max(tg_time) - orbitpadding, np.max(tg_time))
        masked_times.append(start_mask)
        masked_times.append(end_mask)

    lcd = at.filter_tess_lcdict(lcd, filterqualityflags=False, nanfilter=None,
                                timestoignore=masked_times)
    step2_time = lcd['time']
    step2_flux = lcd['pdc']['pdcsap_flux']
    step2_err_flux = lcd['pdc']['pdcsap_flux_err']

    # Step 3. Continue filtering bad times. When did the spacecraft fire
    # thrusters to dump momentum?  NB. quality flags are supposed to purge
    # this, but it never hurts to be cautious.
    ndumps, groups = lcmath.find_lc_timegroups(momentumdump_time,
                                               mingap=dump_interval)

    if ndumps != expected_ndumps:
        print(
            'WRN!: expected {:d} dumps, got {:d} dumps'.
            format(expected_ndumps, ndumps))

    masked_times = []
    for group in groups:
        tg_time = momentumdump_time[group]
        premask = (np.min(tg_time)-dumppadding, np.min(tg_time))
        postmask = (np.max(tg_time), np.max(tg_time)+dumppadding)
        masked_times.append(premask)
        masked_times.append(postmask)

    lcd = at.filter_tess_lcdict(lcd, filterqualityflags=False, nanfilter=None,
                                timestoignore=masked_times)
    step3_time = lcd['time']
    step3_flux = lcd['pdc']['pdcsap_flux']
    step3_err_flux = lcd['pdc']['pdcsap_flux_err']

    if make_diagnostic_plots:

        n_steps = 4 # 1-based

        alltimes = [step0_time, step1_time, step2_time, step3_time]
        allfluxs = [step0_flux, step1_flux, step2_flux, step3_flux]
        allfluxerrs = [step0_err_flux, step1_err_flux, step2_err_flux,
                       step3_err_flux]

        f, axs = plt.subplots(nrows=n_steps, ncols=1, sharex=True,
                             figsize=(14,n_steps*3))

        axs = axs.flatten()

        for ix, ax in enumerate(axs):
            time, flux, err = alltimes[ix], allfluxs[ix], allfluxerrs[ix]

            if ix == 3:
                ax.scatter(time, flux/np.nanmedian(flux), c='k', alpha=0.5,
                           zorder=1, s=10, rasterized=True, linewidths=0)
                continue

            ax.scatter(time, flux, c='k', alpha=0.5, zorder=1, s=10,
                       rasterized=True, linewidths=0)
            if ix == 0:
                minflux = np.nanpercentile(step0_flux, 1)
                ax.scatter(momentumdump_time,
                           minflux*np.ones_like(momentumdump_time), c='r',
                           alpha=1, zorder=2, s=20, rasterized=True,
                           linewidths=0, marker='^')

        ticid = int(lcd['objectid'].lstrip('TIC').strip())
        savdir = '../results/lc_analysis/'+str(ticid)+'/'
        if not os.path.exists(savdir):
            os.mkdir(savdir)
        savdir += 'sector_{:d}'.format(sectornum)
        if not os.path.exists(savdir):
            os.mkdir(savdir)
        savpath = os.path.join(
            savdir,'{:d}_retrieve_no_whitening.png'.format(ticid))
        f.tight_layout()
        f.savefig(savpath, dpi=300, bbox_inches='tight')
        print('saved {:s}'.format(savpath))

    time = lcd['time']
    flux = lcd['pdc']['pdcsap_flux']
    err_flux = lcd['pdc']['pdcsap_flux_err']
    fluxmedian = np.median(flux)
    flux /= fluxmedian
    err_flux /= fluxmedian
    return time, flux, err_flux, lcd


def trapezoidal_spot_model(time, t_spot_mid, T_dur_spot=20/(60*24),
                           antidepth_spot=0.003):
    """
    a trapezoidal spot crossing model: a positive triangle in relative flux vs
    time.  (centered at t_spot_mid, of duration T_dur_spot, of positive depth
    antidepth_spot).

    args:
        time (np.ndarray): times at which to evaluate the model. units are
        assumed days.

        t_spot_mid (float): should be inside times

        T_dur_spot (float): duration, in days, of spot crossing event. default
        is 20 minutes.

        antidepth_spot (float): the height of the spot crossing anomaly in
        relative flux units. default: 0.3%
    """

    if not isinstance(time, np.ndarray):
        raise AssertionError('expected time to be np.ndarray')

    spot_flux = np.zeros_like(time)

    t_start = t_spot_mid - T_dur_spot/2
    t_end = t_spot_mid + T_dur_spot/2

    # before triangle peak
    prepeak = (time >= t_start) & (time <= t_spot_mid)
    spot_flux[prepeak] = (
        antidepth_spot * (time[prepeak] - t_spot_mid) / (T_dur_spot/2)
        +
        antidepth_spot
    )

    # after triangle peak
    postpeak = (time > t_spot_mid) & (time <= t_end)
    spot_flux[postpeak] = (
        - antidepth_spot * (time[postpeak] - t_spot_mid) / (T_dur_spot/2)
        +
        antidepth_spot
    )

    return spot_flux



def fit_transit_mandelagol_and_line(
    sectornum,
    transit_ix, t_start, t_end, time, flux, err_flux, lcfile, fitd,
    trapfit, fixparamdf, ticid, fit_savdir, chain_savdir, nworkers,
    n_mcmc_steps, overwriteexistingsamples, mcmcprogressbar,
    getspocparams, timeoffset, fit_ulinear, fit_uquad,
    inject_spot_crossings=False, tdur=None, seed=42):
    # tdur: transit duration

    lit_period = float(fixparamdf.period_day)
    lit_a_by_rstar = float(fixparamdf.a_by_rstar)
    lit_incl = float(fixparamdf.inclination_deg)
    lit_logg = float(fixparamdf.logg)

    bls_rp = np.sqrt(trapfit['fitinfo']['finalparams'][2])
    u_claret_linear, u_claret_quad = (
        get_limb_darkening_initial_guesses(lcfile, lit_logg)
    )
    u_linear, u_quad = fit_ulinear, fit_uquad
    print('claret 2017 u_linear: {}, u_quad: {}'.
          format(u_claret_linear, u_claret_quad))
    print('fit u_linear: {}, u_quad: {}'.
          format(fit_ulinear, fit_uquad))
    print('WRN!: the FIT values were used, not the theory ones.')

    sel = (time < t_end) & (time > t_start)
    sel_time = time[sel]
    sel_flux = flux[sel]
    sel_err_flux = err_flux[sel]

    t_spot_mid = None
    if inject_spot_crossings:
        if not tdur:
            raise AssertionError('need tdur for trapezoidal spot model')
        # inject spot randomly anywhere in [-0.6,0.6] transit durations of the
        # mid-transit point.
        randphase = np.random.uniform(-0.6, 0.6)
        t_spot_mid = np.median(sel_time) + randphase*tdur
        # T_dur_spot: spot anomaly duration
        sel_flux += trapezoidal_spot_model(sel_time, t_spot_mid,
                                           T_dur_spot=30/(60*24),
                                           antidepth_spot=0.003)
        print('Transit {}: added spot at phase {}'.
              format(transit_ix, randphase))

    # model = transit + line. "transit" as defined by BATMAN has flux=1 out of
    # transit. so our bounds are for a line that should pass near origin.
    fittype = 'mandelagol_and_line'
    initfitparams = {'t0':t_start + (t_end-t_start)/2.,
                     'poly_order0':0,
                     'poly_order1':0.,
                     'rp':bls_rp}
    fixedparams = {'ecc':0.,
                   'omega':90.,
                   'limb_dark':'quadratic',
                   'period':lit_period,
                   'u':[u_linear,u_quad],
                   'sma':lit_a_by_rstar,
                   'incl':lit_incl }
    priorbounds = {'t0':(t_start, t_end),
                   'poly_order0':(-1,1),
                   'poly_order1':(-0.5,0.5),
                   'rp':(0.7*bls_rp, 1.3*bls_rp)}
    ndims = len(initfitparams)

    ##########################################################
    # FIRST: run the fit using the errors given in the data. #
    ##########################################################
    t_num = str(transit_ix).zfill(3)
    mandelagolfit_plotname = (
        str(ticid)+
        '_{:s}_fit_{:d}d_t{:s}_dataerrs.png'.format(fittype, ndims, t_num)
    )
    corner_plotname = (
        str(ticid)+
        '_corner_{:s}_fit_{:d}d_t{:s}_dataerrs.png'.format(fittype, ndims, t_num)
    )
    sample_plotname = (
        str(ticid)+
        '_{:s}_fit_samples_{:d}d_t{:s}_dataerrs.h5'.format(fittype, ndims, t_num)
    )

    mandelagolfit_savfile = os.path.join(fit_savdir, mandelagolfit_plotname)
    corner_savfile = os.path.join(fit_savdir, corner_plotname)
    if not os.path.exists(chain_savdir):
        try:
            os.mkdir(chain_savdir)
        except:
            raise AssertionError('you need to save chains')
    samplesavpath = os.path.join(chain_savdir, sample_plotname)

    print('beginning {:s}'.format(samplesavpath))

    plt.close('all')
    maf_data_errs = lcfit.mandelagol_and_line_fit_magseries(
                    sel_time, sel_flux, sel_err_flux,
                    initfitparams, priorbounds, fixedparams,
                    trueparams=None, magsarefluxes=True,
                    sigclip=None, plotfit=mandelagolfit_savfile,
                    plotcorner=corner_savfile,
                    samplesavpath=samplesavpath, nworkers=nworkers,
                    n_mcmc_steps=n_mcmc_steps, eps=1e-6, n_walkers=500,
                    skipsampling=False,
                    overwriteexistingsamples=overwriteexistingsamples,
                    mcmcprogressbar=mcmcprogressbar, timeoffset=timeoffset,
                    scatterxdata=t_spot_mid, scatteryaxes=0.05)

    fitparamdir = "../results/tess_lightcurve_fit_parameters/"+str(ticid)
    if inject_spot_crossings:
        fitparamdir += "_inject_spot_crossings_seed{}".format(seed)
    if not os.path.exists(fitparamdir):
        os.mkdir(fitparamdir)
    fitparamdir = fitparamdir+'/'+'sector_{:d}'.format(sectornum)
    if not os.path.exists(fitparamdir):
        os.mkdir(fitparamdir)
    maf_savpath = ( os.path.join(
        fitparamdir,
        str(ticid)+"_{:s}_fit_dataerrs_t{:s}.pickle".format(fittype, t_num)
    ) )
    with open(maf_savpath, 'wb') as f:
        pickle.dump(maf_data_errs, f, pickle.HIGHEST_PROTOCOL)
        print('saved {:s}'.format(maf_savpath))

    fitfluxs = maf_data_errs['fitinfo']['fitmags']
    fitepoch = maf_data_errs['fitinfo']['fitepoch']
    fiterrors = maf_data_errs['fitinfo']['finalparamerrs']
    fitepoch_perr = fiterrors['std_perrs']['t0']
    fitepoch_merr = fiterrors['std_merrs']['t0']

    # Winn (2010) eq 14 gives the transit duration
    k = maf_data_errs['fitinfo']['finalparams']['rp']
    b = lit_a_by_rstar * np.cos(lit_incl*u.deg)
    t_dur_day = (
        (lit_period*u.day)/np.pi * np.arcsin(
            1/lit_a_by_rstar * np.sqrt(
                (1 + k)**2 - b**2
            ) / np.sin((lit_incl*u.deg))
        )
    ).to(u.day*u.rad).value

    per_point_cadence = 2*u.min
    npoints_in_transit = (
        int(np.floor(((t_dur_day*u.day)/per_point_cadence).cgs.value))
    )

    # get in-transit indices, so that rms can be measured of the residual _in
    # transit_, rather than for the full timeseries.
    post_ingress = ( (fitepoch - timeoffset) - t_dur_day/2 < sel_time )
    pre_egress = ( sel_time < (fitepoch - timeoffset) + t_dur_day/2  )
    indsintransit = post_ingress & pre_egress
    indsoot = ~indsintransit

    snr, _, empirical_errs = get_snr_of_dip(
        sel_time, sel_flux, sel_time, fitfluxs,
        magsarefluxes=True, atol_normalization=1e-2,
        transitdepth=k**2, npoints_in_transit=npoints_in_transit,
        indsforrms=indsoot)

    sigma_tc_theory = estimate_achievable_tmid_precision(
        snr, t_ingress_min=0.05*t_dur_day*24*60,
        t_duration_hr=t_dur_day*24)

    print('mean fitepoch err: {:.2e}'.format(
          np.mean([fitepoch_merr, fitepoch_perr])))
    print('mean fitepoch err / theory err = {:.2e}'.format(
          np.mean([fitepoch_merr, fitepoch_perr]) / sigma_tc_theory))
    print('mean error from data lightcurve ='+
          '{:.2e}'.format(np.mean(sel_err_flux))+
          '\nmeasured empirical RMS = {:.2e}'.format(empirical_errs))

    empirical_err_flux = np.ones_like(sel_err_flux)*empirical_errs

    # THEN: rerun the fit using the empirically determined errors
    # (measured from RMS of the transit-model subtracted lightcurve).
    mandelagolfit_plotname = (
        str(ticid)+
        '_{:s}_fit_{:d}d_t{:s}_empiricalerrs.png'.format(fittype, ndims, t_num)
    )
    corner_plotname = (
        str(ticid)+
        '_corner_{:s}_fit_{:d}d_t{:s}_empiricalerrs.png'.format(fittype, ndims, t_num)
    )
    sample_plotname = (
        str(ticid)+
        '_{:s}_fit_samples_{:d}d_t{:s}_empiricalerrs.h5'.format(fittype, ndims, t_num)
    )

    mandelagolfit_savfile = os.path.join(fit_savdir, mandelagolfit_plotname)
    corner_savfile = os.path.join(fit_savdir, corner_plotname)
    samplesavpath = os.path.join(chain_savdir, sample_plotname)

    print('beginning {:s}'.format(samplesavpath))

    plt.close('all')
    maf_empc_errs = lcfit.mandelagol_and_line_fit_magseries(
                    sel_time, sel_flux, empirical_err_flux,
                    initfitparams, priorbounds, fixedparams,
                    trueparams=initfitparams, magsarefluxes=True,
                    sigclip=None, plotfit=mandelagolfit_savfile,
                    plotcorner=corner_savfile,
                    samplesavpath=samplesavpath, nworkers=nworkers,
                    n_mcmc_steps=n_mcmc_steps, eps=1e-6, n_walkers=500,
                    skipsampling=False,
                    overwriteexistingsamples=overwriteexistingsamples,
                    mcmcprogressbar=mcmcprogressbar, timeoffset=timeoffset,
                    scatterxdata=t_spot_mid, scatteryaxes=0.05)

    maf_savpath = ( os.path.join(
        fitparamdir,
        str(ticid)+"_{:s}_fit_empiricalerrs_t{:s}.pickle".format(fittype, t_num)
    ) )
    with open(maf_savpath, 'wb') as f:
        pickle.dump(maf_empc_errs, f, pickle.HIGHEST_PROTOCOL)
        print('saved {:s}'.format(maf_savpath))



def fit_phased_transit_mandelagol_and_line(
    sectornum,
    t_starts, t_ends, time, flux, err_flux, lcfile, fitd,
    trapfit, bls_period, litdf, ticid, fit_savdir, chain_savdir, nworkers,
    n_mcmc_steps, overwriteexistingsamples, mcmcprogressbar):

    # parse initial guesses
    lit_period = float(litdf.period_day)
    lit_a_by_rstar = float(litdf.a_by_rstar)
    lit_incl = float(litdf.inclination_deg)
    lit_logg = float(litdf.logg)

    assert lit_period > 0
    assert lit_a_by_rstar > 0
    assert 90 > lit_incl > 0
    assert lit_logg > 0

    bls_rp = np.sqrt(trapfit['fitinfo']['finalparams'][2])
    bls_t0 = trapfit['fitinfo']['fitepoch']
    bls_period = bls_period
    u_linear, u_quad = get_limb_darkening_initial_guesses(lcfile, lit_logg)

    b = lit_a_by_rstar * np.cos(lit_incl*u.deg)
    if not 0 > b > 1:
        b = np.pi/4
        wrnmsg = (
            'WRN! TIC{}: Artificially setting b=pi/4'.
            format(ticid)
        )
        print(wrnmsg)

    bls_lit_t_dur_day = (
        (lit_period*u.day)/np.pi * np.arcsin(
            1/lit_a_by_rstar * np.sqrt(
                (1 + bls_rp)**2 - b**2
            ) / np.sin((lit_incl*u.deg))
        )
    ).to(u.day*u.rad).value

    if pd.isnull(bls_lit_t_dur_day):
        errmsg = (
            'TIC{}: got nan bls_lit_t_dur_day. setting b='.
            format(ticid)
        )
        raise AssertionError(errmsg)

    # fit only +/- n_transit_durations near the transit data. don't try to fit
    # OOT or occultation data.
    sel_inds = np.zeros_like(time).astype(bool)
    for t_start,t_end in zip(t_starts, t_ends):
        these_inds = (time > t_start) & (time < t_end)
        if np.any(these_inds):
            sel_inds |= these_inds

    sel_time = time[sel_inds]
    _ = flux[sel_inds]
    sel_err_flux = err_flux[sel_inds]

    # to construct the phase-folded light curve, fit a line to the OOT flux
    # data, and use the parameters of the best-fitting line to "rectify" each
    # lightcurve. Note that an order 1 legendre polynomial == a line, so we'll
    # use that implementation.
    out_fluxs, in_fluxs, fit_fluxs, time_list, intra_inds_list = [], [], [], [], []
    N_intra = 0
    for t_start,t_end in zip(t_starts, t_ends):
        this_window_inds = (time > t_start) & (time < t_end)
        tmid = t_start + (t_end-t_start)/2
        # flag out slightly more than expected "in transit" points
        prefactor = 1.05
        transit_start = tmid - prefactor*bls_lit_t_dur_day/2
        transit_end = tmid + prefactor*bls_lit_t_dur_day/2

        this_window_intra = (
            (time[this_window_inds] > transit_start) &
            (time[this_window_inds] < transit_end)
        )
        this_window_oot = ~this_window_intra

        this_oot_time = time[this_window_inds][this_window_oot]
        this_oot_flux = flux[this_window_inds][this_window_oot]

        if len(this_oot_flux) == len(this_oot_time) == 0:
            continue
        elif len(this_oot_flux) == len(this_oot_time) == 1:
            # if you have a single point in window, don't fit, just append and
            # continue. (otherwise will break select from time indices above)
            time_list.append( time[this_window_inds] )
            out_fluxs.append( flux[this_window_inds] )
            fit_fluxs.append( flux[this_window_inds] )
            in_fluxs.append( flux[this_window_inds] )
            intra_inds_list.append( (time[this_window_inds]>transit_start) &
                                    (time[this_window_inds]<transit_end) )
            continue

        p = Legendre.fit(this_oot_time, this_oot_flux, 1)
        coeffs = p.coef
        this_window_fit_flux = p(time[this_window_inds])

        time_list.append( time[this_window_inds] )
        out_fluxs.append( flux[this_window_inds] / this_window_fit_flux )
        fit_fluxs.append( this_window_fit_flux )
        in_fluxs.append( flux[this_window_inds] )
        intra_inds_list.append( (time[this_window_inds]>transit_start) &
                                (time[this_window_inds]<transit_end) )

        N_intra += len(time[this_window_inds][this_window_intra])

    # make plots to verify that this procedure is working.
    ix = 0
    for _time, _flux, _fit_flux, _out_flux, _intra in zip(
        time_list, in_fluxs, fit_fluxs, out_fluxs, intra_inds_list):

        outdir = ('../results/lc_analysis/{:s}/sector_{:d}/'.
                  format(str(ticid),sectornum))
        savpath = ( outdir+ '{:s}_phased_divideOOTline_t{:s}.png'.
                    format(str(ticid),str(ix).zfill(3)))

        if os.path.exists(savpath):
            print('found & skipped making {}'.format(savpath))
            ix += 1
            continue

        plt.close('all')
        fig, (a0,a1) = plt.subplots(nrows=2, sharex=True, figsize=(6,6))

        a0.scatter(_time, _flux, c='k', alpha=0.9, label='data', zorder=1,
                   s=10, rasterized=True, linewidths=0)
        a0.scatter(_time[_intra], _flux[_intra], c='r', alpha=1,
                   label='in-transit (for fit)', zorder=2, s=10, rasterized=True,
                   linewidths=0)

        a0.plot(_time, _fit_flux, c='b', zorder=0, rasterized=True, lw=2,
                alpha=0.4, label='linear fit to OOT')

        a1.scatter(_time, _out_flux, c='k', alpha=0.9, rasterized=True,
                   s=10, linewidths=0)
        a1.plot(_time, _fit_flux/_fit_flux, c='b', zorder=0, rasterized=True,
                lw=2, alpha=0.4, label='linear fit to OOT')

        xlim = a1.get_xlim()

        for a in [a0,a1]:
            a.hlines(1, np.min(_time)-10, np.max(_time)+10, color='gray',
                     zorder=-2, rasterized=True, alpha=0.2, lw=1,
                     label='flux=1')

        a1.set_xlabel('time-t0 [days]')
        a0.set_ylabel('relative flux')
        a1.set_ylabel('residual')
        a0.legend(loc='best', fontsize='x-small')
        for a in [a0, a1]:
            a.get_yaxis().set_tick_params(which='both', direction='in')
            a.get_xaxis().set_tick_params(which='both', direction='in')
            a.set_xlim(xlim)

        fig.tight_layout(h_pad=0, w_pad=0)
        fig.savefig(savpath, dpi=300, bbox_inches='tight')
        print('saved {:s}'.format(savpath))
        ix += 1

    if N_intra < 10:
        errmsg = (
            'TIC{}: only got {} points in transits. something wrong.'.
            format(ticid, N_intra)
        )
        raise AssertionError(errmsg)

    sel_flux = np.concatenate(out_fluxs)
    fit_flux = np.concatenate(fit_fluxs)
    assert len(sel_flux) == len(sel_time) == len(sel_err_flux)

    # model = transit only (no line). "transit" as defined by BATMAN has flux=1
    # out of transit.
    fittype = 'mandelagol'
    initfitparams = {'t0':bls_t0,
                     'period':bls_period,
                     'sma':lit_a_by_rstar,
                     'rp':bls_rp,
                     'incl':lit_incl,
                     'u':[u_linear,u_quad]
                    }
    fixedparams = {'ecc':0.,
                   'omega':90.,
                   'limb_dark':'quadratic'}
    priorbounds = {'t0':(bls_t0 - lit_period/10, bls_t0 + lit_period/10),
                   'period':(bls_period-1e-2, bls_period+1e-2),
                   'sma':(lit_a_by_rstar/3, 3*lit_a_by_rstar),
                   'rp':(0.7*bls_rp, 1.3*bls_rp),
                   'incl':(lit_incl-10, 90),
                   'u_linear':(u_linear-1, u_linear+1),
                   'u_quad':(u_quad-1, u_quad+1)
                   }
    cornerparams = {'t0':bls_t0,
                    'period':bls_period,
                    'sma':lit_a_by_rstar,
                    'rp':bls_rp,
                    'incl':lit_incl,
                    'u_linear':u_linear,
                    'u_quad':u_quad }

    ndims = len(initfitparams)

    ##########################################################
    # FIRST: run the fit using the errors given in the data. #
    ##########################################################
    mandelagolfit_plotname = (
        str(ticid)+
        '_phased_{:s}_fit_{:d}d_dataerrs.png'.format(fittype, ndims)
    )
    corner_plotname = (
        str(ticid)+
        '_phased_corner_{:s}_fit_{:d}d_dataerrs.png'.format(fittype, ndims)
    )
    sample_plotname = (
        str(ticid)+
        '_phased_{:s}_fit_samples_{:d}d_dataerrs.h5'.format(fittype, ndims)
    )

    mandelagolfit_savfile = os.path.join(fit_savdir, mandelagolfit_plotname)
    corner_savfile = os.path.join(fit_savdir, corner_plotname)
    if not os.path.exists(chain_savdir):
        try:
            os.mkdir(chain_savdir)
        except:
            raise AssertionError('you need to save chains')
    samplesavpath = os.path.join(chain_savdir, sample_plotname)

    print('beginning {:s}'.format(samplesavpath))

    plt.close('all')

    maf_data_errs = lcfit.mandelagol_fit_magseries(
                    sel_time, sel_flux, sel_err_flux,
                    initfitparams, priorbounds, fixedparams,
                    trueparams=cornerparams, magsarefluxes=True,
                    sigclip=None, plotfit=mandelagolfit_savfile,
                    plotcorner=corner_savfile,
                    samplesavpath=samplesavpath, nworkers=nworkers,
                    n_mcmc_steps=n_mcmc_steps, eps=1e-6, n_walkers=500,
                    skipsampling=False,
                    overwriteexistingsamples=overwriteexistingsamples,
                    mcmcprogressbar=mcmcprogressbar)

    fitparamdir = "../results/tess_lightcurve_fit_parameters/"+str(ticid)
    if not os.path.exists(fitparamdir):
        os.mkdir(fitparamdir)
    fitparamdir = fitparamdir+'/'+'sector_{:d}'.format(sectornum)
    if not os.path.exists(fitparamdir):
        os.mkdir(fitparamdir)
    maf_savpath = ( os.path.join(
        fitparamdir,
        str(ticid)+"_phased_{:s}_fit_dataerrs.pickle".format(fittype)
    ) )
    with open(maf_savpath, 'wb') as f:
        pickle.dump(maf_data_errs, f, pickle.HIGHEST_PROTOCOL)
        print('saved {:s}'.format(maf_savpath))

    fitfluxs = maf_data_errs['fitinfo']['fitmags']
    fitepoch = maf_data_errs['fitinfo']['fitepoch']
    fiterrors = maf_data_errs['fitinfo']['finalparamerrs']
    fitepoch_perr = fiterrors['std_perrs']['t0']
    fitepoch_merr = fiterrors['std_merrs']['t0']

    # Winn (2010) eq 14 gives the transit duration
    k = maf_data_errs['fitinfo']['finalparams']['rp']
    t_dur_day = (
        (lit_period*u.day)/np.pi * np.arcsin(
            1/lit_a_by_rstar * np.sqrt(
                (1 + k)**2 - b**2
            ) / np.sin((lit_incl*u.deg))
        )
    ).to(u.day*u.rad).value

    per_point_cadence = 2*u.min
    npoints_in_transit = (
        int(np.floor(((t_dur_day*u.day)/per_point_cadence).cgs.value))
    )

    # use the whole LC's RMS as the "noise"
    snr, _, empirical_errs = get_snr_of_dip(
        sel_time, sel_flux, sel_time, fitfluxs,
        magsarefluxes=True, atol_normalization=1e-2,
        transitdepth=k**2, npoints_in_transit=npoints_in_transit)

    sigma_tc_theory = estimate_achievable_tmid_precision(
        snr, t_ingress_min=0.05*t_dur_day*24*60,
        t_duration_hr=t_dur_day*24)

    print('mean fitepoch err: {:.2e}'.format(
          np.mean([fitepoch_merr, fitepoch_perr])))
    print('mean fitepoch err / theory err = {:.2e}'.format(
          np.mean([fitepoch_merr, fitepoch_perr]) / sigma_tc_theory))
    print('mean error from data lightcurve ='+
          '{:.2e}'.format(np.mean(sel_err_flux))+
          '\nmeasured empirical RMS = {:.2e}'.format(empirical_errs))

    empirical_err_flux = np.ones_like(sel_err_flux)*empirical_errs

    # THEN: rerun the fit using the empirically determined errors
    # (measured from RMS of the transit-model subtracted lightcurve).
    mandelagolfit_plotname = (
        str(ticid)+
        '_phased_{:s}_fit_{:d}d_empiricalerrs.png'.format(fittype, ndims)
    )
    corner_plotname = (
        str(ticid)+
        '_phased_corner_{:s}_fit_{:d}d_empiricalerrs.png'.format(fittype, ndims)
    )
    sample_plotname = (
        str(ticid)+
        '_phased_{:s}_fit_samples_{:d}d_empiricalerrs.h5'.format(fittype, ndims)
    )

    mandelagolfit_savfile = os.path.join(fit_savdir, mandelagolfit_plotname)
    corner_savfile = os.path.join(fit_savdir, corner_plotname)
    samplesavpath = os.path.join(chain_savdir, sample_plotname)

    print('beginning {:s}'.format(samplesavpath))

    plt.close('all')
    maf_empc_errs = lcfit.mandelagol_fit_magseries(
                    sel_time, sel_flux, empirical_err_flux,
                    initfitparams, priorbounds, fixedparams,
                    trueparams=cornerparams, magsarefluxes=True,
                    sigclip=None, plotfit=mandelagolfit_savfile,
                    plotcorner=corner_savfile,
                    samplesavpath=samplesavpath, nworkers=nworkers,
                    n_mcmc_steps=n_mcmc_steps, eps=1e-6, n_walkers=500,
                    skipsampling=False,
                    overwriteexistingsamples=overwriteexistingsamples,
                    mcmcprogressbar=mcmcprogressbar)

    maf_savpath = ( os.path.join(
        fitparamdir,
        str(ticid)+"_phased_{:s}_fit_empiricalerrs.pickle".format(fittype)
    ) )
    with open(maf_savpath, 'wb') as f:
        pickle.dump(maf_empc_errs, f, pickle.HIGHEST_PROTOCOL)
        print('saved {:s}'.format(maf_savpath))

    # now plot the phased lightcurve
    fitfluxs = maf_empc_errs['fitinfo']['fitmags']
    fittimes = maf_empc_errs['magseries']['times']
    fitepoch = maf_empc_errs['fitinfo']['finalparams']['t0']
    outfile = ( os.path.join(
        fit_savdir,
        str(ticid)+"_phased_{:s}_fit_empiricalerrs.png".format(fittype)
    ) )

    plot_phased_magseries(sel_time, sel_flux, lit_period, magsarefluxes=True,
                           errs=None, normto=False, epoch=fitepoch,
                           outfile=outfile, sigclip=False, phasebin=0.01,
                           plotphaselim=[-.4,.4], plotdpi=400,
                           modelmags=fitfluxs, modeltimes=fittimes,
                           xaxlabel='Time from mid-transit [days]',
                           yaxlabel='Relative flux', xtimenotphase=True)
    print('made {}'.format(outfile))

    # pass numbers that will be fixed during single-transit fitting
    fit_abyrstar, fit_incl, fit_ulinear, fit_uquad = (
        maf_empc_errs['fitinfo']['finalparams']['sma'],
        maf_empc_errs['fitinfo']['finalparams']['incl'],
        maf_empc_errs['fitinfo']['finalparams']['u_linear'],
        maf_empc_errs['fitinfo']['finalparams']['u_quad']
    )
    return fit_abyrstar, fit_incl, fit_ulinear, fit_uquad


def measure_transit_times_from_lightcurve(
    ticid, n_mcmc_steps, n_phase_mcmc_steps, getspocparams=False,
    read_literature_params=True, overwriteexistingsamples=False,
    mcmcprogressbar=False, nworkers=4, chain_savdir=None, lcdir=None,
    n_transit_durations=10, verify_times=False, inject_spot_crossings=False,
    seed=42
    ):

    np.random.seed(seed)

    make_diagnostic_plots = True
    ##########################################

    if not lcdir:
        raise AssertionError('input directory to find lightcurves')
    if not os.path.exists(lcdir):
        os.mkdir(lcdir)
    chain_base = deepcopy(chain_savdir)

    #
    # query MAST, via astrobase and lightkurve APIs, to first get ra/dec
    # given TICID, then to get lightcurve given ra/dec.
    #

    ticres = tic_objectsearch(ticid)

    with open(ticres['cachefname'], 'r') as json_file:
        data = json.load(json_file)

    ra = data['data'][0]['ra']
    dec = data['data'][0]['dec']

    targetcoordstr = '{} {}'.format(ra, dec)

    res = search_lightcurvefile(targetcoordstr, radius=0.1,
                                cadence='short', mission='TESS')

    if len(res.table)==0:
        errmsg = (
            'failed to get any SC data for TIC{}. need other LC source.'.
            format(ticid)
        )
        raise AssertionError(errmsg)

    available_sectors = list(res.table['sequence_number'])

    res.download_all(download_dir=lcdir)

    # For each sector, fit available data with its own phase-fold. Then fit for
    # the individual transit times.  (This is a bit wrong for the joint
    # phase-fold, but unless you're aiming for transit times and not
    # phase-curve science this shouldn't be an issue).
    for sectornum in available_sectors:

        lcfiles = glob(os.path.join(
            lcdir, 'mastDownload', 'TESS', 'tess*',
            'tess*-s{}-*_lc.fits'.format(str(sectornum).zfill(4))))

        if len(lcfiles) != 1:
            import IPython; IPython.embed()
            raise AssertionError(
                'expected to operate on one sector of SC data.'
            )

        lcfile = lcfiles[0]

        fit_savdir = os.path.join('../results/lc_analysis',str(ticid))
        if inject_spot_crossings:
            fit_savdir += '_inject_spot_crossings_seed{}'.format(seed)
        if not os.path.exists(fit_savdir):
            os.mkdir(fit_savdir)
        fit_savdir = os.path.join(fit_savdir, 'sector_'+str(sectornum))
        if not os.path.exists(fit_savdir):
            os.mkdir(fit_savdir)
        chain_savdir = os.path.join(chain_base, 'sector_'+str(sectornum))
        if inject_spot_crossings:
            chain_savdir += '_inject_spot_crossings_seed{}'.format(seed)
        if not os.path.exists(chain_savdir):
            os.mkdir(chain_savdir)
        blsfit_plotname = str(ticid)+'_bls_fit.png'
        trapfit_plotname = str(ticid)+'_trapezoid_fit.png'
        mandelagolfit_plotname = str(ticid)+'_mandelagol_fit_4d.png'
        corner_plotname = str(ticid)+'_corner_mandelagol_fit_4d.png'
        sample_plotname = str(ticid)+'_mandelagol_fit_samples_4d.h5'

        blsfit_savfile = os.path.join(fit_savdir, blsfit_plotname)
        trapfit_savfile = os.path.join(fit_savdir, trapfit_plotname)
        mandelagolfit_savfile = os.path.join(fit_savdir, mandelagolfit_plotname)
        corner_savfile = os.path.join(fit_savdir, corner_plotname)
        ##########################################

        if read_literature_params:

            litdir = "../data/literature_physicalparams/{:d}/".format(ticid)
            if not os.path.exists(litdir):
                os.mkdir(litdir)
            litpath = os.path.join(litdir, 'params.csv')

            if not os.path.exists(litpath):

                eatab = NasaExoplanetArchive.get_confirmed_planets_table()
                # attempt to get physical parameters of planet -- period, a/Rstar, and
                # inclination -- for the initial guesses.
                pl_coords = eatab['sky_coord']
                tcoord = SkyCoord(targetcoordstr, frame='icrs', unit=(u.deg, u.deg))

                print('got match w/ separation {}'.format(
                    np.min(tcoord.separation(pl_coords).to(u.arcsec))))
                pl_row = eatab[np.argmin(tcoord.separation(pl_coords).to(u.arcsec))]

                # all dimensionful
                period = pl_row['pl_orbper'].value
                incl = pl_row['pl_orbincl'].value
                semimaj_au = pl_row['pl_orbsmax']
                rstar = pl_row['st_rad']
                mstar = pl_row['st_mass']
                a_by_rstar = (semimaj_au / rstar).cgs.value

                if not 90 > incl > 0:
                    # exoplanet archive can fail to report inclination
                    incl = 85

                if a_by_rstar == 0:
                    # exoplanet archive can fail to report semimajor axis
                    P = pl_row['pl_orbper']
                    a = ( P**2 * const.G*mstar / (4*np.pi**2) )**(1/3)
                    a_by_rstar = (a.cgs/rstar.cgs).value

                    if not a_by_rstar > 0 :
                        raise AssertionError(
                            'TIC{} failing to get a/rstar'.format(ticid)
                        )

                logg = np.log10( ( const.G * mstar / (rstar**2) ).cgs.value )

                litdf = pd.DataFrame(
                    {'period_day':period,
                     'a_by_rstar':a_by_rstar,
                     'inclination_deg':incl,
                     'logg':logg
                    }, index=[0]
                )
                # get the fixed physical parameters from the data. period_day,
                # a_by_rstar, and inclination_deg are comma-separated in this file.
                litdf.to_csv(litpath, index=False, header=True, sep=',')
                litdf = pd.read_csv(litpath, sep=',')
            else:
                litdf = pd.read_csv(litpath, sep=',')

        else:
            errmsg = 'read_literature_params is required'
            raise AssertionError(errmsg)

        ##########################################

        time, flux, err_flux, lcd = retrieve_no_whitening(
            lcfile, sectornum, make_diagnostic_plots=make_diagnostic_plots)

        if verify_times:
            from verify_time_stamps import manual_verify_time_stamps
            print('\nWRN! got verify_times special mode.\n')
            manual_verify_time_stamps(lcfile, lcd)
            return 1

        # run bls to get initial parameters.
        startp = float(litdf.period_day) - 0.5
        endp = float(litdf.period_day) + 0.5

        blsdict = kbls.bls_parallel_pfind(time, flux, err_flux, magsarefluxes=True,
                                          startp=startp, endp=endp,
                                          maxtransitduration=0.3, nworkers=8,
                                          sigclip=None)
        fitd = kbls.bls_stats_singleperiod(time, flux, err_flux,
                                           blsdict['bestperiod'],
                                           magsarefluxes=True, sigclip=None,
                                           perioddeltapercent=5)

        bls_period = fitd['period']
        #  plot the BLS model.
        make_fit_plot(fitd['phases'], fitd['phasedmags'], None, fitd['blsmodel'],
                      fitd['period'], fitd['epoch'], fitd['epoch'], blsfit_savfile,
                      magsarefluxes=True)

        ingduration_guess = fitd['transitduration']*0.2
        transitparams = [fitd['period'], fitd['epoch'], fitd['transitdepth'],
                         fitd['transitduration'], ingduration_guess]

        # NOTE: this is a hack. better would probably be to use the TLS
        # guesses, which are more robust, and don't get stuck in the wrong
        # minimum like half the time.
        if np.int64(ticid) in half_epoch_off:
            if sectornum in half_epoch_off[np.int64(ticid)]:
                transitparams = [fitd['period'], fitd['epoch']+fitd['period']/2,
                                 fitd['transitdepth'], fitd['transitduration'],
                                 ingduration_guess]

        # fit a trapezoidal transit model; plot the resulting phased LC.
        trapfit = lcfit.traptransit_fit_magseries(time, flux, err_flux,
                                                  transitparams,
                                                  magsarefluxes=True, sigclip=None,
                                                  plotfit=trapfit_savfile)

        period = trapfit['fitinfo']['finalparams'][0]
        t0 = trapfit['fitinfo']['fitepoch']
        transitduration_phase = trapfit['fitinfo']['finalparams'][3]
        tdur = period * transitduration_phase

        # isolate each transit to within +/- n_transit_durations
        tmids, t_starts, t_ends = (
            get_transit_times(fitd, time, n_transit_durations, trapd=trapfit)
        )

        rp = np.sqrt(trapfit['fitinfo']['finalparams'][2])

        # fit the phased transit, within N durations of the transit itself, to
        # determine a/Rstar, inclination, and quadratric terms for fixed
        # parameters. only period from literature.
        fit_abyrstar, fit_incl, fit_ulinear, fit_uquad = (
            fit_phased_transit_mandelagol_and_line(
                sectornum,
                t_starts, t_ends, time, flux, err_flux, lcfile, fitd, trapfit,
                bls_period, litdf, ticid, fit_savdir, chain_savdir, nworkers,
                n_phase_mcmc_steps, overwriteexistingsamples, mcmcprogressbar)
        )

        fixparamdf = pd.DataFrame({
            'period_day':float(litdf['period_day']),
            'a_by_rstar':fit_abyrstar,
            'inclination_deg':fit_incl,
            'logg':float(litdf['logg'])
        }, index=[0])

        for transit_ix, t_start, t_end in list(
            zip(range(len(t_starts)), t_starts, t_ends)
        ):

            timeoffset = t_start + (t_end - t_start)/2
            t_start -= timeoffset
            t_end -= timeoffset
            this_time = time - timeoffset

            try:

                # Method: fit transit + line, 4 parameters: (midtime, slope,
                # intercept, rp).
                fit_transit_mandelagol_and_line(
                    sectornum,
                    transit_ix, t_start, t_end, this_time, flux, err_flux,
                    lcfile, fitd, trapfit, fixparamdf, ticid, fit_savdir,
                    chain_savdir, nworkers, n_mcmc_steps, overwriteexistingsamples,
                    mcmcprogressbar, getspocparams, timeoffset, fit_ulinear,
                    fit_uquad, inject_spot_crossings=inject_spot_crossings,
                    tdur=tdur, seed=seed
                )

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
    parser.add_argument('--sectornum', type=int, default=1,
        help=('string used in sector number (used to glob lightcurves)'))

    parser.add_argument('--n_mcmc_steps', type=int, default=None,
        help=('steps to run in MCMC of individual transits'))
    parser.add_argument('--n_phase_mcmc_steps', type=int, default=None,
        help=('steps to run in MCMC of phased transit'))
    parser.add_argument('--nworkers', type=int, default=4,
        help=('how many workers?'))
    parser.add_argument('--n_transit_durations', type=int, default=4,
        help=('for transit model fitting, how large a time slice around the '
              'transit do you want to fit? [N*transit_duration]'))

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

    parser.add_argument('--getspocparams', dest='spocparams',
        action='store_true', help='whether to parse '
        '../data/toi-plus-2018-09-14.csv for their "true" parameters')
    parser.add_argument('--no-getspocparams', dest='spocparams',
        action='store_false')
    parser.set_defaults(spocparams=False)

    parser.add_argument('--read_literature_params', dest='rlp',
        action='store_true', help='whether to parse '
        '../data/TICID/params.csv for manually-parsed literature params')
    parser.add_argument('--no-read_literature_params', dest='rlp',
        action='store_false')
    parser.set_defaults(rlp=True)

    parser.add_argument('--verify-times', dest='verifytimes',
        action='store_true', help='whether to parse '
        '../data/TICID/params.csv for manually-parsed literature params')
    parser.add_argument('--no-verify-times', dest='verifytimes',
        action='store_false')
    parser.set_defaults(verifytimes=False)

    parser.add_argument('--injectspots', dest='injectspots',
        action='store_true', help='whether to inject trapezoidal spot '
        'crossings into each transit. this was a referee thing.')
    parser.add_argument('--no-injectspots', dest='injectspots',
                        action='store_false')
    parser.set_defaults(injectspots=False)

    parser.add_argument('--chain_savdir', type=str, default=None,
        help=('e.g., /home/foo/bar/'))
    parser.add_argument('--lcdir', type=str, default=None,
        help=('e.g., /home/foo/lightcurves/'))

    parser.add_argument('--seed', type=int, default=42,
        help=('used for random number seeding'))

    args = parser.parse_args()

    measure_transit_times_from_lightcurve(
        args.ticid, args.n_mcmc_steps,
        args.n_phase_mcmc_steps, getspocparams=args.spocparams,
        overwriteexistingsamples=args.overwrite,
        mcmcprogressbar=args.progressbar, nworkers=args.nworkers,
        chain_savdir=args.chain_savdir, lcdir=args.lcdir,
        n_transit_durations=args.n_transit_durations,
        read_literature_params=args.rlp,
        verify_times=args.verifytimes,
        inject_spot_crossings=args.injectspots,
        seed=args.seed
    )
