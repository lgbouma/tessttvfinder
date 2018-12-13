# -*- coding: utf-8 -*-
'''
usage: measure_transit_times_from_lightcurve.py [-h] [--ticid TICID]
                                                [--n_mcmc_steps N_MCMC_STEPS]
                                                [--nworkers NWORKERS]
                                                [--mcmcprogressbar]
                                                [--no-mcmcprogressbar]
                                                [--overwritesamples]
                                                [--no-overwritesamples]
                                                [--getspocparams]
                                                [--no-getspocparams]
                                                [--chain_savdir CHAIN_SAVDIR]

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
  --getspocparams       whether to parse ../data/toi-plus-2018-09-14.csv for
                        their "true" parameters
  --no-getspocparams
  --chain_savdir CHAIN_SAVDIR
                        e.g., /home/foo/bar/
'''
from __future__ import division, print_function

import os, argparse, pickle, h5py
from glob import glob

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import rc
rc('text', usetex=False)

import numpy as np, matplotlib.pyplot as plt, pandas as pd
from astropy.io import fits
from astropy import units as u, constants as const

from astrobase.varbase import lcfit
from astrobase import astrotess as at
from astrobase.periodbase import kbls
from astrobase.varbase.trends import smooth_magseries_ndimage_medfilt
from astrobase import lcmath
from astrobase.services.tic import tic_single_object_crossmatch
from astrobase.varbase.transits import get_snr_of_dip
from astrobase.varbase.transits import estimate_achievable_tmid_precision
from astrobase.plotbase import plot_phased_mag_series

from astrobase.varbase.transits import get_transit_times

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
    sep = 1*u.arcsec
    obj = tic_single_object_crossmatch(ra,dec,sep.to(u.deg).value)
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

    teff = main_hdr['TEFF']
    logg = main_hdr['LOGG']
    metallicity = main_hdr['MH']
    if not isinstance(metallicity,float):
        metallicity = 0 # solar

    ### OLD IMPLEMENTATION THAT MIGHT STRUGGLE B/C OF MULTIPLE OBJECTS:
    ### ra, dec = lc_hdr['RA_OBJ'], lc_hdr['DEC_OBJ']
    ### sep = 0.1*u.arcsec
    ### obj = tic_single_object_crossmatch(ra,dec,sep.to(u.deg).value)
    ### if len(obj['data'])==1:
    ###     teff = obj['data'][0]['Teff']
    ###     logg = obj['data'][0]['logg']
    ###     metallicity = obj['data'][0]['MH'] # often None
    ###     if not isinstance(metallicity,float):
    ###         metallicity = 0 # solar
    ### else:
    ###     raise NotImplementedError

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

    # TODO: should probably determine these coefficients by INTERPOLATING.
    # (especially in cases when you're FIXING them, rather than letting them
    # float).
    print('WRN! skipping interpolation for Claret coefficients.')
    print('WRN! data logg={:.3f}, teff={:.1f}'.format(logg, teff))
    print('WRN! Claret logg={:.3f}, teff={:.1f}'.
          format(bar['logg'],bar['Teff']))

    u_linear = bar['aLSM']
    u_quad = bar['bLSM']

    return float(u_linear), float(u_quad)


def get_alerted_params(ticid):
    import pandas as pd
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


def get_timeseries_and_median_filter(lcfile, mingap=240/(60*24)):
    # mingap: 240 minutes, in units of days

    if lcfile.endswith('.fits.gz'):
        time, flux, err_flux = (
            at.get_time_flux_errs_from_Ames_lightcurve(lcfile, 'PDCSAP')
        )
    else:
        raise NotImplementedError

    # detrending parameters. mingap: minimum gap to determine time group size.
    # smooth_window_day: window for median filtering.
    smooth_window_day = 2.
    cadence_min = 2

    cadence_day = cadence_min / 60. / 24.
    windowsize = int(smooth_window_day/cadence_day)
    if windowsize % 2 == 0:
        windowsize += 1

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

    return time, flux, err_flux, smooth_flux, whitened_flux


def retrieve_no_whitening(lcfile, make_diagnostic_plots=True, orbitgap=1.,
                          orbitpadding=60/(60*24), expected_norbits=2,
                          dump_interval=2.1, expected_ndumps=10,
                          dumppadding=10/(60*24)):
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
        raise AssertionError

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
        savdir = '../results/lc_analysis/'+str(ticid)
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


def fit_transit_mandelagol_only(transit_ix, t_start, t_end, time,
                                whitened_flux, err_flux, lcfile, fitd, trapfit,
                                getspocparams, ticid, fit_savdir,
                                chain_savdir, nworkers, n_mcmc_steps,
                                overwriteexistingsamples, mcmcprogressbar):

    if getspocparams:
        spocparamtuple = get_alerted_params(ticid)
        if isinstance(spoc_b,float) and isinstance(spoc_sma, float):
            # b = a/Rstar * cosi
            cosi = spoc_b / spoc_sma
            spoc_incl = np.degrees(np.arccos(cosi))
            spocparamtuple = tuple(list(spocparamtuple).append(spoc_incl))
    else:
        raise NotImplementedError

    spoc_b, spoc_sma, spoc_t0, spoc_rp, spoc_incl = spocparamtuple

    sel = (time < t_end) & (time > t_start)
    sel_time = time[sel]
    sel_whitened_flux = whitened_flux[sel]
    sel_err_flux = err_flux[sel]

    u_linear, u_quad = get_limb_darkening_initial_guesses(lcfile)
    a_guess = get_a_over_Rstar_guess(lcfile, fitd['period'])

    # trapezoidal fit get more robust transit depth than BLS, for some
    # reason!
    rp = np.sqrt(trapfit['fitinfo']['finalparams'][2])

    initfitparams = {'t0':t_start + (t_end-t_start)/2.,
                     'rp':rp,
                     'sma':a_guess,
                     'incl':85,
                    }

    fixedparams = {'ecc':0.,
                   'omega':90.,
                   'limb_dark':'quadratic',
                   'period':fitd['period'],
                   'u':[u_linear,u_quad]}

    priorbounds = {'rp':(rp-0.01, rp+0.01),
                   't0':(np.min(sel_time), np.max(sel_time)),
                   'sma':(0.5*a_guess,1.5*a_guess),
                   'incl':(75,90) }

    spocparams = {'rp':spoc_rp,
                  't0':spoc_t0,
                  'sma':spoc_sma,
                  'incl':spoc_incl }

    # FIRST: run the fit using the errors given in the data.
    t_num = str(transit_ix).zfill(3)
    mandelagolfit_plotname = (
        str(ticid)+
        '_mandelagol_fit_4d_t{:s}_dataerrs.png'.format(t_num)
    )
    corner_plotname = (
        str(ticid)+
        '_corner_mandelagol_fit_4d_t{:s}_dataerrs.png'.format(t_num)
    )
    sample_plotname = (
        str(ticid)+
        '_mandelagol_fit_samples_4d_t{:s}_dataerrs.h5'.format(t_num)
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
    maf_data_errs = lcfit.mandelagol_fit_magseries(
                    sel_time, sel_whitened_flux, sel_err_flux,
                    initfitparams, priorbounds, fixedparams,
                    trueparams=spocparams, magsarefluxes=True,
                    sigclip=None, plotfit=mandelagolfit_savfile,
                    plotcorner=corner_savfile,
                    samplesavpath=samplesavpath, nworkers=nworkers,
                    n_mcmc_steps=n_mcmc_steps, eps=1e-3, n_walkers=500,
                    skipsampling=False,
                    overwriteexistingsamples=overwriteexistingsamples,
                    mcmcprogressbar=mcmcprogressbar)

    fitparamdir = "../results/tess_lightcurve_fit_parameters/"+str(ticid)
    if not os.path.exists(fitparamdir):
        os.mkdir(fitparamdir)
    fitparamdir = fitparamdir+'/'+'sector_{:d}'.format(sectornum)
    if not os.path.exists(fitparamdir):
        os.mkdir(fitparamdir)
    maf_savpath = (
        os.path.join(
            fitparamdir,
            str(ticid)+ "_mandelagol_fit_dataerrs_t{:s}.pickle".format(t_num)
    ) )
    with open(maf_savpath, 'wb') as f:
        pickle.dump(maf_data_errs, f, pickle.HIGHEST_PROTOCOL)
        print('saved {:s}'.format(maf_savpath))

    fitfluxs = maf_data_errs['fitinfo']['fitmags']

    fitepoch = maf_data_errs['fitinfo']['fitepoch']
    fiterrors = maf_data_errs['fitinfo']['finalparamerrs']
    fitepoch_perr = fiterrors['std_perrs']['t0']
    fitepoch_merr = fiterrors['std_merrs']['t0']

    snr, _, empirical_errs = get_snr_of_dip(
        sel_time, sel_whitened_flux, sel_time, fitfluxs,
        magsarefluxes=True)

    # janky estimate of transit duration and ingress duration
    t_dur_day = (
        np.max(sel_time[fitfluxs < 1]) - np.min(sel_time[fitfluxs < 1])
    )

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
        '_mandelagol_fit_4d_t{:s}_empiricalerrs.png'.format(t_num)
    )
    corner_plotname = (
        str(ticid)+
        '_corner_mandelagol_fit_4d_t{:s}_empiricalerrs.png'.format(t_num)
    )
    sample_plotname = (
        str(ticid)+
        '_mandelagol_fit_samples_4d_t{:s}_empiricalerrs.h5'.format(t_num)
    )

    mandelagolfit_savfile = fit_savdir + mandelagolfit_plotname
    corner_savfile = fit_savdir + corner_plotname
    samplesavpath = chain_savdir + sample_plotname

    print('beginning {:s}'.format(samplesavpath))

    plt.close('all')
    maf_empc_errs = lcfit.mandelagol_fit_magseries(
                    sel_time, sel_whitened_flux, empirical_err_flux,
                    initfitparams, priorbounds, fixedparams,
                    trueparams=spocparams, magsarefluxes=True,
                    sigclip=None, plotfit=mandelagolfit_savfile,
                    plotcorner=corner_savfile,
                    samplesavpath=samplesavpath, nworkers=nworkers,
                    n_mcmc_steps=n_mcmc_steps, eps=2e-2, n_walkers=500,
                    skipsampling=False,
                    overwriteexistingsamples=overwriteexistingsamples,
                    mcmcprogressbar=mcmcprogressbar)

    maf_savpath = (
        os.path.join(
            fitparamdir,
            str(ticid)+ "_mandelagol_fit_empiricalerrs_t{:s}.pickle".format(t_num)
    ) )

    with open(maf_savpath, 'wb') as f:
        pickle.dump(maf_empc_errs, f, pickle.HIGHEST_PROTOCOL)
        print('saved {:s}'.format(maf_savpath))


def fit_transit_mandelagol_and_line(
    sectornum,
    transit_ix, t_start, t_end, time, flux, err_flux, lcfile, fitd,
    trapfit, litparams, ticid, fit_savdir, chain_savdir, nworkers,
    n_mcmc_steps, overwriteexistingsamples, mcmcprogressbar,
    getspocparams, timeoffset, fit_ulinear, fit_uquad):

    lit_period, lit_a_by_rstar, lit_incl = litparams
    bls_rp = np.sqrt(trapfit['fitinfo']['finalparams'][2])
    u_claret_linear, u_claret_quad = get_limb_darkening_initial_guesses(lcfile)
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
                    trueparams=litparams, magsarefluxes=True,
                    sigclip=None, plotfit=mandelagolfit_savfile,
                    plotcorner=corner_savfile,
                    samplesavpath=samplesavpath, nworkers=nworkers,
                    n_mcmc_steps=n_mcmc_steps, eps=1e-6, n_walkers=500,
                    skipsampling=False,
                    overwriteexistingsamples=overwriteexistingsamples,
                    mcmcprogressbar=mcmcprogressbar, timeoffset=timeoffset)

    fitparamdir = "../results/tess_lightcurve_fit_parameters/"+str(ticid)
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
                    mcmcprogressbar=mcmcprogressbar, timeoffset=timeoffset)

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
    trapfit, bls_period, litparams, ticid, fit_savdir, chain_savdir, nworkers,
    n_mcmc_steps, overwriteexistingsamples, mcmcprogressbar):

    # fit only +/- n_transit_durations near the transit data. don't try to fit
    # OOT or occultation data.
    sel_inds = np.zeros_like(time).astype(bool)
    for t_start,t_end in zip(t_starts, t_ends):
        these_inds = (time > t_start) & (time < t_end)
        if np.any(these_inds):
            sel_inds |= these_inds
    sel_time = time[sel_inds]
    sel_flux = flux[sel_inds]
    sel_err_flux = err_flux[sel_inds]

    # initial guesses
    lit_period, lit_a_by_rstar, lit_incl = litparams
    bls_rp = np.sqrt(trapfit['fitinfo']['finalparams'][2])
    bls_t0 = trapfit['fitinfo']['fitepoch']
    bls_period = bls_period
    u_linear, u_quad = get_limb_darkening_initial_guesses(lcfile)

    # model = transit + line. "transit" as defined by BATMAN has flux=1 out of
    # transit. so our bounds are for a line that should pass near origin.
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

    plot_phased_mag_series(sel_time, sel_flux, lit_period, magsarefluxes=True,
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


def measure_transit_times_from_lightcurve(ticid, sectornum,
                                          n_mcmc_steps,
                                          n_phase_mcmc_steps,
                                          getspocparams=False,
                                          read_literature_params=True,
                                          overwriteexistingsamples=False,
                                          mcmcprogressbar=False,
                                          nworkers=4,
                                          chain_savdir='/home/luke/local/emcee_chains/',
                                          lcdir=None,
                                          n_transit_durations=10,
                                          verify_times=False):

    make_diagnostic_plots = True
    ##########################################

    # paths for reading and writing plots
    lcname = ('hlsp_tess-data-alerts_tess_phot_{:s}-s{:s}_tess_v1_lc.fits'.
              format(str(ticid).zfill(11), str(sectornum).zfill(2))
             )

    if not lcdir:
        raise AssertionError('input directory to find lightcurves')

    #NOTE: old. final 0121 string tied to sector number.
    # lcname = 'tess2018206045859-s{:s}-{:s}-0121-s_lc.fits.gz'.format(
    #             str(sectornum).zfill(4),str(ticid).zfill(16))

    lcfile = lcdir + lcname
    if not os.path.exists(lcfile):
        lcfiles = glob(lcdir+'*{:s}*'.format(str(ticid)))
        if len(lcfiles) == 0:
            raise AssertionError('could not find lightcurve matching ticid')
        if len(lcfiles) > 1:
            raise NotImplementedError
        if len(lcfiles) == 1:
            lcfile = lcfiles[0]

    fit_savdir = '../results/lc_analysis/'+str(ticid)
    if not os.path.exists(fit_savdir):
        os.mkdir(fit_savdir)
    fit_savdir = fit_savdir+'/'+'sector_'+str(sectornum)
    if not os.path.exists(fit_savdir):
        os.mkdir(fit_savdir)
    chain_savdir = chain_savdir+'sector_'+str(sectornum)
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

    # # METHOD #1 (outdated)
    # time, flux, err_flux, smooth_flux, whitened_flux = (
    #   get_timeseries_and_median_filter(lcfile) 
    # )
    # if make_diagnostic_plots:
    #     single_whitening_plot(time, flux, smooth_flux, whitened_flux, ticid)

    # METHOD #2:
    time, flux, err_flux, lcd = retrieve_no_whitening(
        lcfile, make_diagnostic_plots=make_diagnostic_plots)

    if verify_times:
        from verify_time_stamps import manual_verify_time_stamps
        print('\nWRN! got verify_times special mode.\n')
        manual_verify_time_stamps(lcfile, lcd)
        return 1

    # run bls to get initial parameters.
    endp = 1.05*(np.nanmax(time) - np.nanmin(time))/2
    blsdict = kbls.bls_parallel_pfind(time, flux, err_flux, magsarefluxes=True,
                                      startp=0.1, endp=endp,
                                      maxtransitduration=0.3, nworkers=8,
                                      sigclip=None)
    fitd = kbls.bls_stats_singleperiod(time, flux, err_flux,
                                       blsdict['bestperiod'],
                                       magsarefluxes=True, sigclip=None,
                                       perioddeltapercent=5)

    bls_period = fitd['period']
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
                                              magsarefluxes=True, sigclip=None,
                                              plotfit=trapfit_savfile)

    # isolate each transit to within +/- n_transit_durations
    tmids, t_starts, t_ends = (
        get_transit_times(fitd, time, n_transit_durations, trapd=trapfit)
    )

    rp = np.sqrt(trapfit['fitinfo']['finalparams'][2])

    if read_literature_params:
        # get the fixed physical parameters from the data
        litpath = (
            '../data/literature_physicalparams/{:d}/params.csv'.format(ticid)
        )
        litdf = pd.read_csv(litpath)
        # NOTE: only period is used, for now
        litparams = tuple(map(float,
            [litdf['period_day'],litdf['a_by_rstar'],litdf['inclination_deg']])
        )
    else:
        raise AssertionError

    # fit the phased transit, within N durations of the transit itself, to
    # determine a/Rstar, inclination, and quadratric terms for fixed
    # parameters. only period from literature.
    fit_abyrstar, fit_incl, fit_ulinear, fit_uquad = (
        fit_phased_transit_mandelagol_and_line(
            sectornum,
            t_starts, t_ends, time, flux, err_flux, lcfile, fitd, trapfit,
            bls_period, litparams, ticid, fit_savdir, chain_savdir, nworkers,
            n_phase_mcmc_steps, overwriteexistingsamples, mcmcprogressbar)
    )

    litparams = tuple(map(float,
            [litdf['period_day'],fit_abyrstar,fit_incl])
    )

    for transit_ix, t_start, t_end in list(
        zip(range(len(t_starts)), t_starts, t_ends)
    ):

        timeoffset = t_start + (t_end - t_start)/2
        t_start -= timeoffset
        t_end -= timeoffset
        this_time = time - timeoffset

        try:

            # # Method #1: fit only mandel-agol transit. (Outdated)
            # fit_transit_mandelagol_only(transit_ix, t_start, t_end, time,
            #                             whitened_flux, err_flux, lcfile, fitd,
            #                             trapfit, getspocparams, ticid,
            #                             fit_savdir, chain_savdir, nworkers,
            #                             n_mcmc_steps, overwriteexistingsamples,
            #                             mcmcprogressbar)

            # Method #2: fit transit + line, 4 parameters: (midtime, slope,
            # intercept, rp).
            fit_transit_mandelagol_and_line(
                sectornum,
                transit_ix, t_start, t_end, this_time, flux, err_flux,
                lcfile, fitd, trapfit, litparams, ticid, fit_savdir,
                chain_savdir, nworkers, n_mcmc_steps, overwriteexistingsamples,
                mcmcprogressbar, getspocparams, timeoffset, fit_ulinear,
                fit_uquad
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

    parser.add_argument('--chain_savdir', type=str, default=None,
        help=('e.g., /home/foo/bar/'))
    parser.add_argument('--lcdir', type=str, default=None,
        help=('e.g., /home/foo/lightcurves/'))

    args = parser.parse_args()

    measure_transit_times_from_lightcurve(
        args.ticid, args.sectornum, args.n_mcmc_steps,
        args.n_phase_mcmc_steps, getspocparams=args.spocparams,
        overwriteexistingsamples=args.overwrite,
        mcmcprogressbar=args.progressbar, nworkers=args.nworkers,
        chain_savdir=args.chain_savdir, lcdir=args.lcdir,
        n_transit_durations=args.n_transit_durations,
        read_literature_params=args.rlp,
        verify_times=args.verifytimes)
