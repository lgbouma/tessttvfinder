# -*- coding: utf-8 -*-
"""
python verify_time_stamps.py &> ../results/verify_tess_timestamps/verify_output.txt &
"""
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
import astropy.units as units

from astropy.coordinates import SkyCoord
from astropy.time import Time

from astrobase.timeutils import get_epochs_given_midtimes_and_period
from numpy import array as nparr
from glob import glob
from parse import parse, search

from scipy.optimize import curve_fit
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from scipy.stats import norm

import emcee, corner
from datetime import datetime
import os, argparse, pickle, h5py
from glob import glob
from multiprocessing import Pool

#############
## LOGGING ##
#############

import logging
from datetime import datetime
from traceback import format_exc

# setup a logger
LOGGER = None
LOGMOD = __name__
DEBUG = False

def set_logger_parent(parent_name):
    globals()['LOGGER'] = logging.getLogger('%s.%s' % (parent_name, LOGMOD))

def LOGDEBUG(message):
    if LOGGER:
        LOGGER.debug(message)
    elif DEBUG:
        print('[%s - DBUG] %s' % (
            datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            message)
        )

def LOGINFO(message):
    if LOGGER:
        LOGGER.info(message)
    else:
        print('[%s - INFO] %s' % (
            datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            message)
        )

def LOGERROR(message):
    if LOGGER:
        LOGGER.error(message)
    else:
        print('[%s - ERR!] %s' % (
            datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            message)
        )

def LOGWARNING(message):
    if LOGGER:
        LOGGER.warning(message)
    else:
        print('[%s - WRN!] %s' % (
            datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            message)
        )

def LOGEXCEPTION(message):
    if LOGGER:
        LOGGER.exception(message)
    else:
        print(
            '[%s - EXC!] %s\nexception was: %s' % (
                datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
                message, format_exc()
            )
        )


####################
# HELPER FUNCTIONS #
####################

def linear_model(xdata, m, b):
    return m*xdata + b

def _plot_absolute_O_minus_C(plname, tess_epoch, diff_seconds, tess_err_tmid,
                             err_prediction_seconds, xlim=None, ylim=None):

    plt.close("all")
    fig,ax = plt.subplots(figsize=(6,4))

    ax.errorbar(tess_epoch,
                diff_seconds,
                tess_err_tmid,
                fmt='ok', ecolor='black', zorder=2, alpha=1, mew=0)

    ax.text(0.96,0.02, '"Prediction" ephemeris is only from literature times;\n'+
            'are TESS times systematically early or late?\n'+
            'Band is $\pm 1\sigma$ error on ephemeris prediction.',
            transform=ax.transAxes, color='gray', fontsize='xx-small',
            va='bottom', ha='right')

    xlabel = 'Epoch (starting from literature ephemeris)'
    ax.set_xlabel(xlabel)
    ylabel = 'Observed - Prediction [seconds]'
    ax.set_ylabel(ylabel)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    xlim = ax.get_xlim()
    xline = np.linspace(min(xlim),max(xlim),100)
    ax.fill_between(xline,
            y1=np.zeros_like(xline)-np.mean(err_prediction_seconds),
            y2=np.zeros_like(xline)+np.mean(err_prediction_seconds),
            zorder=-2, color='gray')

    ax.set_xlim(xlim)

    ax.get_yaxis().set_tick_params(which='both', direction='in')
    ax.get_xaxis().set_tick_params(which='both', direction='in')
    fig.tight_layout(h_pad=0, w_pad=0)

    savdir = '../results/verify_tess_timestamps/'
    savname = '{:s}_absolute_OminusC_for_ephem_precision_check.png'.format(plname)
    savpath = os.path.join(savdir, savname)

    fig.tight_layout()
    fig.savefig(savpath, bbox_inches='tight', dpi=400)
    print('saved {:s}'.format(savpath))


def _plot_relative_O_minus_C(plname, tess_epoch, diff_seconds, tess_err_tmid,
                             err_prediction_seconds, xlim=None, ylim=None):

    plt.close("all")
    fig,ax = plt.subplots(figsize=(6,4))

    ax.errorbar(tess_epoch,
                diff_seconds/err_prediction_seconds,
                tess_err_tmid/err_prediction_seconds,
                fmt='ok', ecolor='black', zorder=2, alpha=1, mew=0)

    OmC_by_err = diff_seconds/err_prediction_seconds
    txt = (
    '"Prediction" ephemeris is only from literature times;\n'+
    'are TESS times systematically early or late?\n'+
    'Average (O-C)/1$\sigma$: {:.2f}\n'.format(np.mean(OmC_by_err))+
    'Average (O-C): {:.1f} seconds\n'.format(np.mean(diff_seconds))+
    '1$\sigma$ error in prediction: {:.1f} seconds\n'.format(np.mean(err_prediction_seconds))+
    'Band is $\pm 1\sigma$ error on ephemeris prediction.'
    )
    ax.text(0.96, 0.02, txt,
            transform=ax.transAxes, color='gray', fontsize='xx-small',
            va='bottom', ha='right')

    xlabel = 'Epoch (starting from literature ephemeris)'
    ax.set_xlabel(xlabel)
    ylabel = '(Observed - Prediction)/(1$\sigma$ Error in Prediction)'
    ax.set_ylabel(ylabel)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    xlim = ax.get_xlim()
    xline = np.linspace(min(xlim),max(xlim),100)
    ax.fill_between(xline,
            y1=np.zeros_like(xline)-1,
            y2=np.zeros_like(xline)+1,
            zorder=-2, color='gray')

    ax.set_xlim(xlim)

    ax.get_yaxis().set_tick_params(which='both', direction='in')
    ax.get_xaxis().set_tick_params(which='both', direction='in')
    fig.tight_layout(h_pad=0, w_pad=0)

    savdir = '../results/verify_tess_timestamps/'
    savname = '{:s}_relative_OminusC_for_ephem_precision_check.png'.format(plname)
    savpath = os.path.join(savdir, savname)

    fig.tight_layout()
    fig.savefig(savpath, bbox_inches='tight', dpi=400)
    print('saved {:s}'.format(savpath))


def _run_ks2sample_vs_gaussian(x):

    # run a two-sample KS test.
    # compare the distribution of observed times (in units of the standard
    # deviation on the prediction) to samples drawn from a normal distribution
    # with the same 

    gaussian_samples = np.random.normal(loc=0, scale=1, size=len(x))
    from scipy import stats
    D, p_value = stats.ks_2samp(x, gaussian_samples)
    print('\n')
    print('='*42)
    ks2samp_txt = (
        'D={:.2f},p={:.2e} for 2sample KS (data vs gaussian)'.
        format(D, p_value)
    )
    print(ks2samp_txt)
    print('='*42)
    print('\n')

    return ks2samp_txt


def _plot_kde_vs_gaussian_relative(x, ks2samp_txt, plname,
                                   err_prediction_seconds,
                                   manualkdebandwidth=None):

    # get the kernel bandwidth for the KDE
    if not manualkdebandwidth:
        bandwidths = 10**np.linspace(-1, 1, 100)
        params = {'bandwidth': bandwidths}
        grid = GridSearchCV(KernelDensity(kernel='gaussian'), params,
                            cv=LeaveOneOut())
        grid.fit(x[:, None])

        print('ran grid search for best kernel width.')
        bandwidth = grid.best_params_['bandwidth']
        print('got {:.3g}'.format(bandwidth))
    else:
        bandwidth = manualkdebandwidth

    # instantiate and fit the KDE model
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde.fit(x[:, None])

    # score_samples returns the log of the probability density
    x_d = np.linspace(-20, 20, num=1000)
    logprob = kde.score_samples(x_d[:, None])

    fig, ax = plt.subplots(figsize=(6,4))
    ax.fill_between(x_d, np.exp(logprob), alpha=0.5, label='KDE from data')
    ax.plot(x, np.full_like(x, -0.02), '|k', markeredgewidth=1, label='data')

    ax.plot(x_d, norm.pdf(x_d), label='gaussian, $\mu=0$, $\sigma=1$')

    sigtxt = (
        '1$\sigma$ error in prediction: {:.1f} seconds'.
        format(np.mean(err_prediction_seconds))
    )
    if not manualkdebandwidth:
        txt = (
            'leaveoneout x-validated KDE bandwidth: {:.3g}\n{:s}\n{:s}'.
            format(bandwidth, ks2samp_txt, sigtxt)
        )
    else:
        txt = (
            'manually selected KDE bandwidth: {:.3g}\n{:s}\n{:s}'.
            format(bandwidth, ks2samp_txt, sigtxt)
        )
    ax.text(0.02, 0.98, txt,
            transform=ax.transAxes, color='gray', fontsize='xx-small',
            va='top', ha='left')

    ax.set_xlabel('(Observed - Prediction)/(1$\sigma$ Error in Prediction)')

    ax.legend(loc='best', fontsize='x-small')

    ax.get_yaxis().set_tick_params(which='both', direction='in')
    ax.get_xaxis().set_tick_params(which='both', direction='in')
    fig.tight_layout(h_pad=0, w_pad=0)

    ax.set_xlim([np.mean(x)-10*np.std(x), np.mean(x)+10*np.std(x)])

    savdir = '../results/verify_tess_timestamps/'
    savname = '{:s}_kde_vs_gaussian_relative.png'.format(plname)
    savpath = os.path.join(savdir, savname)

    fig.tight_layout()
    fig.savefig(savpath, bbox_inches='tight', dpi=400)
    print('saved {:s}'.format(savpath))



def _plot_kde_vs_gaussian_absolute(x, ks2samp_txt, plname,
                                   err_prediction_seconds,
                                   manualkdebandwidth=None):

    # get the kernel bandwidth for the KDE
    if not manualkdebandwidth:
        bandwidths = 10**np.linspace(-1, 1, 100)
        params = {'bandwidth': bandwidths}
        grid = GridSearchCV(KernelDensity(kernel='gaussian'), params,
                            cv=LeaveOneOut())
        grid.fit(x[:, None])

        print('ran grid search for best kernel width.')
        bandwidth = grid.best_params_['bandwidth']
        print('got {:.3g}'.format(bandwidth))
    else:
        bandwidth = manualkdebandwidth

    # instantiate and fit the KDE model
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde.fit(x[:, None])

    # score_samples returns the log of the probability density
    meanerr = np.mean(err_prediction_seconds)
    x_d = np.linspace(-20*meanerr, 20*meanerr, num=1000)
    logprob = kde.score_samples(x_d[:, None])

    fig, ax = plt.subplots(figsize=(6,4))
    ax.fill_between(x_d, np.exp(logprob), alpha=0.5, label='KDE from data')
    ax.plot(x, np.full_like(x, 0), '|k', markeredgewidth=1, label='data')

    ax.plot(x_d, norm.pdf(x_d, loc=0, scale=meanerr),
            label='gaussian, $\mu=0$, $\sigma={:.3g} sec$'.format(meanerr))

    sigtxt = (
        '1$\sigma$ error in prediction: {:.1f} seconds'.
        format(meanerr)
    )
    if not manualkdebandwidth:
        txt = (
            'leaveoneout x-validated KDE bandwidth: {:.3g}\n{:s}\n{:s}'.
            format(bandwidth, ks2samp_txt, sigtxt)
        )
    else:
        txt = (
            'manually selected KDE bandwidth: {:.3g} seconds\n{:s}\n{:s}'.
            format(bandwidth, ks2samp_txt, sigtxt)
        )
    ax.text(0.02, 0.98, txt,
            transform=ax.transAxes, color='gray', fontsize='xx-small',
            va='top', ha='left')

    ax.set_xlabel('Observed - Prediction [seconds]')

    if plname=='WASP-18b':
        loc = 'center left'
    else:
        loc='best'
    ax.legend(loc=loc, fontsize='x-small')

    ax.get_yaxis().set_tick_params(which='both', direction='in')
    ax.get_xaxis().set_tick_params(which='both', direction='in')
    fig.tight_layout(h_pad=0, w_pad=0)

    ax.set_xlim([np.mean(x)-10*np.std(x), np.mean(x)+10*np.std(x)])

    savdir = '../results/verify_tess_timestamps/'
    savname = '{:s}_kde_vs_gaussian_absolute.png'.format(plname)
    savpath = os.path.join(savdir, savname)

    fig.tight_layout()
    fig.savefig(savpath, bbox_inches='tight', dpi=400)
    print('saved {:s}'.format(savpath))


############################################
# MANUAL SANITY-CHECK OF TIMESTAMP HEADERS #
############################################

def manual_verify_time_stamps(lcname, lcd):
    """
    The TESS lightcurve files provide start and end times in three
    distinct time systems: ${\rm JD}_{\rm UTC}$, ${\rm JD}_{\rm TDB}$, and
    ${\rm BJD}_{\rm TDB}$. \citet{urban_explanatory_2012} explain the
    differences.

    First, we verified for a few select lightcurve files that ${\rm
    JD}_{\rm UTC}$ lagged behind ${\rm JD}_{\rm TDB}$ by the expected
    $32.184 + N\,{\rm seconds}$, where $N$ is the number of leap-seconds
    since 1961. At the time of writing, $N=37$.  The offset was as
    expected.

    Then, using the ${\rm JD}_{\rm UTC}$ timestamp, we recalculated the
    barycentric correction computed by SPOC \citep[using
    the][calculator]{eastman_achieving_2010}.  We performed this
    calculation assuming that the observer was located at the Earth's
    geocenter, and using the correct direction for any given star.  This
    gave us times in ${\rm BJD}_{\rm TDB}$ that agreed with the archival
    times to within 1.7 seconds.  This offset is comparable to the
    light-travel delay time expected for TESS on its orbit from perigee of
    $\approx 20R_\oplus$ to apogee of $\approx 60R_\oplus$.

    The above calculation bounds any error in the SPOC barycentric julian
    date correction to be less than 1.7 seconds.  Since this is smaller
    than the effect of interest, we proceed to subsequent tests.
    """

    hdulist = fits.open(lcname)
    main_hdr = hdulist[0].header
    lc_hdr = hdulist[1].header
    lc = hdulist[1].data

    time = lcd['time']              # time from LC in BTJD
    timecorr = lcd['timecorr']      # barycentric correction applied

    # cf. EXP-TESS-ARC-ICD-TM-0014, sec 5.2.2:
    # "subtracting TIMECORR from TIME will give the light arrival time at the
    # spacecraft rather than on the target's center".
    sc_time = time - timecorr

    # you have two times with unambiguous JD start dates.
    tstart_btjd = main_hdr['TSTART'] # observation start time in BTJD
    tstart_tjd_tdb = tstart_btjd + 2457000 # observation start time in TJD_TDB
    tstart_jd_utc = main_hdr['DATE-OBS'] # TSTART as UTC calendar date, JD_UTC.
    t0 = Time(tstart_jd_utc, format='isot', scale='utc')

    tstop_btjd = main_hdr['TSTOP'] # observation start time in TJD
    tstop_bjd_tdb = tstop_btjd + 2457000
    tstop_jd_utc = main_hdr['DATE-END'] # TSTOP as UTC calendar date
    t1 = Time(tstop_jd_utc, format='isot', scale='utc')

    print(tstart_btjd)
    print(tstart_tjd_tdb)
    print(tstart_jd_utc)
    print('\n---- t0 JD UTC: {} ----\n'.format(t0.jd))
    print(tstop_btjd)
    print(tstop_bjd_tdb)
    print(tstop_jd_utc)
    print('\n---- t1 JD UTC: {} ----\n'.format(t1.jd))

    ra = float(main_hdr['RA_OBJ'])*u.deg
    dec = float(main_hdr['DEC_OBJ'])*u.deg
    coord = SkyCoord(ra=ra, dec=dec, frame='icrs')
    print(coord)
    print(coord.to_string('hmsdms'))

    print('\n-----now, must go to Eastman website and convert manually-----')
    print('-----http://astroutils.astronomy.ohio-state.edu/time/utc2bjd.html-----\n')
    print('After PASTING correct results into code...')
    #assert 0

    ########################################
    # PASTE IN EASTMAN RESULTS BELOW
    t0_eastman_bjd_tdb = 2458354.106698818
    t1_eastman_bjd_tdb = 2458381.518842953
    # PASTE IN EASTMAN RESULTS ABOVE
    ########################################

    # # WASP-5b times:
    # t0_eastman_bjd_tdb = 2458354.106560430
    # t1_eastman_bjd_tdb = 2458381.518901647

    # # WASP-18b times:
    # t0_eastman_bjd_tdb = 2458354.105261717
    # t1_eastman_bjd_tdb = 2458381.518164088

    # # WASP-4b times: 
    # t0_eastman_bjd_tdb = 2458354.106698818
    # t1_eastman_bjd_tdb = 2458381.518842953

    # check against the BJD start/end dates they give!
    tstart_btjd = lc_hdr['TSTART'] # observation start time in BTJD
    tstart_bjd_tdb = tstart_btjd + 2457000
    tstart_btjd_utc = lc_hdr['DATE-OBS'] # TSTART as UTC calendar date

    tstop_btjd = lc_hdr['TSTOP'] # observation end time in BTJD
    tstop_bjd_tdb = tstop_btjd + 2457000
    tstop_btjd_utc = lc_hdr['DATE-END'] # TSTOP as UTC calendar date

    print('t0 (Eastman BJD_TDB time - SPOC pipeline BJD_TDB time) [seconds]:')
    print( (tstart_bjd_tdb - t0_eastman_bjd_tdb)*24*60*60 )

    print('t1 (Eastman BJD_TDB time - SPOC pipeline BJD_TDB time) [seconds]:')
    print( (tstop_bjd_tdb - t1_eastman_bjd_tdb)*24*60*60 )

    print('\n')


##########################
# MCMC FITTING FUNCTIONS #
##########################

def log_prior(theta, priorparams):
    """
    see derivation from 2018/11/21.0 and 2018/11/21.1

    Idea is: with multiple priors, e.g., one uniform, two gaussian, they are
    multiplied together. Then, taking the log, you must sum them.
    """

    t0, period, t_sys_offset = theta
    lsfit_t0, lsfit_t0_err, lsfit_period, lsfit_period_err = priorparams

    t_sys_offset_lower = - (5*units.min).to(u.day).value
    t_sys_offset_upper = + (5*units.min).to(u.day).value

    # uniform prior t_sys_offset. Take it as toffset~U[-5 minutes, +5 minutes]
    if not t_sys_offset_lower < t_sys_offset < t_sys_offset_upper:
        return -np.inf

    # gaussian prior on t0, and period
    mu_t0, sigma_t0 = lsfit_t0, lsfit_t0_err
    mu_period, sigma_period  = lsfit_period, lsfit_period_err

    t0_prior = (
        np.log(1.0/(np.sqrt(2*np.pi)*sigma_t0))
        -
        0.5*(t0-mu_t0)**2/sigma_t0**2
    )
    period_prior = (
        np.log(1.0/(np.sqrt(2*np.pi)*sigma_period))
        -
        0.5*(period-mu_period)**2/sigma_period**2
    )

    return t0_prior + period_prior


def offset_model(epochs, t0, period, t_sys_offset):
    return t0 + period*epochs + t_sys_offset


def log_likelihood(theta, data):

    epochs, tmid_obsd, sigma_y = data
    t0, period, t_sys_offset = theta

    yM = offset_model(epochs, t0, period, t_sys_offset)

    return -0.5 * np.sum(np.log(2 * np.pi * sigma_y ** 2)
                         + (tmid_obsd - yM) ** 2 / sigma_y ** 2)


def log_posterior(theta, data, priorparams):

    theta = np.asarray(theta)

    lp = log_prior(theta, priorparams)

    if not np.isfinite(lp):
        return -np.inf

    return lp + log_likelihood(theta, data)


def single_planet_mcmc(
    res, plname, log_posterior=log_posterior, sampledir=None, n_walkers=50,
    burninpercent=0.3, n_mcmc_steps=1000, overwriteexistingsamples=True,
    nworkers=8, plotcorner=True, verbose=True, eps=1e-5):

    ( lsfit_t0, lsfit_t0_err, lsfit_period, lsfit_period_err,
      tess_epoch, tess_tmid, tess_err_tmid
    ) = res

    n_points = len(tess_epoch)
    data = np.array([tess_epoch, tess_tmid, tess_err_tmid])
    data = data.reshape((3,n_points))

    priorparams = lsfit_t0, lsfit_t0_err, lsfit_period, lsfit_period_err

    fitparamnames=['t0 (from lit)','P (from lit)','tsysoffset']

    n_dim = 3  # this determines the model

    samplesavpath = os.path.join(
        sampledir,'{:s}_timing_offset_fit.h5'.format(plname))
    backend = emcee.backends.HDFBackend(samplesavpath)
    if overwriteexistingsamples:
        LOGWARNING('erased samples previously at {:s}'.format(samplesavpath))
        backend.reset(n_walkers, n_dim)

    # if this is the first run, then start from a gaussian ball.
    # otherwise, resume from the previous samples.
    theta_initial = np.array([lsfit_t0, lsfit_period, 0])
    starting_positions = (
        theta_initial + eps*np.random.randn(n_walkers, n_dim)
    )

    isfirstrun = True
    if os.path.exists(backend.filename):
        if backend.iteration > 1:
            starting_positions = None
            isfirstrun = False

    if verbose and isfirstrun:
        LOGINFO(
            'start MCMC with {:d} dims, {:d} steps, {:d} walkers,'.format(
                n_dim, n_mcmc_steps, n_walkers
            ) + ' {:d} threads'.format(nworkers)
        )
    elif verbose and not isfirstrun:
        LOGINFO(
            'continue with {:d} dims, {:d} steps, {:d} walkers, '.format(
                n_dim, n_mcmc_steps, n_walkers
            ) + '{:d} threads'.format(nworkers)
        )

    with Pool(nworkers) as pool:
        sampler = emcee.EnsembleSampler(
            n_walkers, n_dim, log_posterior,
            args=(data, priorparams),
            pool=pool,
            backend=backend
        )
        sampler.run_mcmc(starting_positions, n_mcmc_steps,
                         progress=True)

    if verbose:
        LOGINFO(
            'ended MCMC run with {:d} steps, {:d} walkers, '.format(
                n_mcmc_steps, n_walkers
            ) + '{:d} threads'.format(nworkers)
        )

    reader = emcee.backends.HDFBackend(samplesavpath)

    n_to_discard = int(burninpercent*n_mcmc_steps)

    samples = reader.get_chain(discard=n_to_discard, flat=True)
    log_prob_samples = reader.get_log_prob(discard=n_to_discard, flat=True)
    log_prior_samples = reader.get_blobs(discard=n_to_discard, flat=True)

    # Get best-fit parameters and their 1-sigma error bars
    fit_statistics = list(
        map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
            list(zip( *np.percentile(samples, [16, 50, 84], axis=0))))
    )

    medianparams, std_perrs, std_merrs = {}, {}, {}
    for ix, k in enumerate(fitparamnames):
        medianparams[k] = fit_statistics[ix][0]
        std_perrs[k] = fit_statistics[ix][1]
        std_merrs[k] = fit_statistics[ix][2]

    x, y, sigma_y = data
    returndict = {
        'fittype':'timing_offset_fit',
        'fitinfo':{
            'initial_guess':theta_initial,
            'maxlikeparams':None,
            'medianparams':medianparams,
            'std_perrs':std_perrs,
            'std_merrs':std_merrs
        },
        'samplesavpath':samplesavpath,
        'data':{
            'epoch':x,
            'tmid_BJD':y,
            'err_tmid':sigma_y,
        },
    }

    if plotcorner:
        plotdir = '../results/verify_tess_timestamps/'
        cornersavpath = os.path.join(
            plotdir, '{:s}_corner_timing_offset_fit.png'.format(plname))

        fig = corner.corner(
            samples,
            labels=fitparamnames,
            truths=theta_initial,
            quantiles=[0.16, 0.5, 0.84], show_titles=True
        )

        fig.savefig(cornersavpath, dpi=300)
        if verbose:
            LOGINFO('saved {:s}'.format(cornersavpath))

    return returndict


def evaluate_mcmc_output(rdict, plname, pkldir, burninpct):

    pklsavpath = os.path.join(pkldir, '{:s}_timing_offset_fit.pkl'.
                             format(plname))
    with open(pklsavpath, 'wb') as f:
        pickle.dump(rdict, f, pickle.HIGHEST_PROTOCOL)
        print('saved {:s}'.format(pklsavpath))

    # ask: how strongly are various timing offsets ruled out? Ask for
    # offsets of -10, -20, -30, etc seconds.
    offsets_in_secs = np.arange(-90,-10,1)
    offset_values = offsets_in_secs/(60*60*24)

    samplesavpath = rdict['samplesavpath']
    mp = rdict['fitinfo']['medianparams']
    perrs = rdict['fitinfo']['std_perrs']
    merrs = rdict['fitinfo']['std_merrs']

    reader = emcee.backends.HDFBackend(samplesavpath)
    n_done_steps = reader.iteration
    n_to_discard = int(burninpct*n_done_steps)
    samples = reader.get_chain(discard=n_to_discard, flat=True)

    offset_idx = 2
    offset_samples = samples[:,offset_idx]
    n_samples = len(offset_samples)

    lines, fracs, offset_list = [], [], []
    for offset_value in offset_values:

        frac = (
            len(offset_samples[offset_samples < offset_value]) / n_samples
        )

        offset_in_sec = offset_value*24*60*60

        txt = ('\n{:s} --- {:.6g}% of samples have t_systematic_offset < {:d} sec'.
               format(plname, frac*100, int(np.round(offset_in_sec)))
              )

        print(txt)
        lines.append(txt)

        fracs.append(frac)
        offset_list.append(int(np.round(offset_in_sec)))

    outdf = pd.DataFrame({'percent_of_samples':np.array(fracs),
                          'have_tsys_offset_lt':np.array(offset_list)})

    txtsavpath = os.path.join(pkldir, '{:s}_timing_offset_fit_limits.txt'.
                              format(plname))
    dfsavpath = os.path.join(pkldir, '{:s}_timing_offset_fit_limits.csv'.
                              format(plname))

    with open(txtsavpath, 'w') as f:
        f.writelines(lines)
    print('wrote {:s}'.format(txtsavpath))

    outdf.to_csv(dfsavpath, index=False)
    print('wrote {:s}'.format(dfsavpath))



def precision_of_predicted_ephemeris(plname='WASP-4b',period_guess=1.33823204):
    """
    using the manually collected ephemerides, how precise is the prediction for
    when the transits are supposed to fall?

    (at the epoch of TESS observation)
    """

    manual_fpath = '../data/{:s}_manual.csv'.format(plname)
    df = pd.read_csv(manual_fpath, sep=';', comment=None)

    tmid = nparr(df['t0_BJD_TDB'])
    err_tmid = nparr(df['err_t0'])
    sel = np.isfinite(tmid) & np.isfinite(err_tmid)

    tmid = tmid[sel]
    err_tmid = err_tmid[sel]

    epoch, init_t0 = (
        get_epochs_given_midtimes_and_period(tmid, period_guess, verbose=True)
    )

    xdata = epoch
    ydata = tmid
    sigma = err_tmid

    popt, pcov = curve_fit(
        linear_model, xdata, ydata, p0=(period_guess, init_t0), sigma=sigma
    )

    lsfit_period = popt[0]
    lsfit_period_err = pcov[0,0]**0.5
    lsfit_t0 = popt[1]
    lsfit_t0_err = pcov[1,1]**0.5

    # now: what is the uncertainty on the ephemeris during the time window that
    # tess observes?
    tw = pd.read_csv('../data/tess_sector_time_windows.csv')
    knownplanet_df_files = glob('../data/kane_knownplanet_tess_overlap/'
                                'kane_knownplanets_sector*.csv')
    for knownplanet_df_file in knownplanet_df_files:
        knownplanet_df = pd.read_csv(knownplanet_df_file)
        # if planet is observed in this sector
        if np.isin(plname.split('b')[0], nparr(knownplanet_df['pl_hostname'])):

            # 0-based sector number count
            this_sec_num = (
                int(search('sector{:d}.csv', knownplanet_df_file)[0])
            )
            # 1-based sector number count
            _ = tw[tw['sector_num'] == this_sec_num+1]

            st = float(_['start_time_HJD'].iloc[0])
            et = float(_['end_time_HJD'].iloc[0])
            mt = st + (et - st)/2

            st_epoch = (st - lsfit_t0)/lsfit_period
            et_epoch = (et - lsfit_t0)/lsfit_period
            mt_epoch = int( (mt - lsfit_t0)/lsfit_period )

            # what is range of allowed tmid at the mid-observation epoch?
            tmid_mt_expected = lsfit_t0 + lsfit_period*mt_epoch
            tmid_mt_lower = (
                (lsfit_t0-lsfit_t0_err) +
                (lsfit_period-lsfit_period_err)*mt_epoch
            )
            tmid_mt_upper = (
                (lsfit_t0+lsfit_t0_err) +
                (lsfit_period+lsfit_period_err)*mt_epoch
            )

            tmid_mt_perr = (tmid_mt_upper - tmid_mt_expected)
            tmid_mt_merr = (tmid_mt_expected - tmid_mt_lower)

    print('-'*42)
    print('\n{:s}'.format(plname))
    print('using only the literature times (no TESS times)')
    print('started with period_guess {}'.format(period_guess))
    print('got')
    print('least-squares period {} +/- {}'.
          format(lsfit_period, lsfit_period_err))
    print('least-squares t0 {} +/- {}'.
          format(lsfit_t0, lsfit_t0_err))
    print('converts to')
    print('err_period: {:.2e} seconds. err_t0: {:.2e} seconds'.
          format(lsfit_period_err*24*60*60, lsfit_t0_err*24*60*60))

    print('calculated')
    print('at epoch {:d} = BJD {:.6f}'.format(mt_epoch, tmid_mt_expected))
    print('allowed range of +{:.6f} -{:.6f} days'.
          format(tmid_mt_perr, tmid_mt_merr))
    print('= +{:.6f} -{:.6f} seconds'.
          format(tmid_mt_perr*24*60*60, tmid_mt_merr*24*60*60))
    print('\n')


def make_plots_for_ephem_precision_check(plname, period_guess, xlim=None,
                                           ylim=None, manualkdebandwidth=None):

    # load in the data with ONLY the literature times. fit a linear ephemeris
    # to it.
    manual_fpath = '../data/{:s}_manual.csv'.format(plname)
    mandf = pd.read_csv(manual_fpath, sep=';', comment=None)

    tmid = nparr(mandf['t0_BJD_TDB'])
    err_tmid = nparr(mandf['err_t0'])
    sel = np.isfinite(tmid) & np.isfinite(err_tmid)

    tmid = tmid[sel]
    err_tmid = err_tmid[sel]

    epoch, init_t0 = (
        get_epochs_given_midtimes_and_period(tmid, period_guess, verbose=True)
    )

    xdata = epoch
    ydata = tmid
    sigma = err_tmid

    popt, pcov = curve_fit(
        linear_model, xdata, ydata, p0=(period_guess, init_t0), sigma=sigma
    )

    lsfit_period = popt[0]
    lsfit_period_err = pcov[0,0]**0.5
    lsfit_t0 = popt[1]
    lsfit_t0_err = pcov[1,1]**0.5

    # now get observed tess times! compare to predicted.
    sel_fpath = (
        '../data/{:s}_literature_and_TESS_times_O-C_vs_epoch_selected.csv'.
        format(plname)
    )
    seldf = pd.read_csv(sel_fpath, sep=';', comment=None)

    mytesstimes = nparr(seldf['original_reference'] == 'me')

    tess_tmid = nparr(seldf['sel_transit_times_BJD_TDB'])[mytesstimes]
    tess_err_tmid = nparr(seldf['err_sel_transit_times_BJD_TDB'])[mytesstimes]

    tess_sel = np.isfinite(tess_tmid) & np.isfinite(tess_err_tmid)
    if plname=='WASP-18b':
        tess_sel &= (tess_err_tmid*24*60 < 1)
    tess_tmid = tess_tmid[tess_sel]
    tess_err_tmid = tess_err_tmid[tess_sel]

    tess_epoch, _ = (
        get_epochs_given_midtimes_and_period(
            tess_tmid, period_guess, t0_fixed=lsfit_t0, verbose=True)
    )

    # now: calculate the uncertainty on the ephemeris during the time window that
    # tess observes, based on the literature values.
    tmid_expected = lsfit_t0 + lsfit_period*tess_epoch
    tmid_lower = (
        (lsfit_t0-lsfit_t0_err) +
        (lsfit_period-lsfit_period_err)*tess_epoch
    )
    tmid_upper = (
        (lsfit_t0+lsfit_t0_err) +
        (lsfit_period+lsfit_period_err)*tess_epoch
    )

    tmid_perr = (tmid_upper - tmid_expected)
    tmid_merr = (tmid_expected - tmid_lower)

    # difference between observed TESS time and expectation, in seconds
    diff_seconds = (tess_tmid - tmid_expected)*24*60*60
    err_prediction_seconds = np.mean([tmid_perr, tmid_merr], axis=0)*24*60*60

    OmC_by_err = diff_seconds/err_prediction_seconds
    x = OmC_by_err

    # plot the difference in absolute and relative terms on O-C diagrams.
    _plot_absolute_O_minus_C(plname, tess_epoch, diff_seconds,
                             tess_err_tmid*24*60*60, err_prediction_seconds,
                             xlim=xlim, ylim=ylim)
    _plot_relative_O_minus_C(plname, tess_epoch, diff_seconds,
                             tess_err_tmid*24*60*60, err_prediction_seconds,
                             xlim=xlim, ylim=ylim)

    # check whether the observed samples statistically differ from those drawn
    # from a normal distribution.
    ks2samp_txt = _run_ks2sample_vs_gaussian(x)

    # plot the difference between prediction and observation, estimating
    # distributions thru kernel density estimates.
    _plot_kde_vs_gaussian_relative(x, ks2samp_txt, plname,
                                   err_prediction_seconds,
                                   manualkdebandwidth=manualkdebandwidth)

    if not (manualkdebandwidth is None):
        bw = manualkdebandwidth*np.mean(err_prediction_seconds)
    else:
        bw = None
    _plot_kde_vs_gaussian_absolute(
        diff_seconds, ks2samp_txt, plname, err_prediction_seconds,
        manualkdebandwidth=bw)

    return (
        lsfit_t0, lsfit_t0_err, lsfit_period, lsfit_period_err,
        tess_epoch, tess_tmid, tess_err_tmid
    )



def print_how_precise_are_HJ_ephem_predictions():

    plnames = ['WASP-18b', 'WASP-46b', 'WASP-5b', 'WASP-4b', 'WASP-29b',
               'WASP-6b', 'WASP-45b']
    periodguesses = [0.94145299, 1.4303700, 1.6284246, 1.33823204, 3.92274,
                     3.36100208, 3.1260960]

    for plname, periodguess in zip(plnames, periodguesses):
        precision_of_predicted_ephemeris(
            plname=plname, period_guess=periodguess)


def make_OminusC_precision_checks(overwrite = 1, n_mcmc_steps = 5000):

    sampledir = '/home/luke/local/emcee_chains/'

    plnames = ['WASP-18b', 'WASP-46b', 'WASP-5b', 'WASP-4b', 'WASP-29b',
               'WASP-6b', 'WASP-45b']
    periodguesses = [0.94145299, 1.4303700, 1.6284246, 1.33823204, 3.92274,
                     3.36100208, 3.1260960]
    manualkdebandwidths = [0.3, 0.75, 0.4, 1.5, 0.4, 0.8, 0.8]

    ##########
    pkldir = '../results/verify_tess_timestamps/'
    burninpct = 0.3

    for plname, periodguess, manualkdebandwidth in zip(
        plnames, periodguesses, manualkdebandwidths
    ):

        res = make_plots_for_ephem_precision_check(plname=plname,
                                                   period_guess=periodguess,
                                                   manualkdebandwidth=manualkdebandwidth)

        if plname=='WASP-6b':
            rdict = single_planet_mcmc(res, plname, sampledir=sampledir,
                                       n_walkers=50, burninpercent=burninpct,
                                       n_mcmc_steps=40000,
                                       overwriteexistingsamples=overwrite,
                                       nworkers=16, plotcorner=True, verbose=True,
                                       eps=1e-5)

        else:
            rdict = single_planet_mcmc(res, plname, sampledir=sampledir,
                                       n_walkers=50, burninpercent=burninpct,
                                       n_mcmc_steps=n_mcmc_steps,
                                       overwriteexistingsamples=overwrite,
                                       nworkers=16, plotcorner=True, verbose=True,
                                       eps=1e-5)

        evaluate_mcmc_output(rdict, plname, pkldir, burninpct)


def given_offset_output_quote_result(desired_offset=None):

    from scipy.interpolate import interp1d
    plnames = np.sort(['WASP-18b', 'WASP-46b', 'WASP-5b', 'WASP-4b', 'WASP-29b',
               'WASP-6b', 'WASP-45b'])

    pkldir = '../results/verify_tess_timestamps/'

    dfpaths = [pkldir+plname+'_timing_offset_fit_limits.csv'
               for plname in plnames]

    for plname, dfpath in zip(plnames, dfpaths):

        df = pd.read_csv(dfpath)

        have_tsys_offset_lt = nparr(df['have_tsys_offset_lt'])
        percent_of_samples = nparr(df['percent_of_samples'])
        sel = percent_of_samples > 0

        fn = interp1d(have_tsys_offset_lt[sel],
                      percent_of_samples[sel], fill_value='extrapolate')

        if desired_offset < np.min(have_tsys_offset_lt[sel]):
            print('EXTRAPOLATED:')
            print('{:s} --- fraction {:.12f} of samples have offset < {:.2f} sec'.
                  format(plname, fn(desired_offset), desired_offset)
            )

        else:
            print('{:s} --- fraction {:.12f} of samples have offset < {:.2f} sec'.
                  format(plname, fn(desired_offset), desired_offset)
            )

def main(do_mcmc=0, overwrite=0, n_mcmc_steps=10):

    np.random.seed(42)

    if do_mcmc:
        print_how_precise_are_HJ_ephem_predictions()
        make_OminusC_precision_checks(overwrite=overwrite,
                                      n_mcmc_steps=n_mcmc_steps)

    print('-'*42)
    given_offset_output_quote_result(desired_offset=-30)
    print('-'*42)
    given_offset_output_quote_result(desired_offset=-40)
    print('-'*42)
    given_offset_output_quote_result(desired_offset=-50)
    print('-'*42)
    given_offset_output_quote_result(desired_offset=-55)
    print('-'*42)
    given_offset_output_quote_result(desired_offset=-60)
    print('-'*42)
    given_offset_output_quote_result(desired_offset=-65)
    print('-'*42)
    given_offset_output_quote_result(desired_offset=-70)
    print('-'*42)
    given_offset_output_quote_result(desired_offset=-77.68)
    print('-'*42)
    given_offset_output_quote_result(desired_offset=-81.5)

if __name__=="__main__":

    # if running. 40,000 for WASP-6b because small N statistics.
    do_mcmc = 1
    overwrite = 1
    n_mcmc_steps = 5000

    # # if evaluating
    # do_mcmc = 0
    # overwrite = 0
    # n_mcmc_steps = 10

    main(do_mcmc=do_mcmc, overwrite=overwrite,
         n_mcmc_steps=n_mcmc_steps)
