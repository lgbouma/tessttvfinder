# -*- coding: utf-8 -*-
'''
make O-C plot with an error band for how precise the predicted times were.
'''
from __future__ import division, print_function

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt, pandas as pd, numpy as np

from glob import glob
from shutil import copyfile
import os, pickle, json

from astrobase.timeutils import get_epochs_given_midtimes_and_period
from numpy import array as nparr
from parse import parse, search

from scipy.optimize import curve_fit
from scipy.stats import norm

from astropy.coordinates import SkyCoord
from astropy import units as u


def linear_model(xdata, m, b):
    # m: period
    # b: t0
    return m*xdata + b


def calculate_timing_accuracy(plname, period_guess):
    """
    First, load in the data with ONLY the literature times. Using period_guess,
    get_epochs_given_midtimes_and_period.

    Fit a linear ephemeris to these epochs.

    Calculate the uncertainty on the ephemeris during the time window that tess
    observes, based on the literature values.

    Also calculate difference between observed TESS time and expectation, in
    seconds

    Returns:

        tuple of:
            lsfit_t0, lsfit_t0_err, lsfit_period, lsfit_period_err,
            epoch, tmid, err_tmid,
            tess_epoch, tess_tmid, tess_err_tmid, diff_seconds,
            err_prediction_seconds
    """

    manual_fpath = os.path.join(
        '/home/luke/Dropbox/proj/tessorbitaldecay/data',
        'manual_literature_time_concatenation',
        '{:s}_manual.csv'.format(plname)
    )
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

    # now get observed tess times, and compare to predicted.
    sel_fpath = os.path.join(
        '/home/luke/Dropbox/proj/tessorbitaldecay/data',
        'literature_plus_TESS_times',
        '{:s}_literature_and_TESS_times_O-C_vs_epoch_selected.csv'.
        format(plname)
    )
    seldf = pd.read_csv(sel_fpath, sep=';', comment=None)

    mytesstimes = nparr(seldf['original_reference'] == 'me')

    tess_tmid = nparr(seldf['sel_transit_times_BJD_TDB'])[mytesstimes]
    tess_err_tmid = nparr(seldf['err_sel_transit_times_BJD_TDB'])[mytesstimes]

    tess_sel = np.isfinite(tess_tmid) & np.isfinite(tess_err_tmid)
    tess_tmid = tess_tmid[tess_sel]
    tess_err_tmid = tess_err_tmid[tess_sel]

    tess_epoch, _ = (
        get_epochs_given_midtimes_and_period(
            tess_tmid, period_guess, t0_fixed=lsfit_t0, verbose=True)
    )

    # now: calculate the uncertainty on the ephemeris during the time window that
    # tess observes, based on the literature values.
    tmid_expected = lsfit_t0 + lsfit_period*tess_epoch
    tmid_upper = np.maximum(
        (lsfit_t0+lsfit_t0_err) + tess_epoch*(lsfit_period+lsfit_period_err),
        (lsfit_t0+lsfit_t0_err) + tess_epoch*(lsfit_period-lsfit_period_err)
    )
    tmid_lower = np.minimum(
        (lsfit_t0-lsfit_t0_err) + tess_epoch*(lsfit_period-lsfit_period_err),
        (lsfit_t0-lsfit_t0_err) + tess_epoch*(lsfit_period+lsfit_period_err)
    )


    tmid_perr = (tmid_upper - tmid_expected)
    tmid_merr = (tmid_expected - tmid_lower)

    # difference between observed TESS time and expectation, in seconds
    diff_seconds = (tess_tmid - tmid_expected)*24*60*60
    err_prediction_seconds = np.mean([tmid_perr, tmid_merr], axis=0)*24*60*60

    return (
        lsfit_t0, lsfit_t0_err, lsfit_period, lsfit_period_err,
        epoch, tmid, err_tmid,
        tess_epoch, tess_tmid, tess_err_tmid, diff_seconds,
        err_prediction_seconds
    )






def plot_O_minus_C_with_err_band(plname, period_guess, xlim=None, ylim=None,
                                 savpath=None, ylim1=None):

    # get difference between observed TESS time and expectation, in seconds
    (lsfit_t0, lsfit_t0_err, lsfit_period, lsfit_period_err,
    epoch, tmid, err_tmid,
    tess_epoch, tess_tmid, tess_err_tmid, diff_seconds,
    _ ) = (
        calculate_timing_accuracy(
            plname=plname, period_guess=period_guess)
    )

    model_epoch = np.arange(min(epoch)-1000,max(tess_epoch)+1000,1)
    model_tmid = lsfit_t0 + model_epoch*lsfit_period

    model_tmid_upper = np.maximum(
        (lsfit_t0+lsfit_t0_err) + model_epoch*(lsfit_period+lsfit_period_err),
        (lsfit_t0+lsfit_t0_err) + model_epoch*(lsfit_period-lsfit_period_err)
    )
    model_tmid_lower = np.minimum(
        (lsfit_t0-lsfit_t0_err) + model_epoch*(lsfit_period-lsfit_period_err),
        (lsfit_t0-lsfit_t0_err) + model_epoch*(lsfit_period+lsfit_period_err)
    )

    # make the plot
    fig,(a0,a1) = plt.subplots(nrows=2,ncols=1,figsize=(4,7))

    # all transits axis
    _ix = 0
    for e, tm, err in zip(epoch,tmid,err_tmid):
        if _ix == 0:
            alpha = np.minimum(1-(err/np.max(err_tmid))**(1/2) + 0.1,
                               np.ones_like(err))
            a0.errorbar(e,
                        nparr(tm - linear_model(
                            e, lsfit_period, lsfit_t0))*24*60,
                        err*24*60,
                        fmt='.k', ecolor='black', zorder=10, mew=0,
                        ms=6,
                        elinewidth=1,
                        alpha=alpha,
                        label='pre-TESS')
            _ix += 1
        else:
            alpha = np.minimum(1-(err/np.max(err_tmid))**(1/2) + 0.1,
                               np.ones_like(err))
            a0.errorbar(e,
                        nparr(tm - linear_model(
                            e, lsfit_period, lsfit_t0))*24*60,
                        err*24*60,
                        fmt='.k', ecolor='black', zorder=10, mew=0,
                        ms=7,
                        elinewidth=1,
                        alpha=alpha
                       )

    for ax in [a1]:
        ax.errorbar(tess_epoch,
                    nparr(tess_tmid -
                          linear_model(tess_epoch, lsfit_period, lsfit_t0))*24*60,
                    tess_err_tmid*24*60,
                    fmt='sk', ecolor='black', zorder=9, alpha=1, mew=1,
                    ms=3,
                    elinewidth=1,
                    label='TESS')

    # for the legend
    # a0.errorbar(9001, 9001, np.mean(tess_err_tmid*24*60),
    #             fmt='sk', ecolor='black', zorder=9, alpha=1, mew=1, ms=3,
    #             elinewidth=1, label='TESS')


    bin_tess_y = np.average(nparr(
        tess_tmid-linear_model(tess_epoch, lsfit_period, lsfit_t0)),
        weights=1/tess_err_tmid**2
    )
    bin_tess_err_tmid = (
        np.std(tess_tmid-linear_model(tess_epoch, lsfit_period, lsfit_t0))
        /
        (len(tess_tmid)-1)**(1/2)
    )

    tess_yval = nparr(tess_tmid -
                      linear_model(tess_epoch, lsfit_period, lsfit_t0))*24*60
    print('bin_tess_y (min)'.format(bin_tess_y))
    print('bin_tess_y (sec)'.format(bin_tess_y*60))
    print('std (min) {}'.format(np.std(tess_yval)))
    print('std (sec): {}'.format(np.std(tess_yval)*60))
    print('error measurement (plotted, min): {}'.format(bin_tess_err_tmid*24*60))
    print('error measurement (plotted, sec): {}'.format(bin_tess_err_tmid*24*60*60))
    bin_tess_x = np.median(tess_epoch)

    for ax in [a0]:
        ax.errorbar(bin_tess_x, bin_tess_y*24*60, bin_tess_err_tmid*24*60,
                    alpha=1, zorder=11, label='binned TESS',
                    fmt='s', mfc='firebrick', elinewidth=1,
                    ms=3,
                    mec='firebrick',mew=1,
                    ecolor='firebrick')

    yupper = (
        model_tmid_upper -
        linear_model(model_epoch, lsfit_period, lsfit_t0)
    )
    ylower = (
        model_tmid_lower -
        linear_model(model_epoch, lsfit_period, lsfit_t0)
    )
    err_pred = ( (
        yupper[np.argmin(np.abs(model_epoch-bin_tess_x))]
        -
        ylower[np.argmin(np.abs(model_epoch-bin_tess_x))])
        /2
    )

    print('error prediction (min): {}'.format(err_pred*24*60))
    print('error prediction (sec): {}'.format(err_pred*24*60*60))

    print('in abstract: arrived {:.2f} +/- {:.2f} sec early'.
          format(bin_tess_y*24*60*60, ((err_pred*24*60*60)**2 +
                                       (bin_tess_err_tmid*24*60*60)**2)**(1/2))
    )

    for ax in (a0,a1):
        ax.plot(model_epoch, yupper*24*60, color='#1f77b4', zorder=-1, lw=0.5)
        ax.plot(model_epoch, ylower*24*60, color='#1f77b4', zorder=-1, lw=0.5)
    a0.fill_between(model_epoch, ylower*24*60, yupper*24*60, alpha=0.3,
                    color='#1f77b4', zorder=-2, linewidth=0)
    a1.fill_between(model_epoch, ylower*24*60, yupper*24*60, alpha=0.3,
                    label='pre-TESS ephemeris', color='#1f77b4', zorder=-2,
                    linewidth=0)

    bin_yupper = np.ones_like(model_epoch)*( bin_tess_y*24*60 +
                                            bin_tess_err_tmid*24*60 )
    bin_ylower = np.ones_like(model_epoch)*( bin_tess_y*24*60 -
                                            bin_tess_err_tmid*24*60 )
    a1.plot(model_epoch, bin_yupper, color='firebrick', zorder=-1, lw=0.5)
    a1.plot(model_epoch, bin_ylower, color='firebrick', zorder=-1, lw=0.5)
    a1.fill_between(model_epoch, bin_ylower, bin_yupper, alpha=0.3,
                    color='firebrick', zorder=-2, linewidth=0,
                    label='binned TESS')

    a0.legend(loc='upper right', fontsize='xx-small')
    a1.legend(loc='upper right', fontsize='xx-small')
    a1.set_xlabel('Epoch')
    fig.text(0.,0.5, 'Deviation from predicted transit time [minutes]',
             va='center', rotation=90)
    if xlim:
        a0.set_xlim(xlim)
    else:
        a0.set_xlim((min(epoch)-100,max(tess_epoch)+100))
    if ylim:
        a0.set_ylim(ylim)
    if ylim1:
        a1.set_ylim(ylim1)
    a1.set_xlim((np.floor(bin_tess_x-1.1*len(tess_epoch)/2),
                 np.ceil(bin_tess_x+1.1*len(tess_epoch)/2)))

    a0.text(0.03,0.97,'All transits',ha='left',
            va='top',fontsize='medium',transform=a0.transAxes)
    a1.text(0.03,0.97,'TESS transits',ha='left',
            va='top',fontsize='medium',transform=a1.transAxes)

    for ax in (a0,a1):
        ax.get_yaxis().set_tick_params(which='both', direction='in')
        ax.get_xaxis().set_tick_params(which='both', direction='in')

    fig.tight_layout(h_pad=0.05, w_pad=0)

    fig.savefig(savpath, bbox_inches='tight', dpi=400)
    print('saved {:s}'.format(savpath))
    savpath = savpath.replace('.png','.pdf')
    fig.savefig(savpath, bbox_inches='tight')
    print('saved {:s}'.format(savpath))


def _get_period_guess_given_plname(plname):

    from astroquery.mast import Catalogs

    res = Catalogs.query_object(plname, catalog="TIC", radius=0.5*1/3600)

    if len(res) != 1:
        raise ValueError('for {}, got result:\n{}'.format(plname, repr(res)))

    ticid = int(res["ID"])
    litdir = '../data/literature_physicalparams/{}/'.format(ticid)
    if not os.path.exists(litdir):
        os.mkdir(litdir)
    litpath = os.path.join(litdir,'params.csv')

    try:
        lpdf = pd.read_csv(litpath)
        period_guess = float(lpdf['period_day'])

    except FileNotFoundError:

        from astrobase.services.mast import tic_objectsearch

        ticres = tic_objectsearch(ticid)

        with open(ticres['cachefname'], 'r') as json_file:
            data = json.load(json_file)

        ra = data['data'][0]['ra']
        dec = data['data'][0]['dec']

        targetcoordstr = '{} {}'.format(ra, dec)

        # attempt to get physical parameters of planet -- period, a/Rstar, and
        # inclination -- for the initial guesses.
        from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
        eatab = NasaExoplanetArchive.get_confirmed_planets_table()

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
        a_by_rstar = (semimaj_au / rstar).cgs.value

        litdf = pd.DataFrame(
            {'period_day':period,
             'a_by_rstar':a_by_rstar,
             'inclination_deg':incl
            }, index=[0]
        )
        # get the fixed physical parameters from the data. period_day,
        # a_by_rstar, and inclination_deg are comma-separated in this file.
        litdf.to_csv(litpath, index=False, header=True, sep=',')
        lpdf = pd.read_csv(litpath, sep=',')
        period_guess = float(lpdf['period_day'])

    return period_guess


if __name__=="__main__":

    plnames = ['WASP-4b',
               'WASP-5b',
               'WASP-6b',
               'WASP-18b',
               'WASP-45b',
               'WASP-46b',
               'WASP-19b',
               'WASP-121b',
               'CoRoT-1b'
              ]

    for plname in plnames:

        outpath = '../results/O_minus_C_with_err_band/{}.png'.format(plname)

        period_guess = _get_period_guess_given_plname(plname)

        xlim = None
        ylim = None
        ylim1 = None

        plot_O_minus_C_with_err_band(plname, period_guess, xlim=xlim,
                                     ylim=ylim, savpath=outpath, ylim1=ylim1)
