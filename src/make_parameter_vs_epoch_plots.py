# -*- coding: utf-8 -*-
'''
scatter plots to check for orbital decay.
    * O-C vs epoch
    * transit duration vs epoch
    * depth vs epoch

NB: as downloaded, the ETD .txt data files have insane encoding that everything
fails on. E.g., someone's name gets parsed to include the ";" character. (You
can just remove that characer).

usage:
    choose between make_all_ETD_plots() and make_manually_curated_OminusC_plots()
    in main. then:

    $ python make_parameter_vs_epoch_plots.py
'''
from __future__ import division, print_function

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd, numpy as np

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)

from astropy.table import Table
from astropy.io import ascii
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy import constants

from glob import glob
import os

from parse import parse, search

from astrobase.timeutils import get_epochs_given_midtimes_and_period
from ephemerides_utilities import get_half_epochs_given_occultation_times

from scipy.optimize import curve_fit

from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive

def arr(x):
    return np.array(x)

def linear_model(xdata, m, b):
    return m*xdata + b

def plot_tess_errorbar_check(
    x, O_minus_C, sigma_y, sigma_y_corrected, period, t0,
    chi2, dof, chi2_reduced, f, chi2_corrected,
    chi2_reduced_corrrected,
    savpath=os.path.join('../results/model_comparison/toy_model',
                         'data_maxlikelihood_OminusC.png'),
    xlabel='tess times, epoch', ylabel='deviation from constant period [min]'):

    xfit = np.linspace(np.min(x), np.max(x), 1000)

    fig, (a0,a1) = plt.subplots(nrows=1, ncols=2, figsize=(10,4),
                                sharey=True)

    a0.errorbar(x, O_minus_C, sigma_y, fmt='ok', ecolor='gray')
    a0.plot(xfit, np.ones_like(xfit)*0,
            label='max likelihood linear fit')

    a1.errorbar(x, O_minus_C, sigma_y_corrected, fmt='ok', ecolor='gray')
    a1.plot(xfit, np.ones_like(xfit)*0,
            label='max likelihood linear fit')

    txt0 = 'chi2: {:.2f}\ndof: {:d}\nchi2red: {:.2f}'.format(
        chi2, dof, chi2_reduced)
    txt1 = (
        'multiply errors by f={:.2f}\nchi2: {:.2f}\ndof: {:d}\nchi2red: {:.2f}'.
        format(f, chi2_corrected, dof, chi2_reduced_corrrected)
    )

    a0.text(0.98, 0.02, txt0, ha='right', va='bottom', transform=a0.transAxes)
    a1.text(0.98, 0.02, txt1, ha='right', va='bottom', transform=a1.transAxes)

    a1.legend(loc='best', fontsize='x-small')
    a0.set_xlabel(xlabel)
    a1.set_xlabel(xlabel)
    a0.set_ylabel(ylabel)

    for ax in (a0,a1):
        ax.get_yaxis().set_tick_params(which='both', direction='in')
        ax.get_xaxis().set_tick_params(which='both', direction='in')

    fig.tight_layout(h_pad=0, w_pad=0)
    fig.savefig(savpath, bbox_inches='tight', dpi=350)
    print('made {:s}'.format(savpath))


def scatter_plot_parameter_vs_epoch_manual(
    plname,
    df, yparam, datafile, init_period,
    overwrite=False, savname=None, ylim=None,
    req_precision_minutes = 10, xlim=None,
    occultationtimecsv=None,
    correcterrorbars=False):
    '''
    args:
        df -- made by get_ETD_params
        yparam -- in ['O-C', 'Duration', 'Depth']
        datafile -- e.g., "../data/20180826_WASP-18b_ETD.txt"
    '''

    assert yparam == 'O-C'

    if not savname:
        savname = (
            '../results/' +
            datafile.split('/')[-1].split('.txt')[0] +
            "_"+yparam+"_vs_epoch.pdf"
        )
    if os.path.exists(savname) and overwrite==False:
        print('skipped {:s}'.format(savname))
        return 0

    # fit a straight line (t vs. E) to all the times. then subtract the
    # best-fitting line from the data.
    # TESS midtime errors are taken as the MAXIMUM of (plus error, minus error)
    # -- see retrieve_measured_times.py.
    tmid = arr(df['t0_BJD_TDB'])
    err_tmid = arr(df['err_t0'])

    sel = np.isfinite(err_tmid) & np.isfinite(tmid)
    sel &= (err_tmid*24*60 < req_precision_minutes)

    if plname=='WASP-18b':
        badlist = arr([
        '../results/tess_lightcurve_fit_parameters/100100827/sector_2/100100827_mandelagol_and_line_fit_empiricalerrs_t005.pickle',
        '../results/tess_lightcurve_fit_parameters/100100827/sector_2/100100827_mandelagol_and_line_fit_empiricalerrs_t022.pickle',
        '../results/tess_lightcurve_fit_parameters/100100827/sector_3/100100827_mandelagol_and_line_fit_empiricalerrs_t004.pickle',
        '../results/tess_lightcurve_fit_parameters/100100827/sector_3/100100827_mandelagol_and_line_fit_empiricalerrs_t010.pickle'
        ])

        sel &= ~np.isin(df['picklepath'], badlist)

    epoch, init_t0 = get_epochs_given_midtimes_and_period(
        tmid[sel], init_period, err_t_mid = err_tmid[sel], verbose=True
    )

    # calculate epochs for occultation time measurements, so they can be used
    # in model comparison. (don't use them for determining least squares t0 or
    # period, because they are usually rattier).
    if isinstance(occultationtimecsv, str):
        occ_file = os.path.join('../data/',occultationtimecsv)
    else:
        occ_file = None
    if occ_file:
        print('\n----WRN!! WORKING WITH OCCULTATION TIMES----\n')
        # tmid = t0 + P*E
        # tocc = t0 + P*E + P/2
        occ_df = pd.read_csv(occ_file, sep=';', comment=None)

        if 'tocc_BJD_TDB_w_ltt' in occ_df:
            print('USING OCC TIMES W/ LTT ALREADY ACCOUNTED')
            t_occ_ltt_corrected = np.array(occ_df['tocc_BJD_TDB_w_ltt'])
            err_t_occ = np.array(occ_df['err_tocc'])

        else:
            t_occ_no_ltt = np.array(occ_df['tocc_BJD_TDB_no_ltt'])
            err_t_occ = np.array(occ_df['err_tocc'])

            if plname=='WASP-4b':
                semimaj = 0.0228*u.au # Petrucci+ 2013, table 3
                ltt_corr = (2*semimaj/constants.c).to(u.second)
            else:
                raise NotImplementedError('need to implement ltt correction')

            print('subtracting {:.3f} for occultation light travel time'.
                  format(ltt_corr))

            t_occ_ltt_corrected = t_occ_no_ltt - (ltt_corr.to(u.day)).value

        occ_epoch_full = (t_occ_ltt_corrected - init_t0 - init_period/2) / init_period

        occ_epoch = np.round(occ_epoch_full, 1)

        print('got occultation epochs')
        print(occ_epoch_full)

        f_occ_epochs = np.isfinite(occ_epoch)

        tocc = t_occ_ltt_corrected[f_occ_epochs]
        err_tocc = err_t_occ[f_occ_epochs]
        occ_references = np.array(occ_df['reference'])[f_occ_epochs]
        occ_whereigot = np.array(occ_df['where_I_got_time'])[f_occ_epochs]

    else:
        print('\n----NOT WORKING WITH OCCULTATION TIMES----\n')

    print('{:d} transits collected'.format(len(err_tmid)))

    print('{:d} transits SELECTED (finite & err_tmid < {:d} minute)'.
          format(len(err_tmid[sel]), req_precision_minutes))

    print('{:d} transits with claimed err_tmid < 1 minute'.
          format(len(err_tmid[err_tmid*24*60 < 1.])))

    xvals = epoch
    xdata = xvals
    ydata = tmid[sel]
    sigma = err_tmid[sel]
    sel_references = np.array(df['reference'])[sel]

    # do the TESS error bars make sense? in particular, for TESS times
    # only, is chi^2_reduced ~= 1? if not, maybe over-estimating
    # uncertainties!
    # -> correct them so that chi^2_reduced = 1...
    if np.any(np.array(df['reference']) == 'me'):

        sel_tess = sel & (np.array(df['reference']) == 'me')

        xdata_tess = epoch[sel_references == 'me']
        xdata_tess -= np.sort(xdata_tess)[int(len(xdata_tess)/2)]
        ydata_tess = tmid[sel_tess]
        sigma_tess = err_tmid[sel_tess]

        popt_tess, pcov_tess = curve_fit(
            linear_model, xdata_tess, ydata_tess,
            p0=(init_period, init_t0),
            sigma=sigma_tess
        )

        lsfit_period_tess = popt_tess[0]
        lsfit_t0_tess = popt_tess[1]

        calc_tmids_tess = lsfit_period_tess * xdata_tess + lsfit_t0_tess

        O_minus_C_tess = tmid[sel_tess] - calc_tmids_tess

        chi2 = np.sum( O_minus_C_tess**2 / sigma_tess**2 )
        n_data, n_parameters = len(xdata_tess), 2
        dof = n_data - n_parameters

        chi2_reduced = chi2/dof

        # propose the empirical correction. `f` for fudge.
        f = np.sqrt(chi2/dof)

        sigma_tess_corrected = sigma_tess * f
        chi2_corrected = np.sum(O_minus_C_tess**2 /
                                sigma_tess_corrected**2 )
        chi2_reduced_corrrected = chi2_corrected/dof

        plname = os.path.basename(savname).split("_")[0]
        tesscheckpath = (
            os.path.join('../results/manual_plus_tess_O-C_vs_epoch',
            plname+"_tess_errorbar_check.png"
        ))

        plot_tess_errorbar_check(
            xdata_tess, O_minus_C_tess*24*60, sigma_tess*24*60,
            sigma_tess_corrected*24*60,
            lsfit_period_tess, lsfit_t0_tess, chi2, dof, chi2_reduced,
            f, chi2_corrected, chi2_reduced_corrrected,
            savpath=tesscheckpath)

        # finally, (optionally) update the errors to be used in the analysis!
        # NOTE: this assumes the TESS measurements are always being appended at
        # the end! however TESS data is pretty much always the newest for this
        # project, so this is OK.
        if correcterrorbars:
            print('WRN! ERROR BARS BEFORE EMPIRICAL CORRECTION')
            print(sigma)
            sigma_not_tess = sigma[~(sel_references == 'me')]
            sigma = np.concatenate((sigma_not_tess, sigma_tess_corrected))
            print('WRN! ERROR BARS AFTER EMPIRICAL CORRECTION')
            print(sigma)
        else:
            sigma_not_tess = sigma[~(sel_references == 'me')]
            sigma = np.concatenate((sigma_not_tess, sigma_tess))

    t0_offset = int(np.round(np.nanmedian(ydata), -3))
    savdf = pd.DataFrame(
        {'sel_epoch': xdata,
         'sel_transit_times_BJD_TDB_minus_{:d}_minutes'.format(t0_offset): (
             ydata-t0_offset)*24*60,
         'sel_transit_times_BJD_TDB': ydata,
         'err_sel_transit_times_BJD_TDB': sigma,
         'err_sel_transit_times_BJD_TDB_minutes': (sigma)*24*60,
         'original_reference': np.array(df['reference'])[sel],
         'where_I_got_time': np.array(df['where_I_got_time'])[sel],
        }
    )

    savdf = savdf.sort_values(by='sel_epoch')
    savdf = savdf[['sel_epoch',
                   'sel_transit_times_BJD_TDB_minus_{:d}_minutes'.format(t0_offset),
                   'sel_transit_times_BJD_TDB',
                   'err_sel_transit_times_BJD_TDB',
                   'err_sel_transit_times_BJD_TDB_minutes',
                   'original_reference',
                   'where_I_got_time']]

    savdfpath = (
        os.path.join(
            '../data/',
            'literature_plus_TESS_times',
            os.path.basename(savname.replace('.png', '_selected.csv'))
        )
    )
    savdf.to_csv(savdfpath, sep=';', index=False)
    print('saved {:s}'.format(savdfpath))

    if occ_file:

        occ_savdf = pd.DataFrame(
            {'sel_epoch': occ_epoch[f_occ_epochs],
             'sel_occ_times_BJD_TDB_minus_{:d}_minutes'.format(t0_offset): (
                 tocc-t0_offset)*24*60,
             'sel_occ_times_BJD_TDB': tocc,
             'err_sel_occ_times_BJD_TDB': err_tocc,
             'err_sel_occ_times_BJD_TDB_minutes': (err_tocc)*24*60,
             'original_reference': occ_references,
             'where_I_got_time': occ_whereigot,
            }
        )

        occ_savdf = occ_savdf.sort_values(by='sel_epoch')
        occ_savdf = occ_savdf[['sel_epoch',
                               'sel_occ_times_BJD_TDB_minus_{:d}_minutes'.format(t0_offset),
                               'sel_occ_times_BJD_TDB',
                               'err_sel_occ_times_BJD_TDB',
                               'err_sel_occ_times_BJD_TDB_minutes',
                               'original_reference',
                               'where_I_got_time']]

        occ_savdfpath = (
            '../data/'+
            'literature_plus_TESS_times/'+
            '{:s}_occultation_times_selected.csv'.format(plname)
        )
        occ_savdf.to_csv(occ_savdfpath, sep=';', index=False)
        print('saved {:s}'.format(occ_savdfpath))

    popt, pcov = curve_fit(
        linear_model, xdata, ydata, p0=(init_period, init_t0), sigma=sigma
    )

    lsfit_period = popt[0]
    lsfit_t0 = popt[1]

    if not abs(lsfit_period - init_period) < 1e-4:
        print('WRN! least squares period is worryingly far from given period')
    if not abs(lsfit_period - init_period) < 1e-3:
        print('ERR! least squares period should be close to given period')
        import IPython; IPython.embed()
        raise AssertionError

    calc_tmids = lsfit_period * epoch + lsfit_t0

    # we can now plot "O-C"
    yvals = (tmid[sel] - calc_tmids)*24*60

    ymin, ymax = np.nanmean(yvals)-3*np.nanstd(yvals), \
                 np.nanmean(yvals)+3*np.nanstd(yvals)

    if yparam == 'O-C':
        yerrs = sigma*24*60

    plt.close('all')
    f,ax = plt.subplots(figsize=(8,6))
    # data points
    dq = 1e3*sigma
    ax.scatter(xvals, yvals, marker='o', s=1/(dq**2), zorder=1, c='red')
    # error bars
    ax.errorbar(xvals, yvals, yerr=yerrs,
                elinewidth=0.3, ecolor='lightgray', capsize=2, capthick=0.3,
                linewidth=0, fmt='s', ms=0, zorder=0, alpha=0.75)
    # text for epoch and planet name
    pl_name = datafile.split("/")[-1].split("_")[0]
    ax.text(.04, .04, pl_name,
            ha='left', va='bottom', transform=ax.transAxes, fontsize='small')

    # add upper xscale, with BJD-2450000 on top
    t_cut = 2450000
    ax_upper = ax.twiny()
    ax_upper.errorbar(tmid[sel]-t_cut, yvals, yerr=yerrs, elinewidth=0.3,
                      ecolor='lightgray', capsize=2, capthick=0.3, linewidth=0,
                      fmt='s', ms=0, zorder=0, alpha=0.)
    ax_upper.set_xlabel('BJD-{:d}'.format(t_cut))

    for a in [ax, ax_upper]:
        a.get_yaxis().set_tick_params(which='both', direction='in')
        a.get_xaxis().set_tick_params(which='both', direction='in')

    # make vertical lines to roughly show TESS observation window function for
    # all sectors that this planet is observed in
    tw = pd.read_csv('../data/tess_sector_time_windows.csv')

    knownplanet_df_files = glob('../data/kane_knownplanet_tess_overlap/'
                                'kane_knownplanets_sector*.csv')
    if yparam == 'O-C':
        for knownplanet_df_file in knownplanet_df_files:

            knownplanet_df = pd.read_csv(knownplanet_df_file)
            # if planet is observed in this sector
            if np.isin(pl_name.split('b')[0],
                       arr(knownplanet_df['pl_hostname'])):

                # 0-based sector number count
                this_sec_num = int(
                    search('sector{:d}.csv', knownplanet_df_file)[0])

                # 1-based sector number count
                _ = tw[tw['sector_num'] == this_sec_num+1]

                st = float(_['start_time_HJD'].iloc[0])
                et = float(_['end_time_HJD'].iloc[0])

                st_epoch = (st - lsfit_t0)/lsfit_period
                et_epoch = (et - lsfit_t0)/lsfit_period

                ax.axvline(x=st_epoch, c='green', alpha=0.4, lw=0.5, zorder=-3)
                ax.axvline(x=et_epoch, c='green', alpha=0.4, lw=0.5, zorder=-3)

                ax.fill([st_epoch, et_epoch, et_epoch, st_epoch],
                        [ymin, ymin, ymax, ymax],
                        facecolor='green', alpha=0.05, zorder=-4)

                stxt = 'S' + str(this_sec_num+1)
                ax.text( st_epoch+(et_epoch-st_epoch)/2, ymin+1e-3*24*60, stxt,
                        fontsize='xx-small', ha='center', va='center',
                        zorder=-2)


    xmin, xmax = min(ax.get_xlim()), max(ax.get_xlim())

    #
    # show the plotted linear ephemeris, and the zero-line
    #
    txt = 'M = {:.5f} + {:f} * E'.format(lsfit_t0, lsfit_period)
    ax.text(.04, .96, txt,
            ha='left', va='top', transform=ax.transAxes, fontsize='small')
    ax.hlines(0, xmin, xmax, alpha=0.3, zorder=-1, lw=0.5)

    ax.set_ylabel('O-C [minutes]', fontsize='x-small')
    ax.set_xlabel(
        'Epoch Number '
        '({:d} records; tmids are BJD TDB; TESS windows +/-1 day)'
        .format( len(df)), fontsize='x-small'
    )
    if xlim:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    if ylim:
        ax.set_ylim(ylim)

    # make the legend
    for _dq in np.linspace(np.nanmin(dq), np.nanmax(dq), num=6):
        ax.scatter([],[], c='r', s=1/(_dq**2), label='{:.2E}'.format(_dq))
    ax.legend(scatterpoints=1, frameon=True, labelspacing=0,
              title='err t0 [minutes]', loc='upper right', fontsize='xx-small')

    f.tight_layout()
    f.savefig(savname, bbox_inches='tight')
    print('made {:s}'.format(savname))
    f.savefig(savname.replace('.pdf','.png'), dpi=300, bbox_inches='tight')
    print('made {:s}'.format(savname.replace('.pdf','.png')))


def get_ETD_params(fglob='../data/*_ETD.txt'):
    '''
    read in data manually downloaded from Exoplanet Transit Database.

    returns a dict with dataframe, filename, and metadata.
    '''

    fnames = glob(fglob)

    d = {}
    for k in ['fname','init_period','df', 'init_t0']:
        d[k] = []

    for fname in fnames:

        # get the period reported in discovery
        with open(fname, 'r', errors='ignore') as f:
            lines = f.readlines()
        pline = [l for l in lines if 'Per =' in l]
        assert len(pline) == 1
        init_period = search('Per = {:f}', pline[0])[0]
        t0 = search('HJDmid = {:f}', pline[0])[0]

        # read in the data table
        df = pd.read_csv(fname,
                         sep=';',
                         comment=None,
                         engine='python',
                         skiprows=4
                        )

        d['fname'].append(fname)
        d['init_period'].append(init_period)
        d['df'].append(df)
        d['init_t0'].append(t0)

    return d


def make_transit_time_df(plname, manualtimecsv=None, tesstimecsv=None,
                         asastimecsv=None):
    '''
    Make a dataframe of the manually-curated transit times (if csv name
    passed), and the transit times measured from TESS lightcurves.

    Match against exoplanetarchive to get initial guess of period.  Initial
    guess of t0 is the median time.

    args:
        manualtimecsv (str): path to the manually-curated transit time csv file

        tesstimecsv (str): path to the csv of measured TESS transit times

        asastimecsv (str): path to the csv of measured ASAS transit times

    returns:
        dict with dataframe, filename, and metadata.
    '''


    d = {}
    for k in ['fname','init_period','df', 'init_t0']:
        d[k] = []

    # Using the manually curated times, match to the ETD names in order to get
    # their preliminary fit period.
    eatab = NasaExoplanetArchive.get_confirmed_planets_table()

    thispl = eatab[eatab['NAME_LOWERCASE'] == plname.lower()]

    if len(thispl) != 1:
        raise AssertionError('expected a single match from exoplanet archive')

    init_period = float(thispl['pl_orbper'].value)

    # read in the manually curated data table
    df = None
    if manualtimecsv:
        df = pd.read_csv(manualtimecsv, sep=';', comment=None)

    if tesstimecsv:
        tf = pd.read_csv(tesstimecsv)

        tf['where_I_got_time'] = (
            np.repeat('measured_from_SPOC_alert_LC', len(tf['BJD_TDB']))
        )
        tf['reference'] = np.repeat('me', len(tf['BJD_TDB']))
        tf['epoch'] = np.repeat(np.nan, len(tf['BJD_TDB']))
        tf['comment'] = np.repeat('', len(tf['BJD_TDB']))
        tf.rename(index=str,columns={'BJD_TDB':'t0_BJD_TDB',
                                     't0_bigerr':'err_t0'}, inplace=True)
        if isinstance(df, pd.DataFrame):
            df = pd.concat((df, tf),join='outer',sort=True)
        else:
            df = tf

        if isinstance(manualtimecsv, str):
            outname = manualtimecsv.replace('.csv','_and_TESS_times.csv')
        else:
            outdir = os.path.dirname(tesstimecsv)
            outname = os.path.join(
                outdir, '{}_TESS_times_only.csv'.format(plname)
            )
        df.to_csv(outname, index=False, sep=';')
        print('saved {:s}'.format(outname))

    if asastimecsv:
        # manually curated with extra ASAS time
        at = pd.read_csv(asastimecsv, sep=';', comment=None)
        df = at

    # set t0 as the median time
    t0 = np.nanmedian(df['t0_BJD_TDB'])

    csvname = manualtimecsv if isinstance(manualtimecsv,str) else tesstimecsv
    d['fname'].append(csvname)
    d['init_period'].append(init_period)
    d['df'].append(df)
    d['init_t0'].append(t0)

    return d



def make_all_ETD_plots():

    ######################################
    # make plots based on ETD data alone #
    ######################################
    d = get_ETD_params()

    for df, fname, init_period, init_t0 in list(
        zip(d['df'], d['fname'], d['init_period'], d['init_t0'])
    ):
        for yparam in ['O-C', 'Duration', 'Depth']:

            scatter_plot_parameter_vs_epoch(df, yparam, fname, init_period,
                                            init_t0, overwrite=True)

def make_manually_curated_OminusC_plots(plname, datadir='../data/',
                                        manualtimecsv=None,
                                        tesstimecsv=None,
                                        asastimecsv=None,
                                        occultationtimecsv=None,
                                        ylim=None,
                                        xlim=None,
                                        savname=None,
                                        req_precision_minutes=10,
                                        correcterrorbars=False
                                        ):
    '''
    make O-C diagrams based on manually-curated times
    '''

    if manualtimecsv:
        manual_csv = os.path.join(datadir,
                                  'manual_literature_time_concatenation',
                                  manualtimecsv)
    else:
        manual_csv = None

    if tesstimecsv:
        tesstimecsv = os.path.join(datadir,'measured_TESS_times',tesstimecsv)
    else:
        tesstimecsv = None

    if asastimecsv:
        asastimecsv = os.path.join(datadir,asastimecsv)
    else:
        asastimecsv = None

    d = make_transit_time_df(plname, manualtimecsv=manual_csv,
                             tesstimecsv=tesstimecsv,
                             asastimecsv=asastimecsv)

    for df, fname, init_period in list(
        zip(d['df'], d['fname'], d['init_period'])
    ):

        yparam = 'O-C'

        if tesstimecsv and not asastimecsv:
            savdir = '../results/manual_plus_tess_O-C_vs_epoch/'
        elif asastimecsv and not tesstimecsv:
            savdir = '../results/manual_plus_asas_O-C_vs_epoch/'
        elif asastimecsv and tesstimecsv:
            raise NotImplementedError
        else:
            savdir = '../results/manual_O-C_vs_epoch/'
        if not savname:
            savname = (
                savdir +
                fname.split('/')[-1].split('.csv')[0]+"_"+
                yparam + "_vs_epoch.pdf"
            )
        else:
            savname = savdir + savname

        planetname = os.path.basename(fname).split('_')[0]
        if 'manual_plus_tess' in savdir:
            df.to_csv(savdir+planetname+"_manual_plus_tess.csv", index=False,
                      sep=';')
            print('saved {:s}'.
                  format(savdir+planetname+"_manual_plus_tess.csv"))
        else:
            raise NotImplementedError('need smarter dataframe namesaving')

        scatter_plot_parameter_vs_epoch_manual(
            plname,
            df, yparam, fname, init_period, overwrite=True, savname=savname,
            ylim=ylim, req_precision_minutes = req_precision_minutes,
            xlim=xlim, occultationtimecsv=occultationtimecsv,
            correcterrorbars=correcterrorbars
        )

if __name__ == '__main__':

    make_all_ETD=0
    make_manually_curated=1

    occultationtimecsv = None
    asastimecsv = None
    ylim, xlim = None, None
    req_precision_minutes = 5

    # # WASP-45b
    # plname = 'WASP-45b'
    # manualtimecsv = '{:s}_manual.csv'.format(plname)
    # savname = '{:s}_literature_and_TESS_times_O-C_vs_epoch.png'.format(plname)
    # tesstimecsv = '120610833_measured_TESS_times_9_transits.csv'

    # # WASP-6b
    # plname = 'WASP-6b'
    # manualtimecsv = '{:s}_manual.csv'.format(plname)
    # savname = '{:s}_literature_and_TESS_times_O-C_vs_epoch.png'.format(plname)
    # tesstimecsv = '204376737_measured_TESS_times_8_transits.csv'

    # # WASP-29b
    # plname = 'WASP-29b'
    # manualtimecsv = '{:s}_manual.csv'.format(plname)
    # savname = '{:s}_literature_and_TESS_times_O-C_vs_epoch.png'.format(plname)
    # tesstimecsv = '183537452_measured_TESS_times_7_transits.csv'

    # # WASP-5b
    # plname = 'WASP-5b'
    # manualtimecsv = '{:s}_manual.csv'.format(plname)
    # savname = '{:s}_literature_and_TESS_times_O-C_vs_epoch.png'.format(plname)
    # tesstimecsv = '184240683_measured_TESS_times_16_transits.csv'
    # req_precision_minutes = 30

    # # WASP-4b double-check
    # plname = 'WASP-4b'
    # manualtimecsv = 'WASP-4b_manual_doublechecking.csv'
    # tesstimecsv = '402026209_measured_TESS_times_18_transits_doublechecking.csv'
    # savname = 'WASP-4b_doublechecking_TESS_times_O-C_vs_epoch.png'
    # tesstimecsv = '402026209_measured_TESS_times_20_transits.csv'
    # ylim = [-3.2,1.2]

    # # WASP-4b TESS only
    # manualtimecsv = 'WASP-4b_manual.csv'
    # tesstimecsv = '402026209_measured_TESS_times_18_transits.csv'
    # savname = 'WASP-4b_TESS_times_O-C_vs_epoch.png'
    # ylim = None # [-0.031,0.011], for WASP-18b with hipparcos times!
    # xlim = [2420,2480]

    # # WASP-4b
    # plname = 'WASP-4b'
    # manualtimecsv = '{:s}_manual.csv'.format(plname)
    # occultationtimecsv = '{:s}_manual_occultations.csv'.format(plname)
    # savname = '{:s}_literature_and_TESS_times_O-C_vs_epoch.png'.format(plname)
    # tesstimecsv = '402026209_measured_TESS_times_20_transits.csv'
    # req_precision_minutes = 2 # get a junky one otherwise!
    # ylim = [-2.5,1.5]
    # correcterrorbars = True

    # # WASP-18b just TESS times... do you see nice TTVs?
    # plname = 'WASP-18b'
    # manualtimecsv = None#'WASP-18b_manual_hipparcos_and_ASAS.csv'
    # tesstimecsv = '100100827_measured_TESS_times_48_transits.csv'
    # savname = 'WASP-18b_TESS_times_O-C_vs_epoch_badtransitsremoved.png'
    # ylim = [-2,2] # [-0.031,0.011], for WASP-18b with hipparcos times!
    # xlim = [-40,40]
    # req_precision_minutes = 30
    # correcterrorbars = False

    # # WASP-18b, with ASAS and hipparcos point.
    # manualtimecsv = 'WASP-18b_manual_hipparcos_and_ASAS.csv'
    # tesstimecsv = '100100827_measured_TESS_times_48_transits.csv'
    # savname = 'WASP-18b_all_times_O-C_vs_epoch.png'
    # ylim = [-30,10] # [-0.031,0.011], for WASP-18b with hipparcos times!
    # xlim = None
    # req_precision_minutes = 30

    # # WASP-18b, with ASAS point.
    # manualtimecsv = 'WASP-18b_manual_and_ASAS_times.csv'
    # tesstimecsv = '100100827_measured_TESS_times_48_transits.csv'
    # savname = 'WASP-18b_manual_and_ASAS_times_O-C_vs_epoch.png'
    # ylim = [-3,10] # [-0.031,0.011], for WASP-18b with hipparcos times!
    # xlim = None
    # req_precision_minutes = 10

    # # WASP-18b, no ASAS or Hipparcos point.
    # plname = 'WASP-18b'
    # manualtimecsv = '{:s}_manual_no_hipparcos.csv'.format(plname)
    # savname = '{:s}_literature_and_TESS_times_O-C_vs_epoch.png'.format(plname)
    # tesstimecsv = '100100827_measured_TESS_times_48_transits.csv'
    # asastimecsv = None # 'WASP-18b_manual_and_ASAS_times.csv'
    # ylim = [-2,2] # [-0.031,0.011], for WASP-18b with hipparcos times!
    # xlim = None
    # req_precision_minutes = 10
    # correcterrorbars = False

    # # WASP-18b, no ASAS or Hipparcos point, but with occultations (Avi final).
    # plname = 'WASP-18b'
    # manualtimecsv = '{:s}_manual_no_hipparcos.csv'.format(plname)
    # occultationtimecsv = '{:s}_manual_occultations.csv'.format(plname)
    # savname = '{:s}_literature_and_TESS_times_O-C_vs_epoch.png'.format(plname)
    # tesstimecsv = '100100827_measured_TESS_times_48_transits.csv'
    # asastimecsv = None # 'WASP-18b_manual_and_ASAS_times.csv'
    # ylim = [-2,2] # [-0.031,0.011], for WASP-18b with hipparcos times!
    # xlim = None
    # req_precision_minutes = 10
    # correcterrorbars = False

    # # WASP-46b
    # plname = 'WASP-46b'
    # manualtimecsv = '{:s}_manual.csv'.format(plname)
    # savname = '{:s}_literature_and_TESS_times_O-C_vs_epoch.png'.format(plname)
    # tesstimecsv = '231663901_measured_TESS_times_18_transits.csv'

    # # WASP-121b
    # plname = 'WASP-121b'
    # manualtimecsv = '{:s}_manual.csv'.format(plname)
    # #occultationtimecsv = '{:s}_manual_occultations.csv'.format(plname)
    # savname = '{:s}_literature_and_TESS_times_O-C_vs_epoch.png'.format(plname)
    # tesstimecsv = '22529346_measured_TESS_times_18_transits.csv'
    # req_precision_minutes = 30 # get a junky one otherwise!
    # ylim = [-30,30]
    # correcterrorbars = True

    # # WASP-19b
    # plname = 'WASP-19b'
    # manualtimecsv = '{:s}_manual.csv'.format(plname)
    # #occultationtimecsv = '{:s}_manual_occultations.csv'.format(plname)
    # savname = '{:s}_literature_and_TESS_times_O-C_vs_epoch.png'.format(plname)
    # tesstimecsv = '35516889_measured_TESS_times_29_transits.csv'
    # req_precision_minutes = 30 # get a junky one otherwise!
    # ylim = [-3,3]
    # correcterrorbars = True

    # COROT-1b
    plname = 'CoRoT-1b'
    manualtimecsv = '{:s}_manual.csv'.format(plname)
    #occultationtimecsv = '{:s}_manual_occultations.csv'.format(plname)
    savname = '{:s}_literature_and_TESS_times_O-C_vs_epoch.png'.format(plname)
    tesstimecsv = '36352297_measured_TESS_times_14_transits.csv'
    req_precision_minutes = 30 # get a junky one otherwise!
    ylim = [-5,5]
    correcterrorbars = True


    if make_all_ETD:
        make_all_ETD_plots()

    if make_manually_curated:
        make_manually_curated_OminusC_plots(
            plname,
            manualtimecsv=manualtimecsv,
            tesstimecsv=tesstimecsv,
            asastimecsv=asastimecsv,
            occultationtimecsv=occultationtimecsv,
            ylim=ylim,
            xlim=xlim,
            savname=savname,
            req_precision_minutes=req_precision_minutes,
            correcterrorbars=correcterrorbars
        )
