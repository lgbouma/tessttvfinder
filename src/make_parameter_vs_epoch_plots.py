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

from glob import glob
import os

from parse import parse, search

from ephemerides_utilities import get_ROUGH_epochs_given_midtime_and_period, \
    get_half_epochs_given_occultation_times

def arr(x):
    return np.array(x)

def f_model(xdata, m, b):
    return m*xdata + b

def scatter_plot_parameter_vs_epoch_manual(
    df, yparam, datafile, init_period,
    overwrite=False, savname=None, ylim=None
):
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

    f,ax = plt.subplots(figsize=(8,6))

    # fit a straight line (t vs. E) to all the times. then subtract the
    # best-fitting line from the data.
    tmid = arr(df['t0_BJD_TDB'])
    err_tmid = arr(df['err_t0'])
    epoch, init_t0 = (
        get_ROUGH_epochs_given_midtime_and_period(tmid, init_period)
    )

    # include occultation time measurements in fitting for period and t0
    try:
        if np.any(arr(df['tsec_BJD_TDB'])):

            tsec = arr(df['tsec_BJD_TDB'])
            err_tsec = arr(df['err_tsec'])

            sec_epochs = get_half_epochs_given_occultation_times(
                tsec, init_period, init_t0)

            f_sec_epochs = np.isfinite(sec_epochs)

            tmid = np.concatenate((tmid, tsec[f_sec_epochs]))
            err_tmid = np.concatenate((err_tmid, err_tsec[f_sec_epochs]))
            epoch = np.concatenate((epoch, sec_epochs[f_sec_epochs]))

    except KeyError:
        pass

    sel = np.isfinite(err_tmid) & np.isfinite(tmid)
    sel &= (err_tmid*24*60 < 10) # sub-5 minute precision!

    print('{:d} transits collected'.format(len(err_tmid)))

    print('{:d} transits SELECTED (err_tmid < 10 minute)'.
          format(len(err_tmid[err_tmid*24*60 < 10.])))

    print('{:d} transits with claimed err_tmid < 1 minute'.
          format(len(err_tmid[err_tmid*24*60 < 1.])))

    xvals = epoch[sel]
    xdata = xvals
    ydata = tmid[sel]
    sigma = err_tmid[sel]

    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(
        f_model, xdata, ydata, p0=(init_period, init_t0), sigma=sigma
    )

    lsfit_period = popt[0]
    lsfit_t0 = popt[1]

    if not abs(lsfit_period - init_period) < 1e-4:
        print('WRN! least squares period is worryingly far from given period')
    if not abs(lsfit_period - init_period) < 1e-3:
        print('ERR! least squares period should be close to given period')
        import IPython; IPython.embed()
        raise AssertionError

    calc_tmids = lsfit_period * epoch[sel] + lsfit_t0

    # we can now plot "O-C"
    yvals = (tmid[sel] - calc_tmids)*24*60

    ymin, ymax = np.nanmean(yvals)-3*np.nanstd(yvals), \
                 np.nanmean(yvals)+3*np.nanstd(yvals)

    if yparam == 'O-C':
        yerrs = sigma*24*60

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
                        facecolor='green', alpha=0.2, zorder=-4)

                stxt = 'S' + str(this_sec_num+1)
                ax.text( st_epoch+(et_epoch-st_epoch)/2, ymin+1e-3, stxt,
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
                         delimiter=';',
                         comment=None,
                         engine='python',
                         skiprows=4
                        )

        d['fname'].append(fname)
        d['init_period'].append(init_period)
        d['df'].append(df)
        d['init_t0'].append(t0)

    return d


def get_manual_and_TESS_ttimes(manualtimeglob='../data/*_manual.csv',
                               etd_glob='../data/*_ETD.txt',
                               tesstimecsv=None,
                               asastimecsv=None):
    '''
    Make a dataframe of both the manually-curated transit times, and the
    transit times measured from TESS lightcurves.

    Match against ETD to get initiali guess of period.  Initial guess of t0 is
    the median time.

    args:
        manualtimeglob (str): pattern to the manually-curated transit time csv
        file

        tesstimecsv (str): path to the csv of measured TESS transit times

        asastimecsv (str): path to the csv of measured ASAS transit times

    returns:
        dict with dataframe, filename, and metadata.
    '''

    man_fnames = glob(manualtimeglob)
    etd_fnames = glob(etd_glob)

    d = {}
    for k in ['fname','init_period','df', 'init_t0','etd_fname']:
        d[k] = []

    # Using the manually curated times, match to the ETD names in order to get
    # their preliminary fit period.
    man_pl_names = [fname.split("/")[-1].split("_")[0] for fname in man_fnames]
    etd_pl_names = [etd_fname.split("_")[1] for etd_fname in etd_fnames]

    gd_man_names = np.isin(man_pl_names, etd_pl_names)
    gd_etd_names = np.isin(etd_pl_names, man_pl_names)

    int_man_fnames = np.sort(arr(man_fnames)[gd_man_names])
    int_etd_fnames = np.sort(arr(etd_fnames)[gd_etd_names])

    for man_fname, etd_fname in list(zip(int_man_fnames, int_etd_fnames)):

        # get the period reported in ETD
        with open(etd_fname, 'r', errors='ignore') as f:
            lines = f.readlines()
        pline = [l for l in lines if 'Per =' in l]
        assert len(pline) == 1
        init_period = search('Per = {:f}', pline[0])[0]

        # read in the manually curated data table
        df = pd.read_csv(man_fname, delimiter=';', comment=None)

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
            df = pd.concat((df, tf),join='outer')

            outname = man_fname.replace('.csv','_and_TESS_times.csv')
            df.to_csv(outname, index=False)
            print('saved {:s}'.format(outname))

        if asastimecsv:
            # manually curated with extra ASAS time
            at = pd.read_csv(asastimecsv, delimiter=';', comment=None)
            df = at

        # set t0 as the median time
        t0 = np.nanmedian(df['t0_BJD_TDB'])

        d['fname'].append(man_fname)
        d['etd_fname'].append(etd_fname)
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

def make_manually_curated_OminusC_plots(datadir='../data/',
                                        manualtimeglob=None,
                                        tesstimeglob=None,
                                        asastimeglob=None,
                                        ylim=None):
    '''
    make O-C diagrams based on manually-curated times
    '''

    if manualtimeglob:
        manual_csv = datadir+manualtimeglob
    else:
        manual_csv = None

    if tesstimeglob:
        tesstimecsv = datadir+tesstimeglob
    else:
        tesstimecsv = None

    if asastimeglob:
        asastimecsv = datadir+asastimeglob
    else:
        asastimecsv = None

    d = get_manual_and_TESS_ttimes(manualtimeglob=manual_csv,
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
        savname = (
            savdir +
            fname.split('/')[-1].split('.csv')[0]+"_"+
            yparam + "_vs_epoch.pdf"
        )

        planetname = os.path.basename(fname).split('_')[0]
        if 'manual_plus_tess' in savdir:
            df.to_csv(savdir+planetname+"_manual_plus_tess.csv", index=False)
            print('saved {:s}'.
                  format(savdir+planetname+"_manual_plus_tess.csv"))
        else:
            raise NotImplementedError('need smarter dataframe namesaving')

        scatter_plot_parameter_vs_epoch_manual(
            df, yparam, fname, init_period,
            overwrite=True, savname=savname, ylim=ylim
        )

if __name__ == '__main__':

    make_all_ETD=0
    make_manually_curated=1

    manualtimeglob = 'WASP-18b_manual_and_ASAS_times.csv'
    tesstimeglob = '100100827_measured_TESS_times_29_transits.csv'
    asastimeglob = None # 'WASP-18b_manual_and_ASAS_times.csv'

    ylim = None # [-0.031,0.011], for WASP-18b with hipparcos times!

    if make_all_ETD:
        make_all_ETD_plots()

    if make_manually_curated:
        make_manually_curated_OminusC_plots(
            manualtimeglob=manualtimeglob,
            tesstimeglob=tesstimeglob,
            asastimeglob=asastimeglob
        )
