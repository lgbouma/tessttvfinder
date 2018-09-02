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

def arr(x):
    return np.array(x)

def scatter_plot_parameter_vs_epoch_etd(df, yparam, datafile, init_period,
                                        init_t0, overwrite=False,
                                        savname=None):
    '''
    args:
        df -- made by get_ETD_params
        yparam -- in ['O-C', 'Duration', 'Depth']
        datafile -- e.g., "../data/20180826_WASP-18b_ETD.txt"
    '''

    assert yparam in ['O-C', 'Duration', 'Depth']

    if not savname:
        savname = (
            '../results/' +
            datafile.split('/')[-1].split('.txt')[0]+"_"+yparam+"_vs_epoch.pdf"
        )
    if os.path.exists(savname) and overwrite==False:
        print('skipped {:s}'.format(savname))
        return 0

    f,ax = plt.subplots(figsize=(8,6))

    dq = arr(df['DQ'])
    xvals = arr(df['Epoch'])
    if yparam != 'O-C':
        yvals = arr(df[yparam])
    elif yparam == 'O-C':
        # fit a straight line (t vs. E) to all the times. then subtract the
        # best-fitting line from the data.
        tmid_HJD = arr(df['HJDmid'])
        err_tmid_HJD = arr(df['HJDmid Error'])

        sel = np.isfinite(err_tmid_HJD)
        print('{:d} transits with claimed err_tmid_HJD < 1 minute'.
              format(len(err_tmid_HJD[err_tmid_HJD*24*60 > 1.])))
        # sel &= err_tmid_HJD*24*60 > 1.
        # NOTE: if you claim sub-minute transit time measurements, i don't
        # believe it....

        xvals = arr(df['Epoch'])[sel]
        xdata = xvals
        ydata = tmid_HJD[sel]
        sigma = err_tmid_HJD[sel]

        if repr(init_t0)[:2] == '24' and repr(tmid_HJD[0])[:2] != '24':
            init_t0 -= 2400000

        from scipy.optimize import curve_fit
        def f_model(xdata, m, b):
            return m*xdata + b
        popt, pcov = curve_fit(
            f_model, xdata, ydata, p0=(init_period, init_t0), sigma=sigma)

        lsfit_period = popt[0]
        lsfit_t0 = popt[1]

        assert abs(lsfit_period - init_period) < 1e-4, (
            'least squares period should be close to given period' )

        calc_tmids = lsfit_period * arr(df['Epoch'])[sel] + lsfit_t0

        # we can now plot "O-C"
        yvals = tmid_HJD[sel] - calc_tmids

    ymin, ymax = np.nanmean(yvals)-3*np.nanstd(yvals), \
                 np.nanmean(yvals)+3*np.nanstd(yvals)

    if yparam == 'O-C':
        yerrkey = 'HJDmid Error'
        ylabel = 'O-C [d]'
        yerrs = arr(df[yerrkey])[sel]
    elif yparam == 'Duration':
        yerrkey = yparam+' Error'
        ylabel = 'Duration [min]'
        yerrs = arr(df[yerrkey])
    elif yparam == 'Depth':
        yerrkey = yparam+' Error'
        ylabel = 'Depth [mmag]'
        yerrs = arr(df[yerrkey])

    # data points
    try:
        ax.scatter(xvals, yvals, marker='o', s=100/(dq**2), zorder=1, c='red')
    except:
        import IPython; IPython.embed()
    # error bars
    ax.errorbar(xvals, yvals, yerr=yerrs,
                elinewidth=0.3, ecolor='lightgray', capsize=2, capthick=0.3,
                linewidth=0, fmt='s', ms=0, zorder=0, alpha=0.75)
    # text for epoch and planet name
    pl_name = datafile.split('_')[1]
    ax.text(.96, .96, pl_name,
            ha='right', va='top', transform=ax.transAxes, fontsize='small')

    # make vertical lines to roughly show TESS observation window function for
    # all sectors that this planet is observed in
    tw = pd.read_csv('../data/tess_sector_time_windows.csv')

    knownplanet_df_files = glob('../data/kane_knownplanets_sector*.csv')
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

                ax.axvline(x=st_epoch, c='green', alpha=0.4, lw=0.5)
                ax.axvline(x=et_epoch, c='green', alpha=0.4, lw=0.5)

                ax.fill([st_epoch, et_epoch, et_epoch, st_epoch],
                        [ymin, ymin, ymax, ymax],
                        facecolor='green', alpha=0.2)

                stxt = 'S' + str(this_sec_num+1)
                ax.text( st_epoch+(et_epoch-st_epoch)/2, ymin+1e-3, stxt,
                        fontsize='xx-small', ha='center', va='center')


    xmin, xmax = min(ax.get_xlim()), max(ax.get_xlim())
    if yparam == 'O-C':
        txt = 'M = {:.5f} + {:f} * E'.format(lsfit_t0, lsfit_period)
        ax.text(.04, .96, txt,
                ha='left', va='top', transform=ax.transAxes, fontsize='small')
        # zero line
        ax.hlines(0, xmin, xmax, alpha=0.3, zorder=-1, lw=0.5)

    ax.set_ylabel(ylabel, fontsize='small')
    ax.set_xlabel('Epoch Number ({:d} records; times are HJD)'.format(len(df)))
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

    # make the legend
    for _dq in range(1,6):
        ax.scatter([],[], c='r', s=100/(_dq**2), label='{:d}'.format(_dq))
    ax.legend(scatterpoints=1, frameon=True, labelspacing=0,
              title='data quality', loc='lower left', fontsize='xx-small')


    f.tight_layout()
    f.savefig(savname)
    print('made {:s}'.format(savname))


def scatter_plot_parameter_vs_epoch_manual(df, yparam, datafile, init_period,
                                           init_t0, overwrite=False,
                                           savname=None):
    '''
    args:
        df -- made by get_ETD_params
        yparam -- in ['O-C', 'Duration', 'Depth']
        datafile -- e.g., "../data/20180826_WASP-18b_ETD.txt"
    '''

    assert yparam in ['O-C']

    if not savname:
        savname = (
            '../results/' +
            datafile.split('/')[-1].split('.txt')[0]+"_"+yparam+"_vs_epoch.pdf"
        )
    if os.path.exists(savname) and overwrite==False:
        print('skipped {:s}'.format(savname))
        return 0

    f,ax = plt.subplots(figsize=(8,6))

    xvals = arr(df['epoch'])
    # fit a straight line (t vs. E) to all the times. then subtract the
    # best-fitting line from the data.
    tmid_HJD = arr(df['t0_BJD_TDB'])
    err_tmid_HJD = arr(df['err_t0'])

    sel = np.isfinite(err_tmid_HJD)
    print('{:d} transits with claimed err_tmid_HJD < 1 minute'.
          format(len(err_tmid_HJD[err_tmid_HJD*24*60 > 1.])))
    # sel &= err_tmid_HJD*24*60 > 1.
    # NOTE: if you claim sub-minute transit time measurements, i don't
    # believe it....

    xvals = arr(df['epoch'])[sel]
    xdata = xvals
    ydata = tmid_HJD[sel]
    sigma = err_tmid_HJD[sel]

    if repr(init_t0)[:2] == '24' and repr(tmid_HJD[0])[:2] != '24':
        init_t0 -= 2400000

    from scipy.optimize import curve_fit
    def f_model(xdata, m, b):
        return m*xdata + b
    popt, pcov = curve_fit(
        f_model, xdata, ydata, p0=(init_period, init_t0), sigma=sigma)

    lsfit_period = popt[0]
    lsfit_t0 = popt[1]

    assert abs(lsfit_period - init_period) < 1e-4, (
        'least squares period should be close to given period' )

    calc_tmids = lsfit_period * arr(df['epoch'])[sel] + lsfit_t0

    # we can now plot "O-C"
    yvals = tmid_HJD[sel] - calc_tmids

    ymin, ymax = np.nanmean(yvals)-3*np.nanstd(yvals), \
                 np.nanmean(yvals)+3*np.nanstd(yvals)

    if yparam == 'O-C':
        yerrkey = 'err_t0'
        ylabel = 'O-C [d]'
        yerrs = arr(df[yerrkey])[sel]

    # data points
    dq = 1e3*sigma
    ax.scatter(xvals, yvals, marker='o', s=1/(dq**2), zorder=1, c='red')
    # error bars
    ax.errorbar(xvals, yvals, yerr=yerrs,
                elinewidth=0.3, ecolor='lightgray', capsize=2, capthick=0.3,
                linewidth=0, fmt='s', ms=0, zorder=0, alpha=0.75)
    # text for epoch and planet name
    pl_name = datafile.split("/")[-1].split("_")[0]
    ax.text(.96, .96, pl_name,
            ha='right', va='top', transform=ax.transAxes, fontsize='small')

    # make vertical lines to roughly show TESS observation window function for
    # all sectors that this planet is observed in
    tw = pd.read_csv('../data/tess_sector_time_windows.csv')

    knownplanet_df_files = glob('../data/kane_knownplanets_sector*.csv')
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

                ax.axvline(x=st_epoch, c='green', alpha=0.4, lw=0.5)
                ax.axvline(x=et_epoch, c='green', alpha=0.4, lw=0.5)

                ax.fill([st_epoch, et_epoch, et_epoch, st_epoch],
                        [ymin, ymin, ymax, ymax],
                        facecolor='green', alpha=0.2)

                stxt = 'S' + str(this_sec_num+1)
                ax.text( st_epoch+(et_epoch-st_epoch)/2, ymin+1e-3, stxt,
                        fontsize='xx-small', ha='center', va='center')


    xmin, xmax = min(ax.get_xlim()), max(ax.get_xlim())
    if yparam == 'O-C':
        txt = 'M = {:.5f} + {:f} * E'.format(lsfit_t0, lsfit_period)
        ax.text(.04, .96, txt,
                ha='left', va='top', transform=ax.transAxes, fontsize='small')
        # zero line
        ax.hlines(0, xmin, xmax, alpha=0.3, zorder=-1, lw=0.5)

    ax.set_ylabel(ylabel, fontsize='small')
    ax.set_xlabel('Epoch Number ({:d} records; times are BJD TDB)'.format(len(df)))
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

    # make the legend
    for _dq in np.linspace(np.nanmin(dq), np.nanmax(dq), num=6):
        ax.scatter([],[], c='r', s=1/(_dq**2), label='{:.2E}'.format(_dq))
    ax.legend(scatterpoints=1, frameon=True, labelspacing=0,
              title='err t0 (days)', loc='lower left', fontsize='xx-small')


    f.tight_layout()
    f.savefig(savname)
    print('made {:s}'.format(savname))




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


def get_manual_params(manual_glob='../data/*_manual.csv',
                      etd_glob='../data/*_ETD.txt'):
    '''
    read in data manually curated from literature for transit times and errors.
    also read in data from Exoplanet Transit Database, for initial guess of
    period.  initial guess of t0 is the median time.

    returns a dict with dataframe, filename, and metadata.
    '''

    man_fnames = glob(manual_glob)
    etd_fnames = glob(etd_glob)

    d = {}
    for k in ['fname','init_period','df', 'init_t0','etd_fname']:
        d[k] = []

    # Get things with both.
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

def make_manually_curated_OminusC_plots():

    ##############################################
    # make plots based on manually-curated times #
    ##############################################
    d = get_manual_params()

    for df, fname, init_period, init_t0 in list(
        zip(d['df'], d['fname'], d['init_period'], d['init_t0'])
    ):

        yparam = 'O-C'

        savname = (
            '../results/' +
            fname.split('/')[-1].split('.csv')[0]+"_"+
            yparam + "_vs_epoch.pdf"
        )

        scatter_plot_parameter_vs_epoch_manual(df, yparam, fname, init_period,
                                               init_t0, overwrite=True,
                                               savname=savname)

if __name__ == '__main__':

    # make_all_ETD_plots()

    make_manually_curated_OminusC_plots()
