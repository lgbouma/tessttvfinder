# -*- coding: utf-8 -*-
'''
scatter plots to check for orbital decay.
    * O-C vs epoch
    * transit duration vs epoch
    * depth vs epoch

NB: as downloaded, the ETD .txt data files have insane encoding that everything
fails on. E.g., someone's name gets parsed to include the ";" character. (You
can just remove that characer).
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

def scatter_plot_parameter_vs_epoch(df, yparam, datafile, init_period, t0,
                                    overwrite=False):
    '''
    args:
        df -- made by get_ETD_params
        yparam -- in ['O-C', 'Duration', 'Depth']
        datafile -- e.g., "../data/20180826_WASP-18b_ETD.txt"
    '''

    assert yparam in ['O-C', 'Duration', 'Depth']

    savname = (
        '../results/' +
        datafile.split('/')[-1].split('.txt')[0]+"_"+yparam+"_vs_epoch.pdf"
    )
    if os.path.exists(savname) and overwrite==False:
        print('skipped {:s}'.format(savname))
        return 0

    f,ax = plt.subplots(figsize=(8,6))

    xvals = arr(df['Epoch'])
    yvals = arr(df[yparam])
    dq = arr(df['DQ'])

    ymin, ymax = np.nanmean(yvals)-3*np.nanstd(yvals), \
                 np.nanmean(yvals)+3*np.nanstd(yvals)

    if yparam == 'O-C':
        yerrkey = 'HJDmid Error'
        ylabel = 'O-C [d]'
    elif yparam == 'Duration':
        yerrkey = yparam+' Error'
        ylabel = 'Duration [min]'
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

                st_epoch = (st - t0)/init_period
                et_epoch = (et - t0)/init_period

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
        txt = 'M = {:.5f} + {:f} * E'.format(t0, init_period)
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


def get_ETD_params(fglob='../data/*_ETD.txt'):
    '''
    read in data manually downloaded from Exoplanet Transit Database.

    returns a dict with dataframe, filename, and metadata.
    '''

    fnames = glob(fglob)

    d = {}
    for k in ['fname','init_period','df', 't0']:
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
        d['t0'].append(t0)

    return d


if __name__ == '__main__':

    d = get_ETD_params()

    for df, fname, init_period, t0 in list(
        zip(d['df'], d['fname'], d['init_period'], d['t0'])
    ):
        for yparam in ['O-C', 'Duration', 'Depth']:

            scatter_plot_parameter_vs_epoch(df, yparam, fname, init_period,
                                            t0, overwrite=True)
