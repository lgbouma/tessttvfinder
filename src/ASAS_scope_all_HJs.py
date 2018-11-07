# -*- coding: utf-8 -*-
'''
For WASP-18b, we got a cute ~7 sigma precovery from ~2000-2010 ASAS-3 V band
data.

WASP-18b has a 1.2% transit depth, at V=9.3.

How many HJs are amenable to such a measurement?

(NOTE: this is using TEPCAT's list of well-studied HJs. TESS will probably
provide some new V<10, P<10days, depth>1% HJs over its first year.
'''
from __future__ import division, print_function

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import numpy as np, pandas as pd
from numpy import array as arr

from astropy.io import ascii
from astropy.io import fits
from astropy import units as u, constants as const
from astropy.table import Table

if __name__=="__main__":

    Vmagcut = 13

    tepcatpath = '../data/TEPCAT_observables.csv'

    df = pd.read_csv(tepcatpath, delimiter= ' *, *', engine='python')

    # say we're interested in depths down to 1%
    sel = (df['depth'] >= 1)
    # and we hard-cut V<11
    sel &= (df['Vmag'] <= Vmagcut)
    # and P < 10 days, in order for the fraction of orbit covered by the
    # transit to be high enough. This throws out HD 80606.
    sel &= (df['Period(day)'] <= 10)

    adf = df[sel].sort_values('Period(day)')
    adf['tdur_phase'] = arr(adf['length']/adf['Period(day)'])

    # ASAS query string format
    # 5:26:50,-81:35:12
    asas_query_str = []
    for RAh, RAm, RAs, Decd, Decm, Decs in list(zip(
        arr(adf['RAh']).astype(str),
        arr(adf['RAm']).astype(str),
        arr(adf['RAs']).astype(str),
        arr(adf['Decd']).astype(str),
        arr(adf['Decm']).astype(str),
        arr(adf['Decs']).astype(str),
    )):
        asas_query_str.append(':'.join([RAh, RAm, RAs])+' '+
                              ':'.join([Decd, Decm, Decs]))

    adf['asas_query_str'] = asas_query_str

    savpath = (
        '../data/asas_all_well-studied_HJs_depthgt1pct_Vlt{:d}_Plt10.csv'
        .format(Vmagcut)
    )
    adf.to_csv(savpath, index=False)
    print('saved %s' % savpath)
