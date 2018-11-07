# -*- coding: utf-8 -*-
'''
For WASP-18b, we got a cute ~7 sigma precovery from ~2000-2010 ASAS-3 V band
data.  WASP-18b has a 1.2% transit depth, at V=9.3.

How many HJs are amenable to such a measurement, with **DASCH**?

(probably none!)
'''
from __future__ import division, print_function

import matplotlib as mpl
mpl.use('Agg')

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
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

from ASAS_lightcurve_reduction import HJD_UTC_to_BJD_TDB_eastman, plot_old_lcs


def make_dasch_candidate_HJ_list():

    listpath = '../data/asas_all_well-studied_HJs_depthgt1pct_Vlt13_Plt10.csv'
    df = pd.read_csv(listpath)

    coords = SkyCoord(np.array(df['asas_query_str']),
                      unit=(u.hourangle, u.degree))

    # cf. http://dasch.rc.fas.harvard.edu/index.php
    dasch_sensitive = SkyCoord(150*u.deg, 30*u.deg, frame='galactic')

    seps = dasch_sensitive.separation(coords)

    df['coord'] = coords.to_string('decimal')
    df['sep_from_dasch_sensitive'] = seps

    outdf = df.sort_values(by='sep_from_dasch_sensitive')

    savname = (
        '../data/dasch_separations_well-studied_HJs_depthgt1pct_Vlt13_Plt10.csv'
    )
    outdf.to_csv(savname)
    print('saved {:s}'.format(savname))


def test_XO6b():
    '''
    i went to http://dasch.rc.fas.harvard.edu/lightcurve.php, and input
    6:19:10.36 +73:49:39.6, which are the coordinates for XO-6b, which i chose
    because it was closest to the region of high DASCH plate coverage.

    arcsec  Nobs Nplot   mag     id
        0   5144  1636 10.61 APASS_J061910.4+734940

    so after quality checks, 1636 data points (!).
    '''

    # options when running
    make_lc_plots = True

    plname = 'XO-6b'
    period, epoch = 3.7650007, 2456652.71245 # 2017AJ....153...94C
    # decimal ra, dec of target used only for BJD conversion
    coord = SkyCoord('6:19:10.36 +73:49:39.6', unit=(u.hourangle, u.deg))
    ra = coord.ra.value
    dec = coord.dec.value

    from astropy.io.votable import parse
    temp = parse('../data/XO-6b_DASCH.xml')
    tab = temp.get_first_table().to_table()

    # convert HJDs to BJDs with Eastman's applet
    hjds = nparr(tab['ExposureDate']) # NOTE probably not precise to the 6dp's
    tab['HJD'] = hjds

    # NOTE: Eastman's calculator does not work for dates before Dec 15th, 1949.
    #tab['BJD_TDB'] = HJD_UTC_to_BJD_TDB_eastman(hjds, ra, dec)

    tab['BJD_TDB'] = hjds # FIXME: for purposes of making something

    times, mags, errs = (nparr(tab['BJD_TDB']),
                         nparr(tab['magcal_magdep']),
                         nparr(tab['magcal_local_rms']))

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

    if make_lc_plots:
        plot_old_lcs(times, mags, stimes, smags, phzd, period, epoch, sfluxs,
                      plname, savdir='../results/DASCH_lightcurves/',
                     telescope='DASCH')


    import IPython; IPython.embed()



if __name__=="__main__":

    makelist = 0

    if makelist:
        make_dasch_candidate_HJ_list()

    test_XO6b()
