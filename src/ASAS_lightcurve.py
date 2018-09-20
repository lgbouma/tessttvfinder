# -*- coding: utf-8 -*-
'''
Tools for working with ASAS lightcurves.
'''
from __future__ import division, print_function

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import numpy as np, pandas as pd
from numpy import array as nparr

from astropy.io import ascii
from astropy.io import fits
from astropy import units as u, constants as const
from astropy.table import Table

from astrobase.periodbase import kbls
from astrobase.plotbase import plot_phased_mag_series
from astrobase import periodbase, checkplot
from astrobase.lcmath import phase_magseries, sigclip_magseries
from astrobase.varbase import lcfit

from glob import glob
from parse import parse, search
import os, pickle

def read_ASAS_lightcurve(lcfile):
    '''
    querying the ASAS All Star Catalog at
        http://www.astrouw.edu.pl/asas/?page=aasc
    gives lightcurves in a particular format. This is the function that reads
    them.
    '''

    with open(lcfile, 'r') as f:
        lines = f.readlines()

    start_ind = [ix for ix, line in enumerate(lines) if
                 'LIGHT CURVE BEGINS NEXT LINE' in line]

    if len(start_ind) != 1:
        raise AssertionError('expected a single start line per lcfile')
    start_ind = start_ind[0]

    lines = lines[start_ind:]

    comment_lines, data_lines = [], []
    for line in lines:
        if line.startswith('#'):
            comment_lines.append(line)
        else:
            data_lines.append(line.rstrip('\n').split())


    temp_data_lines = []
    for d in data_lines:
        temp_data_lines.append( [
        float(d[0]), float(d[1]), float(d[2]), float(d[3]),
        float(d[4]), float(d[5]), float(d[6]), float(d[7]),
        float(d[8]), float(d[9]), float(d[10]), str(d[11]),
        int(d[12])
        ])

    colnames = ['HJD','MAG_3','MAG_0','MAG_1',
                'MAG_2','MAG_4','MER_3','MER_0','MER_1',
                'MER_2','MER_4','GRADE','FRAME']
    print('taking %s as column names' % repr(colnames))

    df = pd.DataFrame(temp_data_lines, columns=colnames)

    # get n_datasets and number of points in each. split up the data lines
    # appropriately.
    n_obs = []
    for cl in comment_lines:
        result = search('ndata= {:d}', cl)
        if not result==None:
            n_obs.append(result[0])

    # in each "dataset" might have slightly different mean magnitudes. these
    # are "independent" datasets (in that they were observed by ASAS in
    # different "fields", similar to HAT).
    n_datasets = len(n_obs)

    if not (np.sum(n_obs)==len(data_lines)):
        raise AssertionError('each line in datalines should be an observation')

    sedges = np.concatenate((nparr([0]), np.cumsum(n_obs)))
    dslices = [slice(sedges[ix], sedges[ix+1]) for ix in range(len(sedges)-1)]

    return df, dslices


def HJD_UTC_to_BJD_TDB_eastman(hjd_arr, ra, dec):
    '''
    CITE: Eastman et al. (2010)

    example query:
    'http://astroutils.astronomy.ohio-state.edu/time/convert.php?'+
    'JDS=2450000.123,2450000,2451234.123&RA=219.90206&DEC=-60.837528&'+
    'FUNCTION=hjd2bjd'

    As written, I hackily split the query into twos. NOTE: will need to
    generalize some day, maybe.  max querystr length is something like 8000
    characters, which is why the split is needed.
    '''
    import urllib.request

    hjd_longstr = ','.join(list(map(str,hjd_arr)))

    n_times = len(hjd_arr)
    n_characters_per_time = int(np.mean(list(map(len,hjd_longstr.split(',')))))

    if 2 < n_characters_per_time*n_times < 10000:
        hjd_strs = [ ','.join(list(map(str,hjd_arr[0:int(n_times/2) ]))),
                     ','.join(list(map(str,hjd_arr[int(n_times/2):])))
                   ]
    else:
        raise NotImplementedError('need to generalize code for N queries')

    querystrs=[
        'http://astroutils.astronomy.ohio-state.edu/time/convert.php?'+
        'JDS={:s}'.format(hjd_str)+
        '&RA={:f}&DEC={:f}&'.format(ra, dec)+
        'FUNCTION=hjd2bjd'
        for hjd_str in hjd_strs
    ]

    reqs = [urllib.request.Request(querystr) for querystr in querystrs]
    responses = [urllib.request.urlopen(req) for req in reqs]
    bjds = [response.read().decode('ascii').rstrip('\n') for response in
            responses]

    bjds = [bjd.split('\n') for bjd in bjds]
    bjds_flattened = [v for sublist in bjds for v in sublist]

    bjds = nparr(list(map(float, bjds_flattened)))

    return bjds


def wrangle_ASAS_lightcurve(df, dslices, ra, dec, min_N_obs=10):
    '''
    do the following janitorial tasks:

    * populate the 'dataset_number' column
    * only take "datasets" with N_observations > 10
    * select only "A" and "B" quality frames
    * convert HJDs to BJDs with Eastman's applet
    * normalize different "datasets" to the same median magnitude by introducing
      an artificial additive shift in magnitude to each.
    * enforce that the actual medians are close to the originally measured
      medians
    * produce a "best aperture" column, which is whichever has the lowest RMS.

    return df with the columns:
        Index(['HJD', 'MAG_3', 'MAG_0', 'MAG_1', 'MAG_2', 'MAG_4', 'MER_3',
        'MER_0', 'MER_1', 'MER_2', 'MER_4', 'GRADE', 'FRAME', 'dataset_number',
        'BJD_TDB', 'SMAG_0', 'SMAG_1', 'SMAG_2', 'SMAG_3', 'SMAG_4'],
        dtype='object')
    '''

    # populate the 'dataset_number' column
    dataset_number = []
    for ix, dslice in enumerate(dslices):
        if len(dataset_number)==0:
            dataset_number = [ix] * (dslice.stop - dslice.start)
        else:
            dataset_number.extend(
                [ix] * (dslice.stop - dslice.start))

    assert len(dataset_number) == len(df)

    df['dataset_number'] = dataset_number
    del dataset_number

    # only take "datasets" with N_observations > 10
    min_N_obs = 10

    print('{:d} points before dropping short datasets'.format(len(df)))

    if len(dslices)>1:
        for ix, dslice in enumerate(dslices):
            N_obs = len(df.iloc[dslice])
            if N_obs < min_N_obs:
                print('dropping points {:d} to {:d}'.format(
                    dslice.start, dslice.stop))
                df.drop(list(range(dslice.start, dslice.stop)), inplace=True,
                       axis='index')
                _ = dslices.pop(ix)

    print('{:d} points after dropping short datasets'.format(len(df)))
    print('{:d} points before dropping poor quality frames'.format(len(df)))

    # select only "A" and "B" quality frames
    df = df[(df['GRADE']=='A') | (df['GRADE']=='B')]

    print('{:d} points after dropping poor quality frames'.format(len(df)))

    # convert HJDs to BJDs with Eastman's applet
    hjds = nparr(df['HJD']) + 2450000
    df['HJD'] = hjds
    df['BJD_TDB'] = HJD_UTC_to_BJD_TDB_eastman(hjds, ra, dec)

    ##########################################
    # normalize different "datasets" to the same median magnitude by introducing
    # an artificial additive shift in magnitude to each.
    ##########################################

    # for each of the 5 apertures, calculate their median magnitude over all
    # observations
    globalmedians = [np.nanmedian(df['MAG_{:d}'.format(ix)])
                     for ix in range(0,5)]

    # for each surviving dataset shift each aperture's magnitudes to the same
    # global median
    datasets = np.sort(np.unique(df['dataset_number']))

    shifted_mags = []
    for dataset in datasets:
        tdf = df[df['dataset_number']==dataset]
        slice_shifted_mags = {}
        for ap in range(0,5):
            this_mag = nparr(tdf['MAG_{:d}'.format(ap)])
            current_median = np.nanmedian(this_mag)
            this_shifted_mag = this_mag + (globalmedians[ap] - current_median)
            slice_shifted_mags['ap_{:d}'.format(ap)] = this_shifted_mag
        shifted_mags.append(slice_shifted_mags)

    # get a flat array for each aperture
    for ap in range(0,5):
        thisap = []
        for shifted_mag in shifted_mags:
            thisap.extend(shifted_mag['ap_{:d}'.format(ap)])
        df['SMAG_{:d}'.format(ap)] = nparr(thisap)

    # enforce that the actual medians are close to the originally measured
    # medians
    datamedian = nparr([
        np.nanmedian(df[df['dataset_number']==dataset]['SMAG_{:d}'.format(ix)])
        for ix in range(0,5) for dataset in datasets])

    desiredmedians = nparr(
        [np.repeat(gm, len(datasets)) for gm in globalmedians]).flatten()

    np.testing.assert_array_almost_equal(datamedian, desiredmedians,
                                         decimal=10)

    # produce a "best aperture" column, which is whichever has the lowest RMS.
    stddev_aps = []
    for ap in range(0,5):
        stddev_aps.append(
            np.std(nparr(df['SMAG_{:d}'.format(ap)]))
        )
    print('stddev_aps: %s' % repr(stddev_aps))
    best_ap_ind = np.argmin(stddev_aps)
    df['SMAG_bestap'] = nparr(df['SMAG_{:d}'.format(best_ap_ind)])
    df['SERR_bestap'] = nparr(df['MER_{:d}'.format(best_ap_ind)])
    print('chose %d as best ap' % best_ap_ind)

    return df


def plot_asas_lcs(times, mags, stimes, smags, phasedict, period, epoch, sfluxs,
                 savdir='../results/ASAS_lightcurves/'):

    f,ax=plt.subplots(figsize=(12,6))
    ax.scatter(times, mags)
    ax.set_xlabel('BJD TDB')
    ax.set_ylabel('ASAS mag (best ap)')
    ax.set_ylim([max(ax.get_ylim()), min(ax.get_ylim())])
    f.savefig(savdir+'WASP-18b_bestap.png', dpi=400)
    plt.close('all')

    f,ax=plt.subplots(figsize=(12,6))
    ax.scatter(stimes, smags)
    ax.set_xlabel('BJD TDB')
    ax.set_ylabel('sigclipped ASAS mag (best ap)')
    ax.set_ylim([max(ax.get_ylim()), min(ax.get_ylim())])
    f.savefig(savdir+'WASP-18b_sigclipped_bestap.png', dpi=400)
    plt.close('all')

    f,ax=plt.subplots(figsize=(12,6))
    ax.scatter(phasedict['phase'], phasedict['mags'])
    ax.set_xlabel('phase')
    ax.set_ylabel('sigclipped ASAS mag (best ap)')
    ax.set_ylim([max(ax.get_ylim()), min(ax.get_ylim())])
    ax.set_xlim([-.6,.6])
    f.savefig(savdir+'WASP-18b_phased_on_discovery_params.png', dpi=400)
    plt.close('all')

    outfile = savdir+'WASP-18b_phased_on_discovery_params_binned.png'
    plot_phased_mag_series(stimes, smags, period, magsarefluxes=False,
                           errs=None, normto=False, epoch=epoch,
                           outfile=outfile, sigclip=False, phasebin=0.02,
                           plotphaselim=[-.6,.6], plotdpi=400)

    outfile = savdir+'WASP-18b_fluxs_phased_on_discovery_params_binned.png'
    plot_phased_mag_series(stimes, sfluxs, period, magsarefluxes=True,
                           errs=None, normto=False, epoch=epoch,
                           outfile=outfile, sigclip=False, phasebin=0.02,
                           plotphaselim=[-.6,.6], plotdpi=400)

def run_asas_periodograms(times, mags, errs):

    blsdict = kbls.bls_parallel_pfind(times, mags, errs,
                                      magsarefluxes=False, startp=0.5,
                                      endp=1.5, maxtransitduration=0.3,
                                      nworkers=8, sigclip=[15,3])
    gls = periodbase.pgen_lsp(times, mags, errs, magsarefluxes=False,
                              startp=0.5, endp=1.5, nworkers=8,
                              sigclip=[15,3])
    outdir = '../results/ASAS_lightcurves/'
    outname = 'WASP-18b_BLS_GLS.png'
    outpath = outdir+outname
    cpf = checkplot.twolsp_checkplot_png(blsdict, gls, times, mags, errs,
                                         outfile=outpath, objectinfo=None)



if __name__ == "__main__":

    plname = 'WASP-18b'
    # table 1 of Hellier et al 2009 discovery paper
    period = 0.94145299
    epoch = 2454221.48163
    # decimal ra, dec of target used only for BJD conversion
    ra, dec = 24.354311, -45.677887

    # options when running
    try_to_recover_periodograms = False
    make_lc_plots = True

    # file parsing
    lcdir = '../data/ASAS_lightcurves/'
    asas_lcs = glob(lcdir+'*.txt')
    lcfile = asas_lcs[0]

    fit_savdir = '../results/ASAS_lightcurves/'
    chain_savdir = '/Users/luke/local/emcee_chains'
    savdir='../results/ASAS_lightcurves/'

    #########
    # begin #
    #########
    tempdf, dslices = read_ASAS_lightcurve(lcfile)
    df = wrangle_ASAS_lightcurve(tempdf, dslices, ra, dec)

    times, mags, errs = (nparr(df['BJD_TDB']), nparr(df['SMAG_bestap']),
                         nparr(df['SERR_bestap']))

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

    if try_to_recover_periodograms:
        run_asas_periodograms(times, mags, errs)

    if make_lc_plots:
        plot_asas_lcs(times, mags, stimes, smags, phzd, period, epoch, sfluxs)

    ####################################################################
    # fit the lightcurve, show the phased result, get the transit time #
    ####################################################################
    # get the guess physical parameters from Hellier+ 2009
    true_b, true_sma, true_t0, true_rp = (
        0.25, 0.02026, epoch, np.sqrt(0.00875) )
    u_linear, u_quad = 0, 0

    true_incl = None
    if isinstance(true_b,float) and isinstance(true_sma, float):
        # b = a/Rstar * cosi
        cosi = true_b / true_sma
        true_incl = np.degrees(np.arccos(cosi))

    initfitparams = {'t0':epoch, 'rp':true_rp, 'sma':true_sma, 'incl':85,
                     'u':[u_linear,u_quad] }

    fixedparams = {'ecc':0., 'omega':90., 'limb_dark':'quadratic',
                   'period':period }

    priorbounds = {'rp':(true_rp-0.1, true_rp+0.1),
                   'u_linear':(u_linear-1, u_linear+1),
                   'u_quad':(u_quad-1, u_quad+1),
                   't0':(epoch-0.03, epoch+0.03),
                   'sma':(0.7*true_sma,1.3*true_sma), 'incl':(75,90) }

    discoveryparams = {'rp':true_rp, 't0':true_t0, 'u_linear':u_linear,
                       'u_quad':u_quad, 'sma':true_sma, 'incl':true_incl }

    mandelagolfit_plotname = ( str(plname)+'_mandelagol_fit_6d_fixperiod.png')
    corner_plotname = ( str(plname)+'_corner_mandelagol_fit_6d_fixperiod.png')
    sample_plotname = ( str(plname)+'_mandelagol_fit_samples_6d_fixperiod.h5')

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

    mandelagolfit = lcfit.mandelagol_fit_magseries(
                    stimes, sfluxs, serrs,
                    initfitparams, priorbounds, fixedparams,
                    trueparams=discoveryparams, magsarefluxes=True,
                    sigclip=None, plotfit=mandelagolfit_savfile,
                    plotcorner=corner_savfile,
                    samplesavpath=samplesavpath, nworkers=8,
                    n_mcmc_steps=1000, eps=1e-2, n_walkers=500,
                    skipsampling=False,
                    overwriteexistingsamples=False,
                    mcmcprogressbar=True)

    maf_savpath = "../data/"+str(plname)+"_mandelagol_fit_6d_fixperiod.pickle"
    with open(maf_savpath, 'wb') as f:
        pickle.dump(mandelagolfit, f, pickle.HIGHEST_PROTOCOL)
        print('saved {:s}'.format(maf_savpath))

    fitfluxs = mandelagolfit['fitinfo']['fitmags']
    initfluxs = mandelagolfit['fitinfo']['initmags']

    import IPython; IPython.embed()
    #FIXME

    outfile = savdir+'WASP-18b_phased_mandelagolfit.png'
    plot_phased_mag_series(stimes, sfluxs, period, magsarefluxes=True,
                           errs=serrs, normto=False, epoch=epoch,
                           outfile=outfile, sigclip=False, phasebin=0.02,
                           plotphaselim=[-.6,.6], plotdpi=400, linewidth=0,
                           elinewidth=0)


