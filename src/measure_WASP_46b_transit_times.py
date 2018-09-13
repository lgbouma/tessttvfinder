'''
we have a many alerted lightcurve. they have some transits. measure the times
that they fall at by fitting models!
'''
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from mast_utils import tic_single_object_crossmatch
from astropy import units as u

from astrobase.varbase import lcfit
from astrobase import astrotess as at
from astrobase.periodbase import kbls
from astrobase.varbase.trends import smooth_magseries_ndimage_medfilt
from astrobase import lcmath

np.random.seed(42)

def get_limb_darkening_initial_guesses(lcfile):

    # get object RA/dec, so that you can get the Teff/logg by x-matching TIC,
    # so that you can get the theoretical limb-darkening coefficients.
    hdulist = fits.open(lcfile)
    main_hdr = hdulist[0].header
    lc_hdr = hdulist[1].header
    lc = hdulist[1].data
    ra, dec = lc_hdr['RA_OBJ'], lc_hdr['DEC_OBJ']
    sep = 0.1*u.arcsec
    obj = tic_single_object_crossmatch(ra,dec,sep.to(u.deg).value)
    if len(obj['data'])==1:
        teff = obj['data'][0]['Teff']
        logg = obj['data'][0]['logg']
        metallicity = obj['data'][0]['MH'] # often None
        if not isinstance(metallicity,float):
            metallicity = 0 # solar
    else:
        raise NotImplementedError

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

    u_linear = bar['aLSM']
    u_quad = bar['bLSM']

    return float(u_linear), float(u_quad)

def get_transit_times(fitd, time, N):
    '''
    Given a BLS period, epoch, and transit ingress/egress points, compute
    the times within ~N transit durations of each transit.  This is useful
    for fitting & inspecting individual transits.
    '''

    tmids = [fitd['epoch'] + ix*fitd['period'] for ix in range(-1000,1000)]
    sel = (tmids > np.nanmin(time)) & (tmids < np.nanmax(time))
    tmids_obsd = np.array(tmids)[sel]
    if not fitd['transegressbin'] > fitd['transingressbin']:
        raise AssertionError('careful of the width...')
    tdur = (
        fitd['period']*
        (fitd['transegressbin']-fitd['transingressbin'])/fitd['nphasebins']
    )

    t_Is = tmids_obsd - tdur/2
    t_IVs = tmids_obsd + tdur/2

    # focus on the times around transit
    t_starts = t_Is - 5*tdur
    t_ends = t_Is + 5*tdur

    return tmids, t_starts, t_ends

######################
# PLOTTING UTILITIES #
######################
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



def main(ticid):

    ##########################################
    # detrending parameters. mingap: minimum gap to determine time group size.
    # smooth_window_day: window for median filtering.
    mingap = 240./60./24.
    smooth_window_day = 2.
    cadence_min = 2
    make_diagnostic_plots = True

    cadence_day = cadence_min / 60. / 24.
    windowsize = int(smooth_window_day/cadence_day)
    if windowsize % 2 == 0:
        windowsize += 1

    # paths for reading and writing data 
    lcdir = '../data/tess_lightcurves/'
    lcname = 'tess2018206045859-s0001-{:s}-111-s_llc.fits.gz'.format(
                str(ticid).zfill(16))
    lcfile = lcdir + lcname

    fit_savdir = '../results/lc_analysis/'
    blsfit_plotname = str(ticid)+'_bls_fit.png'
    trapfit_plotname = str(ticid)+'_trapezoid_fit.png'
    mandelagolfit_plotname = str(ticid)+'_mandelagol_fit_6d.png'
    corner_plotname = str(ticid)+'_corner_mandelagol_fit_6d.png'
    sample_plotname = str(ticid)+'_mandelagol_fit_samples_6d.h5'

    blsfit_savfile = fit_savdir + blsfit_plotname
    trapfit_savfile = fit_savdir + trapfit_plotname
    mandelagolfit_savfile = fit_savdir + mandelagolfit_plotname
    corner_savfile = fit_savdir + corner_plotname
    chain_savdir = '/Users/luke/local/emcee_chains/'
    samplesavpath = chain_savdir + sample_plotname
    ##########################################

    time, flux, err_flux = at.get_time_flux_errs_from_Ames_lightcurve(
                                lcfile, 'PDCSAP')

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

    if make_diagnostic_plots:
        single_whitening_plot(time, flux, smooth_flux, whitened_flux, ticid)

    # run bls to get initial parameters.
    endp = 1.05*(np.nanmax(time) - np.nanmin(time))/2
    blsdict = kbls.bls_parallel_pfind(time, flux, err_flux, magsarefluxes=True,
                                      startp=0.1, endp=endp,
                                      maxtransitduration=0.3, nworkers=8,
                                      sigclip=10.)
    fitd = kbls.bls_stats_singleperiod(time, flux, err_flux,
                                       blsdict['bestperiod'],
                                       magsarefluxes=True, sigclip=10.,
                                       perioddeltapercent=5)

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
                                              magsarefluxes=True,
                                              sigclip=10.,
                                              plotfit=trapfit_savfile)

    # fit a Mandel & Agol model to each single transit.

    tmids, t_starts, t_ends = get_transit_times(fitd, time, 5)

    for transit_ix, t_start, t_end in list(
        zip(range(len(t_starts)), t_starts, t_ends)
    ):

        try:
            sel = (time < t_end) & (time > t_start)
            sel_time = time[sel]
            sel_whitened_flux = whitened_flux[sel]
            sel_err_flux = err_flux[sel]

            u_linear, u_quad = get_limb_darkening_initial_guesses(lcfile)

            rp = np.sqrt(fitd['transitdepth'])

            initfitparams = {'t0':t_start + (t_end-t_start)/2.,
                             'rp':rp,
                             'sma':6.17,
                             'incl':85,
                             'u':[u_linear,u_quad] }

            fixedparams = {'ecc':0.,
                           'omega':90.,
                           'limb_dark':'quadratic',
                           'period':fitd['period'] }

            priorbounds = {'rp':(rp-0.01, rp+0.01),
                           'u_linear':(u_linear-1, u_linear+1),
                           'u_quad':(u_quad-1, u_quad+1),
                           't0':(np.min(sel_time), np.max(sel_time)),
                           'sma':(5.8,6.6),
                           'incl':(75,90) }

            spocparams = {'rp':0.14,
                          't0':1326.0089,
                          'u_linear':u_linear,
                          'u_quad':u_quad,
                          'sma':6.17,
                          'incl':None }

            t_num = str(transit_ix).zfill(3)
            mandelagolfit_plotname = (
                str(ticid)+'_mandelagol_fit_6d_t{:s}.png'.format(t_num)
            )
            corner_plotname = (
                str(ticid)+'_corner_mandelagol_fit_6d_t{:s}.png'.format(t_num)
            )
            sample_plotname = (
                str(ticid)+'_mandelagol_fit_samples_6d_t{:s}.h5'.format(t_num)
            )

            mandelagolfit_savfile = fit_savdir + mandelagolfit_plotname
            corner_savfile = fit_savdir + corner_plotname
            chain_savdir = '/Users/luke/local/emcee_chains/'
            samplesavpath = chain_savdir + sample_plotname

            print('beginning {:s}'.format(samplesavpath))

            plt.close('all')
            mandelagolfit = lcfit.mandelagol_fit_magseries(
                                sel_time, sel_whitened_flux, sel_err_flux,
                                initfitparams, priorbounds, fixedparams,
                                trueparams=spocparams, magsarefluxes=True,
                                sigclip=10., plotfit=mandelagolfit_savfile,
                                plotcorner=corner_savfile, samplesavpath=samplesavpath,
                                nworkers=8, n_mcmc_steps=10000, eps=1e-1, n_walkers=500,
                                skipsampling=False, overwriteexistingsamples=False)
        except:
            print('transit {:d} failed, continue')
            continue


if __name__ == '__main__':

    ticid = 231663901 # WASP-46b's TICID

    main(ticid)
