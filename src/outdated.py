def retrieve_measured_times_mcmc(ticid):
    # NOTE: outdated. phase out soon!

    sampledir = '/Users/luke/local/emcee_chains/'
    fpattern = '{:s}_mandelagol_fit_samples_6d_t???.h5'.format(str(ticid))
    fnames = glob(sampledir+fpattern)

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

    # paths for reading and writing plots
    lcdir = '../data/tess_lightcurves/'
    lcname = 'tess2018206045859-s0001-{:s}-111-s_llc.fits.gz'.format(
                str(ticid).zfill(16))
    lcfile = lcdir + lcname
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

    # fit Mandel & Agol model to each single transit, +/- 5 transit durations
    tmids, t_starts, t_ends = get_transit_times(fitd, time, 5)

    t0_list, t0_merrs, t0_perrs, t0_bigerrs, samplepaths = [],[],[],[],[]

    for transit_ix, t_start, t_end in list(
        zip(range(len(t_starts)), t_starts, t_ends)
    ):

        try:
            sel = (time < t_end) & (time > t_start)
            sel_time = time[sel]
            sel_whitened_flux = whitened_flux[sel]
            sel_err_flux = err_flux[sel]

            u_linear, u_quad = get_limb_darkening_initial_guesses(lcfile)
            a_guess = get_a_over_Rstar_guess(lcfile, fitd['period'])

            rp = np.sqrt(fitd['transitdepth'])

            initfitparams = {'t0':t_start + (t_end-t_start)/2., 'rp':rp,
                             'sma':a_guess, 'incl':85, 'u':[u_linear,u_quad] }

            fixedparams = {'ecc':0., 'omega':90., 'limb_dark':'quadratic',
                           'period':fitd['period'] }

            priorbounds = {'rp':(rp-0.01, rp+0.01),
                           'u_linear':(u_linear-1, u_linear+1),
                           'u_quad':(u_quad-1, u_quad+1),
                           't0':(np.min(sel_time), np.max(sel_time)),
                           'sma':(0.7*a_guess,1.3*a_guess), 'incl':(75,90) }

            t_num = str(transit_ix).zfill(3)
            sample_plotname = (
                str(ticid)+'_mandelagol_fit_samples_6d_t{:s}.h5'.format(t_num)
            )

            chain_savdir = '/Users/luke/local/emcee_chains/'
            samplesavpath = chain_savdir + sample_plotname

            print('beginning {:s}'.format(samplesavpath))

            mandelagolfit = lcfit.mandelagol_fit_magseries(
                            sel_time, sel_whitened_flux, sel_err_flux,
                            initfitparams, priorbounds, fixedparams,
                            trueparams=None, magsarefluxes=True,
                            sigclip=10., plotfit=None,
                            plotcorner=None,
                            samplesavpath=samplesavpath, nworkers=8,
                            n_mcmc_steps=42, eps=1e-1, n_walkers=500,
                            skipsampling=True, overwriteexistingsamples=False)

            fitparams = mandelagolfit['fitinfo']['finalparams']
            fiterrs = mandelagolfit['fitinfo']['finalparamerrs']

            t0_list.append(fitparams['t0'])
            t0_merrs.append(fiterrs['std_merrs']['t0'])
            t0_perrs.append(fiterrs['std_perrs']['t0'])
            t0_bigerrs.append(max(
                fiterrs['std_merrs']['t0'],fiterrs['std_perrs']['t0']))
            samplepaths.append(samplesavpath)

        except Exception as e:
            print(e)
            print('transit {:d} failed, continue'.format(transit_ix))
            continue

    t0, t0_merr, t0_perr, t0_bigerr = (
        nparr(t0_list),nparr(t0_merrs),nparr(t0_perrs),nparr(t0_bigerrs)
    )

    df = pd.DataFrame({
        't0_BTJD':t0, 't0_merr':t0_merr, 't0_perr':t0_perr,
        't0_bigerr':t0_bigerr, 'BJD_TDB':t0+2457000, 'samplepath':samplepaths
    })
    outdir = '../data/'
    outname = (
        str(ticid)+'_measured_TESS_times_{:d}_transits.csv'.format(transit_ix)
    )
    df.to_csv(outdir+outname, index=False)
    print('saved to {:s}'.format(outdir+outname))



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
              format(len(err_tmid_HJD[err_tmid_HJD*24*60 < 1.])))
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


