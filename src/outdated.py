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


