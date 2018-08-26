'''
Using the `tessmaps` output for transiting planets observed by TESS (Stephen
Kane's base catalog), compile lists of planet properties by crossmatching
against Southworth's TEPCAT.

usage:
    $ python wrangle_tepcat.py > ../results/TEPCAT_TESS_overlap_list.txt
'''
from __future__ import division, print_function

import os, argparse
import pandas as pd, numpy as np

from astropy import units as u

from glob import glob

if __name__ == '__main__':

    # output of calculating which planets are in which sector
    infodir = '../../tessmaps/results/'
    fnames = [infodir+'kane_knownplanets_sector{:d}.csv'.format(sn)
                for sn in range(13)]

    tepcatpath = '../data/TEPCAT_allplanets.csv'
    tepcat_df = pd.read_csv(tepcatpath, delimiter= ' *, *', engine='python')

    for ix, fname in enumerate(fnames):
        sec_num = int(fname.split('sector')[-1].split('.csv')[0])
        assert sec_num >= 0
        assert sec_num <= 12

        df_with_pointings = pd.read_csv(fname)

        if ix == 0:
            print('{:d} systems in "TEPCAT well-studied transiting planet" catalog'.
                  format(len(tepcat_df)))

        ptng_names = np.array(df_with_pointings['pl_hostname'])
        tepcat_names = np.array(tepcat_df['System'])

        ptng_renamed = []
        for n in ptng_names:
            rename = n.replace(' ','_')
            if rename.endswith('_A'):
                rename = rename.replace('_A','')
            elif rename.endswith('_B'):
                rename = rename.replace('_B','')

            # zero-filling is required. e.g., WASP-035, HATS-03, K2-003
            str_zfills = [3,2,3,2]
            for str_ix, substr in enumerate(['WASP-','HATS-','K2-','CoRoT-']):
                if substr in rename:
                    number = int(rename.split(substr)[-1])
                    numstr = str(number).zfill(str_zfills[str_ix])
                    rename = substr+numstr
                    break

            ptng_renamed.append(rename)

        ptng_renamed = np.array(ptng_renamed)

        # crossmatch by appropriately formatted names
        tepcat_inds = np.isin(tepcat_names, ptng_renamed)
        ptng_inds = np.isin(ptng_renamed, tepcat_names)

        obsd_and_tepcat = ptng_renamed[ptng_inds]
        obsd_and_not_tepcat = ptng_renamed[~ptng_inds]
        print('\n')
        print(42*'#')
        print('sector {:d}: {:d} obsd and in tepcat'.
              format(sec_num, len(obsd_and_tepcat)))
        print('sector {:d}: {:d} obsd and not in tepcat (incl RV planets)'.
              format(sec_num, len(obsd_and_not_tepcat)))


        ########################################## 
        # for each sector: list of planet names, and salient properties from
        # TEPCAT.  including: Teff, a/R, planet mass.
        print('The following transiting planets are in TEPCAT:\n')

        # cut by TEPCAT name crossmatch. collect Tmags and sectors obsd.
        df_obsd = tepcat_df.iloc[tepcat_inds]

        a_by_Rstar = ( (np.array(df_obsd['a(AU)'])*u.AU) /
                      (np.array(df_obsd['R_A'])*u.Rsun)).cgs.value
        df_obsd['a_by_Rstar'] = a_by_Rstar

        df_with_pointings['System'] = ptng_renamed
        df_obsd = pd.merge(df_obsd, df_with_pointings, how='left', on='System')

        cols = ['System', 'a_by_Rstar', 'Teff', 'Period', 'a(AU)', 'R_A', 'M_b', 'Tmag',
                'total_sectors_obsd', 'Discovery_reference',
                'Recent_reference']

        print(df_obsd[cols].
              sort_values(['a_by_Rstar','Teff'], ascending=[True,False]).
              to_string(index=False, col_space=12)
             )

        csvpath = '../results/tepcat_obsd_by_TESS_sector{:d}.csv'.format(sec_num)
        df_obsd[cols].sort_values(
            ['a_by_Rstar','Teff'], ascending=[True,False]
            ).to_csv(csvpath, index=False)
        print('\nsaved to {:s}\n'.format(csvpath))

        ########################################## 
        not_tepcat = df_with_pointings.iloc[~ptng_inds]
        not_tepcat_but_transits = not_tepcat[not_tepcat['is_transiting']==1]

        cols = ['pl_hostname', 'is_transiting', 'Tmag', 'total_sectors_obsd']
        print(5*'-')
        if len(not_tepcat_but_transits[cols]) > 0:
            print('The following transiting planets were not in TEPCAT, and so \n'
                  'didnt make the list, but are still known transiting planets:\n')
            print(not_tepcat_but_transits[cols].
                  sort_values(['total_sectors_obsd','Tmag'], ascending=[False, True]).
                  to_string(index=False, col_space=12)
                 )
        else:
            print('In this sector, no known transiting planets were not \n'
                  'also in TEPCAT\n')
