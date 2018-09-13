'''
For good targets (things crossmatched with Southworth's main TEPCAT database),
compile list of discovery ephemerides.
'''
from __future__ import division, print_function

import os, argparse
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd, numpy as np

from astropy import units as u
from astropy.coordinates import SkyCoord

from glob import glob

if __name__ == '__main__':

    tepcatpath = '../data/TEPCAT_observables.csv'
    tepcat_obsevables = pd.read_csv(tepcatpath, delimiter= ' *, *', engine='python')

    outdir = '../results/'
    fnames = [outdir+'tepcat_obsd_by_TESS_sector{:d}.csv'.format(sn)
                for sn in range(13)]

    for ix, fname in enumerate(fnames):
        sec_num = int(fname.split('sector')[-1].split('.csv')[0])
        assert sec_num >= 0
        assert sec_num <= 12

        df = pd.read_csv(fname)

        df = pd.merge(df, tepcat_obsevables, how='left', on='System')

        outname = 'tepcat_obsd_by_TESS_sector{:d}_with_ephemerides.csv'.format(sec_num)
        df.to_csv(outdir+outname, index=False)
        print('saved {:s}'.format(outdir+outname))
