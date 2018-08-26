import pandas as pd, numpy as np

def compute_tess_sector_time_windows():
    '''
    TESS began its operations on July 25, 2018.  In a direction of RA= 0h0m0s,
    this is

        HJD = 2458324.500000

    (in different directions, it changes by up to 24 hours!)

    computed using http://britastro.org/computing/applets_dt.html
    '''

    tess_start_HJD = 2458324.50
    sector_duration = 27.3 # days, roughly
    sector_d = {}
    sector_d['start_time_HJD'] = []
    sector_d['end_time_HJD'] = []
    sector_d['sector_num'] = []
    for i in range(1,14):
        sector_d['start_time_HJD'].append(tess_start_HJD + (i-1)*sector_duration)
        sector_d['end_time_HJD'].append(tess_start_HJD + i*sector_duration)
        sector_d['sector_num'].append(i)

    df = pd.DataFrame(sector_d)

    df.to_csv('../data/tess_sector_time_windows.csv', index=False)

if __name__ == '__main__':
    compute_tess_sector_time_windows()
