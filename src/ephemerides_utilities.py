'''
Functions here are named:
    get_ROUGH_epochs_given_midtime_and_period
    get_half_epochs_given_occultation_times
'''
import numpy as np

from astrobase.timeutils import get_epochs_given_midtimes_and_period

def get_half_epochs_given_occultation_times(tsec, init_period, t0):
    '''
    assume zero eccentricity. then

        t_sec = t0 + period*(epoch + 0.5)

    where "epoch" is the integer epoch counter for the transits.

    this function returns the half-epochs. t0 should come from the more
    constraining transit times (rather than occultation times).
    '''

    epoch_plushalf = (tsec - t0)/init_period

    epoch_plushalf_rounded = np.round(epoch_plushalf,1)

    # ensure the output array only has half-epoch secondary eclipse times.
    out_epochs = []
    for ix, er in enumerate(epoch_plushalf_rounded):
        if not repr(er)[-1] == str(5) and not repr(er)[-1] == str(0):
            out_epochs.append(np.nan)
        else:
            out_epochs.append(er)

    out_epochs = np.array(out_epochs)

    return out_epochs
