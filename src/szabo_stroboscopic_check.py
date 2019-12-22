"""
wasp-18 showed those ratty "sinusoidal" modulations. does Szabo+13's
stroboscopic sampling thing explain?
"""

import numpy as np
from astropy import units as u

P_orb = 0.9414526*u.day
P_samp = 2*u.minute

P_induced = (
    P_orb /
    (P_orb/P_samp - np.floor(P_orb/P_samp))
)

print(P_induced.to(u.day))
