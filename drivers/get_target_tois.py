"""
Generally, we're interested in any "known planet" TOIs with Rp > 4Re, and
period < 30 days.
"""

import os
import numpy as np, pandas as pd
from cdips.utils.catalogs import get_exofop_toi_catalog

from numpy import array as nparr

from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u, constants as const
from astropy import units

from astrobase.services.identifiers import simbad_to_tic


def get_bonomo_2017_mergetable():

    t7 = Table.read("../data/Bonomo_2017_table7.vot", format="votable")
    t8 = Table.read("../data/Bonomo_2017_table8.vot", format="votable")
    t9 = Table.read("../data/Bonomo_2017_table9.vot", format="votable")

    df7 = t7.to_pandas()
    df8 = t8.to_pandas()
    df9 = t9.to_pandas()

    # fix the unicode
    for k in ['Planet','Forbit','Fcirc','Fcomp']:
        df8[k] = list(map(lambda x: x.decode(), df8[k]))
    for k in ['Star','Ref','n_Ref']:
        df7[k] = list(map(lambda x: x.decode(), df7[k]))

    planets = nparr([''.join(p.split()) for p in nparr(df8['Planet'])])
    stars = nparr(df7['Star'])

    Mstar = nparr(df7['Mstar'])
    Rstar = nparr(df7['Rstar'])
    Teff = nparr(df7['Teff'])

    Mplanet = nparr(df9['Mp'])
    Rplanet = nparr(df7['Rplanet'])

    sma = nparr(df9['smaxis'])
    period = nparr(df9['Period'])

    forbit = nparr(df8['Forbit'])
    fcirc = nparr(df8['Fcirc'])
    fcomp = nparr(df8['Fcomp'])

    df7_planets = nparr([''.join(s.split()) + 'b' for s in stars])

    try:
        np.testing.assert_array_equal(planets, df7_planets)
    except AssertionError:
        print(np.setdiff1d(df7_planets, planets))
        print(np.setdiff1d(planets, df7_planets))

    df = pd.DataFrame({
        'planet':planets,
        'star':stars,
        'Mstar':Mstar,
        'Rstar':Rstar,
        'Teff':Teff,
        'Mplanet':Mplanet,
        'Rplanet':Rplanet,
        'sma':sma,
        'period':period,
        'forbit':forbit,
        'fcirc':fcirc,
        'fcomp':fcomp,
        'ecc':nparr(df8['ecc']),
        'ecc_merr':nparr(df8['e_ecc']),
        'ecc_perr':nparr(df8['E_ecc'])
    })

    return df


def get_select_tois():

    df_toi = get_exofop_toi_catalog()

    sel = (
        (df_toi['TFOPWG Disposition'] != 'FP')
        &
        (df_toi['Period (days)'] < 30)
        &
        (df_toi['Planet Radius (R_Earth)'] > 4)
    )

    sdf_toi = df_toi[sel]

    return sdf_toi


#
# helper functions from Ragozzine & Wolf 2009
#

def calc_ωdot_GR(Mstar, P_orb, e, sma):
    n = 2*np.pi/P_orb
    ωdot_GR = 3*const.G*Mstar*n / (sma * const.c**2 * (1-e*e))
    return ωdot_GR

def _g2(e):
    return (1-e**2)**(-2)

def calc_ωdot_rotstar(Mstar, P_orb, e, sma, Rstar, P_rot_star=15*u.day,
                      k2_star=0.0351):

    n = 2*np.pi/P_orb

    ν_star = 2*np.pi/P_rot_star

    ωdot_rotstar = (
        0.5*k2_star*(Rstar/sma)**5 * ν_star**2 * sma**3 / (const.G*Mstar)
        * _g2(e) * n
    )

    return ωdot_rotstar

def calc_ωdot_rotplanet(Mplanet, P_orb, e, sma, Rplanet, k2_planet=0.59):

    n = 2*np.pi/P_orb

    P_rot_planet = P_orb # assumed
    ν_planet = 2*np.pi/P_rot_planet

    ωdot_rotplanet = (
        0.5*k2_planet*(Rplanet/sma)**5 * ν_planet**2 * sma**3 /
        (const.G*Mplanet)
        * _g2(e) * n
    )

    return ωdot_rotplanet

def _f2(e):
    return (1-e**2)**(-5) * (1 + 1.5*e**2 + (1/8)*e**4)

def calc_ωdot_tidalstar(Rstar, sma, Mplanet, Mstar, e, P_orb,
                        k2_star=0.0351):

    n = 2*np.pi/P_orb

    ωdot_tidalstar = (
        15/2*k2_star*(Rstar/sma)**5 * (Mplanet/Mstar)
        * _f2(e) * n
    )

    return ωdot_tidalstar

def calc_ωdot_tidalplanet(Rplanet, sma, Mstar, Mplanet, e, P_orb,
                          k2_planet=0.59):

    n = 2*np.pi/P_orb

    ωdot_tidalplanet = (
        15/2*k2_planet*(Rplanet/sma)**5 * (Mstar/Mplanet)
        * _f2(e) * n
    )

    return ωdot_tidalplanet


def _get_precession_rates(mdf):

    Mstar = nparr(mdf.Mstar)*u.Msun
    Rstar = nparr(mdf.Rstar)*u.Rsun
    Mplanet = nparr(mdf.Mplanet)*u.Mjupiter
    Rplanet = nparr(mdf.Rplanet)*u.Rjupiter
    P_orb = nparr(mdf.period)*u.day
    e = nparr(mdf.ecc)
    sma = nparr(mdf.sma)*u.au

    k2_sun = 0.0351 # Goodman's notes
    k2_jupiter = 0.59

    ωdot_GR = calc_ωdot_GR(Mstar, P_orb, e, sma).cgs

    ωdot_rotstar = calc_ωdot_rotstar(Mstar, P_orb, e, sma, Rstar,
                                     P_rot_star=15*u.day, k2_star=k2_sun).cgs

    ωdot_rotplanet = calc_ωdot_rotplanet(Mplanet, P_orb, e, sma, Rplanet,
                                         k2_planet=k2_jupiter).cgs

    ωdot_tidalstar = calc_ωdot_tidalstar(Rstar, sma, Mplanet, Mstar, e,
                                         P_orb, k2_star=k2_sun).cgs

    ωdot_tidalplanet = calc_ωdot_tidalplanet(Rplanet, sma, Mstar, Mplanet, e,
                                             P_orb, k2_planet=k2_jupiter).cgs

    ωdot_tot = (
        ωdot_GR + ωdot_rotstar + ωdot_rotplanet +
        ωdot_tidalstar + ωdot_tidalplanet
    )

    mdf['omegadot_tot'] = ωdot_tot
    mdf['omegadot_GR'] = ωdot_GR
    mdf['omegadot_rotstar'] = ωdot_rotstar
    mdf['omegadot_rotplanet'] = ωdot_rotplanet
    mdf['omegadot_tidalstar'] = ωdot_tidalstar
    mdf['omegadot_tidalplanet'] = ωdot_tidalplanet

    contribs = ['GR', 'rotstar', 'rotplanet', 'tidalstar', 'tidalplanet']
    for c in contribs:
        mdf[f'frac_from_{c}'] = mdf[f'omegadot_{c}']/ωdot_tot

    return mdf


def get_target_tois():

    sdf_toi = get_select_tois()

    eccpath = '../data/Bonomo_2017_eccentric_subset.csv'

    if not os.path.exists(eccpath):
        df_b17 = get_bonomo_2017_mergetable()

        forbit = nparr(df_b17['forbit'])
        is_ecc = (forbit == 'E')

        df_ecc = df_b17[is_ecc]
        planets = nparr(df_ecc['planet'])
        stars = nparr(df_ecc['star'])

        ticids = []
        for s in stars:
            ticids.append(simbad_to_tic(str(s)))

        assert len(ticids) == len(stars)

        df_ecc['ticid'] = ticids
        df_ecc.to_csv(eccpath, index=False)

    else:
        df_ecc = pd.read_csv(eccpath)

    mdf = df_ecc[df_ecc.period < 30].merge(sdf_toi, how='left', left_on='ticid', right_on='TIC ID')

    mdf = _get_precession_rates(mdf)


    smdf = mdf[~pd.isnull(mdf['Sectors'])]


    print(f'N eccentric from Bonomo+17, P<30d: {len(mdf)}')
    print(f'N eccentric from Bonomo+17, P<30d, with TOI catalog xmatch: {len(smdf)}')

    scols = ['planet', 'ticid', 'Teff', 'ecc', 'Sectors', 'omegadot_tot', 'frac_from_GR', 'frac_from_tidalplanet']
    print(smdf[scols].sort_values(by='omegadot_tot', ascending=False))


if __name__ == "__main__":
    get_target_tois()
