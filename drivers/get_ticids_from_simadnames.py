from astrobase.services.identifiers import simbad_to_tic

# Shai's HJs v0.1
# simbadnames = ['KELT-16', 'HATS-70', 'Kepler-91', 'KELT-9', 'OGLE-TR-56']

# # Knutson's RV gamma dots, plus Triaud+2017
# simbadnames = ['WASP-53', 'WASP-81', 'HAT-P-2', 'HAT-P-4', 'HAT-P-7',
#                'HAT-P-10', 'HAT-P-11', 'HAT-P-13', 'HAT-P-22', 'HAT-P-29',
#                'HAT-P-32', 'WASP-10', 'WASP-22', 'XO-2', 'Kepler-1']

# Lots of TESS data for these, or of historic interest
# #'OGLE2-TR-L9', 'OGLE-TR-211': failed!
# simbadnames = ['WASP-100', 'WASP-62', 'WASP-126']


#FIXME: TODO RUN THESE WHEN SIMBAD QUERIES WORK ...
# >2 sectors of data, or of other interest
# simbadnames = ['K2-182', 'K2-180', 'KELT-15', 'WASP-119']
# simbadnames = ['Kepler-1658', 'WASP-19', 'HAT-P-7']

simbadnames = [
    'sig Ori E',
    # 'RZ Psc',
    # 'V1959 Ori',
    # 'V1999 Ori',
    # 'V2227 Ori',
    # 'V2559 Ori'
    # 'RY Lup' # 'DF Tau'
    #'CD-36 3202', 'HAT-P-11', 'WASP-12', 'iot Dra'
]

ticids = []
for simbadname in simbadnames:
    ticids.append(simbad_to_tic(simbadname))

for simbadname, ticid in zip(simbadnames, ticids):
    print('{} = {}'.format(simbadname, ticid))

print(*ticids, sep=' ')
