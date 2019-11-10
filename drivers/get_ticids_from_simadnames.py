from astrobase.services.convert_identifiers import simbad2tic

# Shai's HJs v0.1
# simbadnames = ['KELT-16', 'HATS-70', 'Kepler-91', 'KELT-9', 'OGLE-TR-56']

# # Knutson's RV gamma dots, plus Triaud+2017
# simbadnames = ['WASP-53', 'WASP-81', 'HAT-P-2', 'HAT-P-4', 'HAT-P-7',
#                'HAT-P-10', 'HAT-P-11', 'HAT-P-13', 'HAT-P-22', 'HAT-P-29',
#                'HAT-P-32', 'WASP-10', 'WASP-22', 'XO-2', 'Kepler-1']

# Lots of TESS data for these, or of historic interest
# 'OGLE2-TR-L9', 'OGLE-TR-211'
simbadnames = ['WASP-100', 'WASP-62']

ticids = []
for simbadname in simbadnames:
    ticids.append(simbad2tic(simbadname))

for simbadname, ticid in zip(simbadnames, ticids):
    print('{} = {}'.format(simbadname, ticid))

print(*ticids, sep=' ')
