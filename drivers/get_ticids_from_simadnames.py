from astrobase.services.convert_identifiers import simbad2tic

simbadnames = ['KELT-16', 'HATS-70', 'Kepler-91', 'KELT-9', 'OGLE-TR-56']

ticids = []
for simbadname in simbadnames:
    ticids.append(simbad2tic(simbadname))

for simbadname, ticid in zip(simbadnames, ticids):
    print('{} = {}'.format(simbadname, ticid))

print(*ticids, sep=' ')
