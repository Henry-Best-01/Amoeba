'''
Exactly as titled, this script converts a magnification map from binary data
into a .fits file, for faster loading
'''
import numpy as np
import QuasarModelFunctions as QMF
from astropy.io import fits

fname = '/Users/henrybest/PythonStuff/LensingMaps/SampleMagMaps/map_3/map.bin'
wname = 'map.fits'

with open(fname, 'rb') as f:
    MagMap = np.fromfile(f, 'i', count=-1, sep='')
MagMap2d = QMF.ConvertMagMap(MagMap)

hdu = fits.PrimaryHDU(MagMap2d)
hdu.writeto(wname)
