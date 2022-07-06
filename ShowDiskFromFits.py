'''
This opens a .fits file and plots the accretion disk image
'''

import numpy as np
import matplotlib.pyplot as plt
import QuasarModelFunctions as QMF
from scipy.ndimage import rotate
from astropy.io import fits
from astropy import constants as const
from astropy import units as u


file = '/Users/henrybest/PythonStuff/DiskImages/Sim5Disks/Sim5ThinDisk8.0Msun0.1Spin40Inc.fits'

with fits.open(file) as hdul:
    hdu = hdul[0].data
rotatedhdu = rotate(hdu, 40, reshape=False)

plt.imshow(rotatedhdu)
plt.show()


















