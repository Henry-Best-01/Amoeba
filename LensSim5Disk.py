'''
This makes the convolution between a sim5 disk and a magnification map.
It pulls some light curves and displays them.
'''
import numpy as np
import QuasarModelFunctions as QMF
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2
from astropy import constants as const
from astropy import units as u
from astropy.nddata import block_reduce
from astropy.io import fits


inc_ang = 10
mexponentquasar = 8.7
spin=0.7
mapfile = '/Users/henrybest/PythonStuff/LensingMaps/SampleMagMaps/map_1/map.fits'
GRdisk = '/Users/henrybest/PythonStuff/DiskImages/Sim5Disks/Sim5ThinDisk'+str(mexponentquasar)+'Msun0Spin'+str(inc_ang)+'Inc.fits'
GRdisk2 = '/Users/henrybest/PythonStuff/DiskImages/Sim5Disks/Sim5ThinDisk'+str(mexponentquasar)+'Msun'+str(spin)+'Spin'+str(inc_ang)+'Inc.fits'


zlens = 0.6
zquasar = 2.1
mlens = ((1))*const.M_sun.to(u.kg)

with fits.open(mapfile) as f:
    MagMap2d = f[0].data

convo, pxsize = QMF.ConvolveSim5Map(MagMap2d, GRdisk, zlens = zlens, zquasar = zquasar, mquasarexponent=mexponentquasar,
                            mlens = mlens, verbose = True)
convo2, pxsize2 = QMF.ConvolveSim5Map(MagMap2d, GRdisk2, zlens = zlens, zquasar = zquasar, mquasarexponent=mexponentquasar,
                            mlens = mlens, verbose = True)

lc1, tr1 = QMF.PullRandLC(convo, pxsize, 700, 20, returntrack = True)
lc2, tr2 = QMF.PullRandLC(convo, pxsize, 700, 20, returntrack = True)
lc3, tr3 = QMF.PullRandLC(convo, pxsize, 700, 20, returntrack = True)
lc4, tr4 = QMF.PullRandLC(convo, pxsize, 700, 20, returntrack = True)

slc1 = convo2[tr1]
slc2 = convo2[tr2]
slc3 = convo2[tr3]
slc4 = convo2[tr4]


t = np.linspace(0, 20, len(lc1))

fig, ax = plt.subplots(2, sharex='all')
ax[0].set_title("Spin 0 Case Light Curves")
ax[1].set_title("Spin Param = "+str(spin))
supertitle = "Sim5 Light Curves: Mass = $10^{"+str(mexponentquasar)+r"}$ $M_{\odot}$, Inclination Angle = "+str(inc_ang)+r"$\degree$"

fig.suptitle(supertitle)
fig.supxlabel("Time [years, at 700 km/s]")
fig.supylabel("Relative Brightness")

ax[0].plot(t, (lc1 - min(lc1)) / (max(lc1) - min(lc1)))
ax[0].plot(t, (lc2 - min(lc2)) / (max(lc2) - min(lc2)))
ax[0].plot(t, (lc3 - min(lc3)) / (max(lc3) - min(lc3)))
ax[0].plot(t, (lc4 - min(lc4)) / (max(lc4) - min(lc4)))

ax[1].plot(t, (slc1 - min(slc1)) / (max(slc1) - min(slc1)))
ax[1].plot(t, (slc2 - min(slc2)) / (max(slc2) - min(slc2)))
ax[1].plot(t, (slc3 - min(slc3)) / (max(slc3) - min(slc3)))
ax[1].plot(t, (slc4 - min(slc4)) / (max(slc4) - min(slc4)))

plt.show()






















