'''
This script aims to compare both Gyoto and Sim5 disks, to make sure they're
relatively compatable and similar
'''
import numpy as np
import QuasarModelFunctions as QMF
from astropy.io import fits
import matplotlib.pyplot as plt




mass = 8.6
inc_ang = 20

sim5file = "/Users/henrybest/PythonStuff/DiskImages/Sim5Disks/Sim5ThinDisk"+str(mass)+"Msun0Spin"+str(inc_ang)+"Inc.fits"
gyotofile = "/Users/henrybest/PythonStuff/DiskImages/DataSetBand300Res/ImageFilelog"+str(mass)+"0.15"+str(inc_ang)+"gband300res.disk"

with fits.open(sim5file) as f:
    sim5disk = f[0].data

gyotodisk = QMF.LoadDiskImage(gyotofile, 300)
dummydisk = gyotodisk[(150-31):(150+31), (150-31):(150+31)]
gyotodisk = dummydisk

gyotodisk/=np.max(gyotodisk)
sim5disk/=np.max(sim5disk)

x = np.linspace(-50, 50, 62)
y = np.linspace(-50, 50, 62)

i = np.linspace(-50, 50, 720)
j = np.linspace(-50, 50, 720)

X, Y = np.meshgrid(x, y, indexing='ij')
I, J = np.meshgrid(i, j)

fig, ax = plt.subplots(3, 2)
ax[0, 0].contourf(X, Y, gyotodisk, 20, cmap='plasma')
ax[0, 1].contourf(I, J, sim5disk, 20, cmap='plasma')
ax[1, 0].plot(x, gyotodisk[:, 31])
ax[1, 1].plot(i, sim5disk[360, :])
ax[2, 0].plot(y, gyotodisk[31, :])
ax[2, 1].plot(j, sim5disk[:, 360])

ax[0, 0].set_aspect(1)
ax[0, 1].set_aspect(1)


fig.suptitle("Comparison Between Gyoto and Sim5 Disk Images")
fig.supxlabel("Distance $[R_{g}]$")
ax[0, 0].set_title("Gyoto Disk")
ax[0, 1].set_title("Sim5 Disk")
ax[1, 0].set_title("Central Profile [x axis]")
ax[2, 0].set_title("Central Profile [y axis]")
ax[1, 1].set_title("Central Profile [x axis]")
ax[2, 1].set_title("Central Profile [y axis]")
ax[0, 0].set_ylabel("Distance $[R_{g}]$")
ax[1, 0].set_ylabel("Relative Amplitude")
ax[2, 0].set_ylabel("Relative Amplitude")


plt.show()












