'''
This script is designed to convolve the accretion disk map with a magnification map, then pull a large number of
light curves from it.
LC is shape [a], where a is the LC index. It's length depends on parameters
tracks is shape [a][b][c], where a is LC index, b is 'x' or 'y' component, and c is time
'''


import QuasarModelFunctions as QMF
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy import constants as const

inc_ang = 10     # degrees
mexponentquasar = 8.5
velocity = 4000  # km/s
time=3653  # days


disk = '/Users/henrybest/PythonStuff/DiskImages/DataSetBand300Res/ImageFilelog'+str(mexponentquasar)+'0.15'+str(inc_ang)+'gband300res.disk'
mapfile = '/Users/henrybest/PythonStuff/LensingMaps/SampleMagMaps/map_1/map.bin'
delaymap = QMF.CreateTimeDelayMap(disk, inc_ang, massquasar = 10**mexponentquasar*const.M_sun.to(u.kg))
maxtime = int(np.max(delaymap) + 1.5)+time
illum = QMF.DampedWalk(np.ones(maxtime))
reverbdisk = QMF.CreateReverbSnapshots(delaymap, time, illumination=illum, massquasar = 10**mexponentquasar*const.M_sun.to(u.kg))


convolution, pixelsize, magmap2d = QMF.ConvolveMap(mapfile, disk, diskres = 300, zlens = 0.4, zquasar = 2.1, mquasarexponent = mexponentquasar, 
                nmapERs = 25, diskfov = 0.12, diskposition = 4000, verbose=True, returnmag2d=True)


ncurves = 3
LC = []
tracks = []
for jj in range(ncurves):
    curve, track = QMF.PullRandLC(convolution, pixelsize, velocity, time/365.25, returntrack=True) #Can pull ~5k/second
    LC.append(curve)
    tracks.append(track)

times = np.arange(time/3, dtype=int) * 3
tracklength = ((track[0][-1] - track[0][0])**2+(track[1][-1] - track[1][0])**2)**0.5  #in pixels
pxstep = tracklength * 3 / time #px per observation step

print(tracklength)

time = np.linspace(0, 10, np.size(LC[0]))
fig, ax = plt.subplots()
ax.set_title("Realizations of 10-Year Microlensing Light Curves")
ax.set_ylabel("Normalized Flux [arb]")
ax.set_xlabel("Time [years]")
plt.plot(time, LC[0]/max(LC[0]))
plt.plot(time, LC[1]/max(LC[1]))
plt.plot(time, LC[2]/max(LC[2]))
plt.show()




#fig1, ax1 = plt.subplots()
#x = np.linspace(0, np.size(convolution, 0), np.size(convolution, 0))
#y = np.linspace(0, np.size(convolution, 1), np.size(convolution, 1))
#X, Y = np.meshgrid(x, y, indexing='ij')



### Plot tracks used ###
#ax1.contourf(X, Y, convolution, 20, cmap='plasma')
#for jj in range(ncurves):
#    ax1.plot((tracks[jj][0][0], tracks[jj][0][-1]), (tracks[jj][1][0], tracks[jj][1][-1]))
#
#plt.show()











