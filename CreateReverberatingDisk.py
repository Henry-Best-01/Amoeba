'''
This script aims to create a model for a reverberating disk using the lamppost
geometry. The scale size of the accretion disk follows lambda-size prescription

I use an input lamppost "flickering" of the random walk model.

This gets ADDED to the microlensing lightcurve of the non-reverberating disk.
'''

import numpy as np
import matplotlib.pyplot as plt
import QuasarModelFunctions as QMF
from astropy.io import fits
import numpy.random as random
from scipy.fft import fft, fftfreq, ifft
from scipy import signal
from scipy.interpolate import UnivariateSpline


steps = 1200
mlabel='8.5'

file = "/Users/henrybest/PythonStuff/DiskImages/DataSetBand300res/ImageFile"
f1 = 'log'+mlabel+'0.1545gband300res.disk'

mass = 10**float(mlabel) * 2 * 10**30


incangle3 = 45

diskres = 300
geounits = 4000
init_wavelength = 464  #g-band midpoint
compared_wavelength = 658 #z-band midpoint (900), r-band midpoint (658), i-band midpoint (750)
z_wavelength = 900
scale_exponent = 4/3 #Thin disk = 4/3
steptimescale = 60*60  # 1 step = 1 light hour or 1 light day
timeunits = 'hours'

flickering = QMF.DampedWalk(np.ones(steps), dampingfactor = 0.01)

hdug, hdur, _ = QMF.CreateReverbDisk(file+f1, steps, incangle3, mass = mass,
                     diskres = diskres, geounits = 4000, init_wavelength=464, compared_wavelength = 658,
                     scale_exponent = scale_exponent, steptimescale = steptimescale,
                     inputsignal = True, illumination = flickering, returnscalechange=False)

_, hdui, _ = QMF.CreateReverbDisk(file+f1, steps, incangle3, mass = mass,
                     diskres = diskres, geounits = 4000, init_wavelength=464, compared_wavelength = 750,
                     scale_exponent = scale_exponent, steptimescale = steptimescale,
                     inputsignal = True, illumination = flickering, returnscalechange=False)

_, hduz, _ = QMF.CreateReverbDisk(file+f1, steps, incangle3, mass = mass,
                     diskres = diskres, geounits = 4000, init_wavelength=464, compared_wavelength = 900,
                     scale_exponent = scale_exponent, steptimescale = steptimescale,
                     inputsignal = True, illumination = flickering, returnscalechange=False)





lc1 = np.sum(hdug, axis=(0, 1))
lc2 = np.sum(hdur, axis=(0, 1))
lc3 = np.sum(hdui, axis=(0, 1))
lc4 = np.sum(hduz, axis=(0, 1))


fft1 = fft(lc1)
fft2 = fft(lc2)
fft3 = fft(lc3)
fft4 = fft(lc4)


freq1 = fftfreq(lc1.size)
lags = signal.correlation_lags(len(flickering), len(lc1))


corr1 = signal.correlate(lc1-np.average(lc1), flickering-np.average(flickering))
scorr1 = signal.correlate(lc1-np.average(lc1), lc1-np.average(lc1))
corr1 /= np.max(corr1)
scorr1 /= np.max(scorr1)
corr2 = signal.correlate(lc2-np.average(lc2), flickering-np.average(flickering))
scorr2 = signal.correlate(lc2-np.average(lc2), lc2-np.average(lc2))
corr2 /= np.max(corr2)
scorr2 /= np.max(scorr2)
corr3 = signal.correlate(lc3-np.average(lc3), flickering-np.average(flickering))
scorr3 = signal.correlate(lc3-np.average(lc3), lc3-np.average(lc3))
corr3 /= np.max(corr3)
scorr3 /= np.max(scorr3)
corr4 = signal.correlate(lc4-np.average(lc4), flickering-np.average(flickering))
scorr4 = signal.correlate(lc4-np.average(lc4), lc4-np.average(lc4))
corr4 /= np.max(corr4)
scorr4 /= np.max(scorr4)



spline = UnivariateSpline(lags, scorr1 - np.max(scorr1)/2, s=0)
r1, r2 = spline.roots()
corrpeak = abs(np.argmax(corr1)-steps)
FWHM = int(abs(r1-r2)*10)/10



fig, ax = plt.subplots(4)
ax[0].plot(flickering/np.average(flickering), c = 'black', label="input signal")
ax[0].set_title(r"Reverberation of Accretion Disks in Lamp-Post Geometry With $log_{10}$ ( $\frac{M}{M_{\odot}}$ ) = "+mlabel)
ax[0].plot(lc1/np.average(lc1), label="45$\degree$, g' band")
ax[0].plot(lc2/np.average(lc2), label="45$\degree$, r' band")
ax[0].plot(lc1/np.average(lc3), label="45$\degree$, i' band")
ax[0].plot(lc2/np.average(lc4), label="45$\degree$, z' band")


ax[0].set_xlabel("Time ["+timeunits+"]")
ax[0].set_ylabel("Relative Observed Brightness")

ax[1].plot(np.log(freq1), np.log(fft1))
ax[1].plot(np.log(freq1), np.log(fft2))
ax[1].plot(np.log(freq1), np.log(fft3))
ax[1].plot(np.log(freq1), np.log(fft4))


ax[1].set_title("Power Spectrum of Reverberating Disks at Various $\Theta$$_{inc}$")
ax[1].set_xlabel("Log Frequency ["+timeunits+"$^{-1}$]")
ax[1].set_ylabel("Log Power")


ax[2].plot(lags, corr1)
ax[2].plot(lags, corr2)
ax[2].plot(lags, corr3)
ax[2].plot(lags, corr4)


ax[2].set_title("Cross-correlation. Peak = "+str(corrpeak)+" "+timeunits)
ax[2].set_xlabel("Time lags ["+timeunits+"]")
ax[2].set_ylabel("Relative Cross-correlation")
ax[2].plot([np.argmax(corr1)-steps, np.argmax(corr1)-steps], [np.max(corr1), np.min(corr1)], c='black')


ax[3].plot(lags, scorr1)
ax[3].plot(lags, scorr2)
ax[3].plot(lags, scorr3)
ax[3].plot(lags, scorr4)
ax[3].set_title("Auto-correlation of Reverberation. FWHM = "+str(FWHM))
ax[3].set_xlabel("Time lags ["+timeunits+"]")
ax[3].set_ylabel("Relative Auto-correlation")
ax[3].plot([r1, r2], [0.5, 0.5], c='black')

ax[0].legend()
           
print(np.shape(hdug))

savedata = fits.PrimaryHDU(hdug)
hdul = fits.HDUList([savedata])
hdul.writeto('ReverbDiskgband8.5M45deg.fits')



plt.show()





'''        
fig, ax = plt.subplots()
ax.contourf(delaymap1 * hdu[:, :, 0])
ax.set_aspect(1)
plt.show()
'''
        

## Below is the original code which got transfered into QMF.CreateReverbDisk:
'''
disk_size_ratio = (compared_wavelength/init_wavelength)**scale_exponent
dsr3 = (z_wavelength/init_wavelength)**scale_exponent

dummyhdu = np.empty([int(diskres), int(diskres)])
hdu = np.empty([int(diskres), int(diskres), steps])
ii = 0
with open(file+f2, 'r') as f:
    for line in f:
        line = line.strip()
        columns = line.split()
        dummyhdu[:, ii] = np.asarray(columns, dtype=float)
        ii += 1
for jj in range(steps):
    hdu[:, :, jj] = (dummyhdu != 0)

flickering = QMF.DampedWalk(np.ones(steps)) #This is the input signal for lamppost model

hdu2 = hdu.copy()
hdu3 = hdu.copy()

rstep1 = (0.12) * geounits * QMF.GetGeometricalUnit(mass) / diskres
rstep2 = rstep1 * disk_size_ratio
rstep3 = rstep1 * dsr3
#print(diskres*rstep1/(3e8*60*60), diskres*rstep2/(3e8*60*60)) # This is disks' radii in light-hours

delaymap1 = np.empty([int(diskres), int(diskres)])
delaymap2 = delaymap1.copy()
delaymap3 = delaymap1.copy()
incangle *= np.pi/180

for xx in range(diskres):
    for yy in range(diskres):
        z1 = rstep1 * (diskres/2 - yy) * np.sin(incangle)
        z2 = rstep2 * (diskres/2 - yy) * np.sin(incangle)
        z3 = rstep3 * (diskres/2 - yy) * np.sin(incangle)
        x1 = rstep1 * (diskres/2 - yy) * np.cos(incangle)
        x2 = rstep2 * (diskres/2 - yy) * np.cos(incangle)
        x3 = rstep3 * (diskres/2 - yy) * np.cos(incangle)
        y1 = rstep1 * (diskres/2 - xx)
        y2 = rstep2 * (diskres/2 - xx)
        y3 = rstep3 * (diskres/2 - xx)
        r1 = (x1**2 + y1**2)**0.5
        r2 = (x2**2 + y2**2)**0.5
        r3 = (x3**2 + y3**2)**0.5
        delaymap1[xx, yy] = abs(z1 - (z1**2 + r1**2)**0.5)//steptimescale
        delaymap2[xx, yy] = abs(z2 - (z2**2 + r2**2)**0.5)//steptimescale
        delaymap3[xx, yy] = abs(z3 - (z3**2 + r3**2)**0.5)//steptimescale



for jj in range(steps):
    for xx in range(diskres):
        for yy in range(diskres):
            if hdu[xx, yy, jj] == 1:
                hdu[xx, yy, jj] = flickering[jj - int(delaymap1[xx, yy])]
                hdu2[xx, yy, jj] = flickering[jj - int(delaymap2[xx, yy])]
                hdu3[xx, yy, jj] = flickering[jj - int(delaymap3[xx, yy])]
'''

































