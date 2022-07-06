'''
This file holds all functions required to microlens a quasar image and produce an intrinsic variability model
'''

from numpy import *
from astropy import units as u
from astropy import constants as const
from scipy.integrate import quad


c = const.c                                                     # [m/s]			
G = const.G	                                                # [m^(3) kg ^(-1) s^(-2)]		
sigma = const.sigma_sb                                          # [W m^(-2) K^(-4)
h = const.h                                                     # [J s]		#Planck Const
k = const.k_B                                                   # [J/K]	#Boltzmann Const
M_Proton = const.m_p                                            # [kg]
Thompson_Cross_Section = const.sigma_T                          # [m^2]


def KepVelocity (r, M):
        '''
        This calculates the magnitude of Keplerian Velocity at a distance r, on the Acc. Disk
        r should be in meters
        M should be in solar masses, or a Units.Quantity object
        Returns object Units.Quantity
        '''
       
        if type(M) != u.Quantity:
                M *= const.M_sun.to(u.kg)
        if type(r) != u.Quantity:
                r *= u.m
        if r == 0: return(0)            # We don't want to return NaN
        else:
                return ((G * M.to(u.kg) / r.to(u.m) )**(0.5))

def AddVels (v1 = 0, v2 = 0, v3 = 0, output = 0):
        '''
        This Approximately adds relativistic velocities by converting to gamma factors, then adding together and returning beta.
        If units were not included, m/s units are assumed.
        Output gives final velocity in units of c.
        Only values with gamma greater than threshold are added through a relativistic approximation--if everything is lower than threshold, it adds non-relativistically.
        '''
        from astropy import units as u
        if type(v1) == u.Quantity:
                beta1 = v1.to(u.m/u.s).value / 3e8
        else:
                beta1 = v1 / 3e8
        if type(v2) == u.Quantity:
                beta2 = v2.to(u.m/u.s).value / 3e8
        else:
                beta2 = v2 / 3e8
        if type(v3) == u.Quantity:
                beta3 = v3.to(u.m/u.s).value / 3e8
        else:
                beta3 = v3 / 3e8
        nonrel = False
        gamma1 = (1 - beta1**2)**(-0.5) - 1     # The deviation of each gamma factor from 1 is added in this approx.
        gamma2 = (1 - beta2**2)**(-0.5) - 1     # Splitting velocities into components is not a great stratergy in the first place however.
        gamma3 = (1 - beta3**2)**(-0.5) - 1

        thresh = 0.01                           # Threshold in natural units where we shouldn't add directly.
        
        if gamma1 > thresh and gamma2 > thresh and gamma3 > thresh:
                gamma = 1 + gamma1 + gamma2 + gamma3
        elif gamma1 > thresh and gamma2 > thresh and gamma3 <= thresh:
                gamma = 1 + gamma1 + gamma2
        elif gamma1 > thresh and gamma2 <= thresh and gamma3 > thresh:
                gamma = 1 + gamma1 + gamma3
        elif gamma1 > thresh and gamma2 <= thresh and gamma3 <= thresh:
                gamma = 1 + gamma1
        elif gamma1 <= thresh and gamma2 > thresh and gamma3 > thresh:
                gamma =1 + gamma2 + gamma3
        elif gamma1 <= thresh and gamma2 > thresh and gamma3 <= thresh:
                gamma = 1 + gamma2
        elif gamma1 <= thresh and gamma2 <= thresh and gamma3 > thresh:
                gamma = 1 + gamma3
        else:
                gamma = 1
                nonrel = True
        
        beta = (1 - (1/gamma)**2)**0.5
        if nonrel == True:
                beta = beta1 + beta2 + beta3
        assert beta < 1
        return beta

                
def AccDiskTemp (r, R_min, M_acc, M):
        '''
        This returns the temperature of a Thin Disk at some input distance r, with params R_min, M_acc, M.
        Assumed units:
        r = distance from BH (meters)
        R_min = ISCO size (meters)
        M_acc = Accretion rate (M_sun / year)
        M = Mass of central BH (M_sun)
        Alternatively, astropy.units may be used.

        Output is astropy.quantity, units K.
        '''
        if type(r) == u.Quantity:
                r = r.to(u.m)
        else: r *= u.m                                  #Assumed was in meters
        if type(R_min) == u.Quantity:
                R_min = R_min.to(u.m)
        else: R_min *= u.m                              #Assumed was in meters
        if type(M_acc) == u.Quantity:
                M_acc = M_acc.to(u.kg/u.s)
        else: M_acc *= (const.M_sun/u.yr).to(u.kg/u.s)  #Assumed was in M_sun / year
        
        if type(M) == u.Quantity:
                M = M.to(u.kg)
        else: M *= const.M_sun.to(u.kg)                 #Assumed was in M_sun
        if r < R_min:
                return 0*u.K
        else:
                return (((3.0 * G * M * M_acc * (1.0 - (R_min / r)**(0.5))) / (8.0 * pi * sigma * (r**3.0)) )**(0.25)).decompose()  # This calculates the temperature of the fluid on the Acc. Disk 

def AccDiskTempAlpha (r, R_min, M_acc, M, alpha):
        '''
        This similarly calculates the temp. of the Acc Disk, though with an additional alpha parameter to modify dependence.
        Output is dimensionless temperature, since units do not properly cancel without alpha = -3/4.
        '''
        if type(r) == u.Quantity:
                r = r.to(u.m)
        if type(R_min) == u.Quantity:
                R_min = R_min.to(u.m)
        if type(M_acc) == u.Quantity:
                M_acc = M_acc.to(u.kg/u.s)
        else:
                M_acc *= const.M_sun.to(u.kg)/u.yr.to(u.s)      #Assumed was in M_sun / year
                M_acc = M_acc.value
        
        if type(M) == u.Quantity:
                M = M.to(u.kg)
        else:
                M *= const.M_sun.to(u.kg)                       #Assumed was in M_sun
                M = M.value
        if r < R_min:
                return 0
        
        return( ( (3.0 * G * M * M_acc * (1.0 - (R_min / r)**(0.5))) / (8.0 * pi * sigma) )**(0.25)).decompose().value * (r**alpha)


def PlanckLaw (T, lam):
        '''
        I plan to pass in lam in units of [nm]. Otherwise, attach the units and it will convert.
        '''
        if type(T) != u.Quantity:        
                T *= u.K
        if type(lam) != u.Quantity:
                lam *= u.nm
        
        return ((2.0 * h * c**(2.0) * (lam.to(u.m))**(-5.0) * ((e**(h * c / (lam.to(u.m) * k * T)).decompose().value - 1.0)**(-1.0))).to(u.W/(u.m**3)))		# This will return the Planck Law wavelength function at the temperature input

def GetGeometricUnit(mass):
        '''
        This function simply returns what the length (in meters) of a geometric unit is for a given mass (in kg)
        '''
        if type(mass) != u.Quantity:
                mass *= u.kg
        return (G * mass / c**2).decompose().value
        

def RelativisticBeaming(speed, angle):
        '''
        This function estimates the relativistic beaming effect from an object moving with speed and angle.
        The speed does not have a direction requirement (ie: no need to take a parallel component), however
        the angle is measured from the direction moving away from the observer. Therefore, anything traveling away
        will have angle ~ 0 (or 2pi), and anything moving towards the observer will have angle ~pi.

        speed is measured in units of speed of light (0 to 1)
        angle is measured in radians
        '''
        
        if speed > 1:
                speed *= u.m/u.s
                print("Superluminous speed detected, assuming input was in m/s")
                speed = (speed / c).value
        if angle > 6.3:
                angle *= np.pi/180
                print("Angle greater than 2pi detected, assuming input was in degrees")

        beta = speed
        dPhi = 0.00001

        if angle >= 0 and angle <= pi/2:                                        # This is used to calculate the angle w/r to the beaming angle
                int_ang_1= arccos( (beta - cos(pi - angle - dPhi)) / (beta * cos(pi-angle-dPhi) - 1) )
                int_ang_2 = arccos( (beta - cos(pi - angle)) / (beta * cos(pi-angle) - 1) )
        if angle > pi/2 and angle <= pi:
                int_ang_1= arccos( (beta - cos(pi - angle + dPhi)) / (beta * cos(pi-angle+dPhi) - 1) )
                int_ang_2 = arccos( (beta - cos(pi - angle)) / (beta * cos(pi-angle) - 1) )
        if angle > pi and angle <= 3*pi/2:
                int_ang_1=  arccos( (beta - cos(pi - angle - dPhi)) / (beta * cos(pi-angle-dPhi) - 1) )
                int_ang_2 =  arccos( (beta - cos(pi - angle)) / (beta * cos(pi-angle) - 1) )
        if angle > 3*pi/2 and angle <= 2*pi:
                int_ang_1=  arccos( (beta - cos(pi - angle + dPhi)) / (beta * cos(pi-angle+dPhi) - 1) )
                int_ang_2 =  arccos( (beta - cos(pi - angle)) / (beta * cos(pi-angle) - 1) )
                                        
        value = abs(abs(int_ang_1) - abs(int_ang_2))/ dPhi  # This calculates the phi angle in the acc. disk's frame of reference, which gets beamed into our frame of reference
        return(value)



def ConvertMagMap(MagMap):
        '''
        The aim of this is to convert the 1-dim form of the lensing maps into a 2-dim image.
        Once it's done, the 2d map can (and should) be saved as a fits file for future use.
        Assumes a square map is input!
        '''
        res = int(size(MagMap)**0.5)
        MapXY = zeros([res, res])
        for i in range(res):
                for j in range(res):
                        MapXY[i, j] = MagMap[res*i + j]

        return(MapXY)


def ReadThroughput(file):
        '''
        This reads filter throughput files and outputs them
        '''
        
        f = open(file)
        g = fromfile(f, dtype = float, count = -1, sep = ' ')
        output = zeros([int(size(g)/2), 2])
        for i in range(size(g)):
                if i % 2 == 0: output[int(i/2), 0] = g[i]                      # output [i, 0] is the wavelength value
                elif i % 2 == 1: output[int(i/2), 1] = g[i]                    # output [i, 1] is the throughput value (on a scale of 0 to 1)
                else: return(null)
        f.close()
        return(output)       



def CalculateAverageThroughputWavelength(throughput):
        '''
        This operation will weight each wavelength in the throughput by its value, to find the best single wavelength
        that represents it. It uses simple weighting to calculate this.
        '''
        for i in range(size(throughput[:, 0])):
                total += throughput[i, 0] * throughput[i, 1]
        value = total / size(throughput[:, 0])
        return(value)


def AngDiameterDistance(z, Om0=0.3, OmL=0.7):
        '''
        This funciton takes in a redshift value of z, and calculates the angular diameter distance. This is given as the
        output. This assumes LCDM model.
        '''
        multiplier = (9.26* 10 **25) * (10/7) * (1 / (1 + z))                   # This need not be integrated over
        integrand = lambda z_p: ( Om0 * (1 + z_p)**(3.0) + OmL )**(-0.5)        # This must be integrated over
        integral, err = quad(integrand, 0, z)
        value = multiplier * integral * u.m
        return(value)


def AngDiameterDistanceDifference(z1, z2, Om0=0.3, OmL=0.7):
        '''
        This function takes in 2 redshifts, designed to be z1 = redshift (lens) and z2 = redshift (source). It then
        integrates the ang. diameter distance between the two. This assumes LCDM model.
        '''
        assert z1 < z2
        multiplier = (9.26* 10 **25) * (10/7) * (1 / (1 + z2))
        integrand = lambda z_p: ( Om0 * (1 + z_p)**(3.0) + OmL )**(-0.5)        # This must be integrated over
        integral1, err1 = quad(integrand, 0, z1)
        integral2, err2 = quad(integrand, 0, z2)
        value = multiplier * (integral2 - integral1) * u.m
        return(value)


def CalculateLuminosityDistance(z, Om0=0.3, OmL=0.7):
        '''
        This calculates the luminosity distance using the AngdiameterDistance formula above for flat lam-CDM model
        '''
        return((1 + z)**2 * AngDiameterDistance(z, Om0, OmL))


def CalcEinsteinRadius (z1, z2, M_lens=((1)) * const.M_sun.to(u.kg), Om0=0.3, OmL=0.7):
        '''
        This function takes in values of z_lens and z_source (not simply by finding 
        the difference of the two! See AngDiameterDistanceDifference function above!). The output is the
        Einstein radius of the lens, in radians. This assumes LCDM model.
        '''
        D_lens = AngDiameterDistance(z1, Om0, OmL)
        D_source = AngDiameterDistance(z2, Om0, OmL)
        D_LS = AngDiameterDistanceDifference(z1, z2, Om0, OmL)
        value = ((( 4 * G * M_lens / c**2) * D_LS / (D_lens * D_source))**(0.5)).value
        return(value)


def LoadDiskImage(diskfile, diskres):
        '''
        As writen, this loads a disk image file from a .disk file.
        '''
        import numpy as np
        hdu = empty([int(diskres), int(diskres)])
        ii = 0
        with open(diskfile, 'r') as f:
            for line in f:
                line = line.strip()
                columns = line.split()
                hdu[:, ii] = np.asarray(columns, dtype=float)
                ii += 1
        return(hdu)


def TimeDependentLCGenerator(MagMap, disk, ncurves, vtrans, time, zlens = 0.5, zquasar = 2.1, mquasarexponent = 8.0, mlens = 1.0*const.M_sun.to(u.kg),
                             nmapERs = 25, numGRs = 100, diskfov = 0.12, diskposition = 4000, diskres = 300, sim5 = True, rotation=False, verbose=False):
        '''
        This aims to create ncurves light curves using a time dependent disk. It essentially takes into consideration how many disk images are fed
        in, and assigns each slice a portion of the overall light curves. The convolutions must be discarded at each time step, or else too much
        memory will be used to store each convolution.
        The tracks are calculated in the first convolution, and maintained throughout.
        Rotation allows for a simple image rotation to be performed, input in degrees.
        For Gyoto parameters, insert diskfov (field of view on disk image) and diskposition (how many R_g's the camera is from the source). Change sim5 = True to sim5 = False.
        For Sim5 parameters, insert numGRs (number of gravitational radii the disk is calculated to). Keep sim5 = True.
        '''
        import numpy as np
        import QuasarModelFunctions as QMF
        from scipy.fft import fft2, ifft2
        from astropy import constants as const
        from astropy import units as u
        from astropy.io import fits

        if verbose==True: print('Magnification Map has Size =', np.shape(MagMap))

        if disk.ndim == 3:
                nconvos = np.size(disk, 2)
        elif disk.ndim == 2:
                nconvos = 1
        else:
                print("Invalid Disk Dimensions. Please pass in a 2-D grid for non-time dependent convolutions (x, y), or 3-D grid for time dependent convolutions (x, y, t)")
                return

        if sim5==True:
                for xxx in range(nconvos):
                        convo, bivarspline, pxsize = QMF.ConvolveSim5Map(MagMap, disk[:, :, xxx], zlens = zlens, zquasar = zquasar, mquasarexponent = mquasarexponent, mlens = mlens,
                                                            nmapERs = nmapERs, numGRs = numGRs, rotation=rotation, verbose=verbose)
                        if xxx == 0:
                                LCs = []
                                tracks = []
                                for nn in range(ncurves):
                                        LC, track = QMF.PullRandLC(convo.real, pxsize, vtrans, time, returntrack=True, bivarspline = bivarspline)  # Get initial LCs and tracks based on input parameters
                                        LCs.append(LC)
                                        tracks.append(track)
                                lencurve = len(LC)
                                pointsperslice = lencurve/nconvos
                        else:
                                for nn in range(ncurves):
                                        LCs[nn][int(xxx*pointsperslice):] = bivarspline(tracks[nn][0][int(xxx*pointsperslice):], tracks[nn][1][int(xxx*pointsperslice):], grid=False)

        else:
                for xxx in range(nconvos):
                        convo, bivarspline, pxsize = QMF.ConvolveMap(MagMap, disk[:, :, xxx], diskres = diskres, zlens = zlens, zquasar = zquasar, mquasarexponent = mquasarexponent, mlens = mlens,
                                                        nmapERs = nmapERs, diskfov = diskfov, diskposition = diskposition, rotation=rotation, verbose=verbose)
                        if xxx == 0:
                                LCs = []
                                tracks = []
                                for nn in range(ncurves):
                                        LC, track = QMF.PullRandLC(convo.real, pxsize, vtrans, time, returntrack=True, bivarspline = bivarspline)  # Get initial LCs and tracks based on input parameters
                                        LCs.append(LC)
                                        tracks.append(track)
                                lencurve = len(LC)
                                pointsperslice = lencurve/nconvos
                        else:
                                for nn in range(ncurves):
                                        LCs[nn][int(xxx*pointsperslice):] = bivarspline(tracks[nn][0][int(xxx*pointsperslice):], tracks[nn][1][int(xxx*pointsperslice):], grid=False)
        return (LCs, tracks)
        

def ConvolveSim5Map(MagMap, disk, zlens = 0.5, zquasar = 2.1, mquasarexponent = 8.0, mlens = 1.0*const.M_sun.to(u.kg),
                nmapERs = 25, numGRs = 100, rotation=False, verbose=False, returnmag2d=False, returnbivarspline=True):
        '''
        This makes the convolution between a Sim5 disk and a magnification map. The difference is we physically know the screen size
        in physical units, as opposed to the field of view calculation required for GYOTO disks.
        
        '''
        import numpy as np
        import QuasarModelFunctions as QMF
        import matplotlib.pyplot as plt
        from scipy.fft import fft2, ifft2
        from scipy.ndimage import rotate
        from scipy.interpolate import RectBivariateSpline
        from astropy import constants as const
        from astropy import units as u
        from skimage.transform import rescale 
        from astropy.io import fits
        if type(MagMap) == str:
                with open(MagMap, 'rb') as f:
                        dummymap = np.fromfile(f, 'i', count=-1, sep='')
                        MagMap = dummymap
        if verbose==True: print('Magnification Map Opened. Shape =', np.shape(MagMap))
        if type(disk) == str:
                with fits.open(disk) as f:
                        hdu = f[0].data
                        disk = hdu
        diskres = np.size(disk, 0)
        if verbose==True: print('Disk Opened. Size =', np.shape(disk))
        if type(rotation) != bool:
                newimg = rotate(disk, rotation, axes=(0, 1), reshape=False)
                disk = newimg
                if verbose==True: print("Disk Rotated")
        if MagMap.ndim == 2:
                MagMap2d = MagMap
        else:                   
                MagMap2d = QMF.ConvertMagMap(MagMap)
                if verbose==True: print('Magnification Map Changed. Shape =', np.shape(MagMap2d))
        mquasar = 10**mquasarexponent*const.M_sun.to(u.kg)
        diskpxsize = numGRs * QMF.GetGeometricUnit(mquasar)*u.m / diskres
        pixelsize = QMF.CalcEinsteinRadius(zlens, zquasar, M_lens = mlens) * QMF.AngDiameterDistance(zquasar) * nmapERs / np.size(MagMap2d, 0)
        if verbose==True: print('A pixel on the mag map is', pixelsize)
        if verbose==True: print('A pixel on the disk map is', diskpxsize)

        pixratio = diskpxsize.value/pixelsize.value
        dummydiskimg = rescale(disk, pixratio)
        disk = dummydiskimg
        if verbose==True: print("The disk's shape is now:", np.shape(disk))    
        
        dummymap = np.zeros(np.shape(MagMap2d))
        dummymap[:np.size(disk, 0), :np.size(disk, 1)] = disk
        convolution = ifft2(fft2(dummymap) * fft2(MagMap2d))
        output = convolution
        if returnbivarspline==True:
                x = np.linspace(0, np.size(convolution, 0), np.size(convolution, 0), dtype='int')
                y = np.linspace(0, np.size(convolution, 1), np.size(convolution, 1), dtype='int')
                output = RectBivariateSpline(x, y, convolution.real)

                if verbose==True: print("Convolution Completed")
                
                if returnmag2d==True:
                        return convolution, output, pixelsize, MagMap2d
                return convolution, output, pixelsize
                        
        
        if verbose==True: print("Convolution Completed")
        
        if returnmag2d==True:
                return output, pixelsize, MagMap2d
        return output, pixelsize


def ConvolveMap(MagMap, disk, diskres = 300, zlens = 0.5, zquasar = 2.1, mquasarexponent = 8.0, mlens = 1.0*const.M_sun.to(u.kg),
                nmapERs = 25, diskfov = 0.12, diskposition = 4000, rotation=False, verbose=False, returnmag2d=False, returnbivarspline=True):
        '''
        This function returns the convolution of a magnification map and a projected disk image
        mquasarexponent is input as log10(M_quasar/M_sun)
        nmapERs is how many Einstein Radii the magnification map is
        diskfov and diskposition are related to where the observer is positioned in creating the disk image
        diskfov is the angle observed in radians
        disk position is how many geometric units away the observer is
        this function also returns the pixel size, needed for calculating light curves with some given velocity
        '''
        import numpy as np
        import QuasarModelFunctions as QMF
        import matplotlib.pyplot as plt
        from scipy.fft import fft2, ifft2
        from scipy.ndimage import rotate
        from scipy.interpolate import RectBivariateSpline
        from astropy import constants as const
        from astropy import units as u
        from skimage.transform import rescale 
        if type(MagMap) == str:
                with open(MagMap, 'rb') as f:
                        dummymap = np.fromfile(f, 'i', count=-1, sep='')
                        MagMap = dummymap
        if verbose==True: print('Magnification Map Opened. Shape =', np.shape(MagMap))
        if type(disk) == str:
                dummydisk = QMF.LoadDiskImage(disk, diskres)
                disk = dummydisk
        else:
                diskres = np.size(disk, 0)
        if verbose==True: print('Disk Opened. Shape =', np.shape(disk))

        if type(rotation) != bool:
                newimg = rotate(disk, rotation, axes=(0, 1), reshape=False)
                disk = newimg
                if verbose==True: print("Disk Rotated")

        if MagMap.ndim == 2:
                MagMap2d = MagMap
        else:                   
                MagMap2d = QMF.ConvertMagMap(MagMap)
                if verbose==True: print('Magnification Map Changed. Shape =', np.shape(MagMap2d))
        mquasar = 10**mquasarexponent*const.M_sun.to(u.kg)
        pixelsize = QMF.CalcEinsteinRadius(zlens, zquasar, M_lens = mlens) * QMF.AngDiameterDistance(zquasar) * nmapERs / np.size(MagMap2d, 0)
        diskpxsize = diskfov * diskposition * QMF.GetGeometricUnit(mquasar)*u.m / np.size(disk, 0)
        if verbose==True: print('A pixel on the mag map is', pixelsize)
        if verbose==True: print('A pixel on the disk map is', diskpxsize)

        pixratio = diskpxsize.value/pixelsize.value
        dummydiskimg = rescale(disk, pixratio)
        disk = dummydiskimg
        if verbose==True: print("The disk's shape is now:", np.shape(disk))    

        dummymap = np.zeros(np.shape(MagMap2d))
        dummymap[:np.size(disk, 0), :np.size(disk, 1)] = disk
        convolution = ifft2(fft2(dummymap) * fft2(MagMap2d))
        output = convolution
        if returnbivarspline==True:
                x = np.linspace(0, np.size(convolution, 0), np.size(convolution, 0), dtype='int')
                y = np.linspace(0, np.size(convolution, 1), np.size(convolution, 1), dtype='int')
                output = RectBivariateSpline(x, y, convolution.real)

                if verbose==True: print("Convolution Completed")
                
                if returnmag2d==True:
                        return convolution, output, pixelsize, MagMap2d
                return convolution, output, pixelsize
                        
        
        if verbose==True: print("Convolution Completed")
        
        if returnmag2d==True:
                return output, pixelsize, MagMap2d
        return output, pixelsize
                            
def PullLightCurve(convolution, pixelsize, vtrans, time, startposition = (1000, 1000), angle = 0, bivarspline = False):
        '''
        This function takes in a convolution from above and takes a particular light curve from it.
        vtrans may be inserted as a u.Quantity, but if not it is assumed to be in km/s.
        time is how long the light curve lasts in physical time units. May be u.Quantity, else assumed in years.
        startposition is the starting point on the convolution (in pixels). angle is the angle to travel after in degrees (from x-axis).
        If the path falls off the convolution, an error will occur.
        If a random light curve is desired, use "PullRandLC" below.
        '''
        from astropy import constants as const
        from astropy import units as u
        import numpy as np

        if type(vtrans) == u.Quantity:
                vtrans = vtrans.to(u.m/u.s) 
        else:
                vtrans *= u.km.to(u.m)*u.m/u.s
        if type(time) == u.Quantity:
                time = time.to(u.s)
        else:
                time *= u.yr.to(u.s) * u.s
        length_traversed = vtrans * time
        px_traversed = int(length_traversed / pixelsize + 0.5)
        xtraversed = px_traversed * np.cos(angle * np.pi/180)
        ytraversed = px_traversed * np.sin(angle * np.pi/180)
        assert startposition[0]+xtraversed > 0
        assert startposition[0]+xtraversed < np.size(convolution, 0)
        assert startposition[1]+ytraversed > 0
        assert startposition[1]+ytraversed < np.size(convolution, 1)


        if type(bivarspline) == bool:
                xpositions = np.linspace(startposition[0], startposition[0]+xtraversed, px_traversed, dtype='int')
                ypositions = np.linspace(startposition[1], startposition[1]+ytraversed, px_traversed, dtype='int')                
                light_curve = convolution[xpositions, ypositions]
        else:
                xpositions = np.linspace(startposition[0], startposition[0]+xtraversed, px_traversed)
                ypositions = np.linspace(startposition[1], startposition[1]+ytraversed, px_traversed)
                light_curve = convolution(xpositions, ypositions)
                
        return light_curve

        
def PullRandLC(convolution, pixelsize, vtrans, time, bivarspline = False, returntrack=False):
        '''
        Almost identical to PullLightCurve function above, but this time a random curve is drawn instead of
        a specific one.
        Returning the track will allow both plotting tracks on the magnification map and also comparing different
        models along identical tracks.
        '''
        from astropy import constants as const
        from astropy import units as u
        import numpy as np
        from numpy.random import rand 

        if type(vtrans) == u.Quantity:
                vtrans = vtrans.to(u.m/u.s) 
        else:
                vtrans *= u.km.to(u.m)*u.m/u.s
        if type(time) == u.Quantity:
                time = time.to(u.s)
        else:
                time *= u.yr.to(u.s) * u.s
        length_traversed = vtrans * time
        px_traversed = int(length_traversed / pixelsize + 0.5)

        xbounds = [abs(px_traversed), np.size(convolution, 0)-abs(px_traversed)]
        ybounds = [abs(px_traversed), np.size(convolution, 1)-abs(px_traversed)]

        xstart = xbounds[0] + rand() * (xbounds[1] - xbounds[0])
        ystart = ybounds[0] + rand() * (ybounds[1] - ybounds[0])
        startposition = [xstart, ystart]
        angle = rand() * 2*np.pi

        xtraversed = px_traversed * np.cos(angle)
        ytraversed = px_traversed * np.sin(angle)

        if type(bivarspline) == bool:
                xpositions = np.linspace(startposition[0], startposition[0]+xtraversed, px_traversed, dtype='int')
                ypositions = np.linspace(startposition[1], startposition[1]+ytraversed, px_traversed, dtype='int')                
                light_curve = convolution[xpositions, ypositions]
        else:
                xpositions = np.linspace(startposition[0], startposition[0]+xtraversed, px_traversed)
                ypositions = np.linspace(startposition[1], startposition[1]+ytraversed, px_traversed)
                light_curve = bivarspline(xpositions, ypositions, grid=False)
        track = [xpositions, ypositions]


        if returntrack==True:
                return light_curve, track
        else:
                return light_curve
        

def AddSaltPepperNoise(XYMap, strength=10):
        '''
        This function adds random values up to the input strength on the XYMap. For each pixel, a gaussian integer
        error up to strength value is added, and negative values are treated as 0 additional noise.
        '''
        import numpy as np
        dummymap = XYMap.copy()
        noise = random.normal(scale=strength, size=(np.size(XYMap, 0), np.size(XYMap, 1)))
        mask = noise > 0
        
        dummymap += noise*mask


        return(dummymap)
                        

def FillInAccDisk(DiskImage):
        '''
        This function fills in an accretion disk's image by linearly interpolating between ISCO edge values.
        This new disk is then returned as the output.
        It starts looping through x-values at a given y value and climbs up until the ISCO is encountered

        '''
        centerindex = int(size(DiskImage, 0)/2)
        DiskOut = DiskImage.copy()
        filledinpix = 0
        for yy in range(size(DiskImage, 1)):
                if sum(DiskOut[:, yy]) > 0:
                        if DiskOut[centerindex, yy] == 0:
                                for xxmin in range(centerindex):
                                        if DiskOut[centerindex - xxmin, yy] > 0:
                                                diskleftindex = centerindex - xxmin
                                                break
                                for xxmax in range(centerindex):
                                        if DiskOut[centerindex + xxmax, yy] > 0:
                                                diskrightindex = centerindex + xxmax
                                                break
                                iscolength = diskrightindex - diskleftindex
                                
                                if iscolength < centerindex:
                                        llist = [DiskOut[diskleftindex, yy], DiskOut[diskleftindex - 1, yy],
                                                 DiskOut[diskleftindex - 2, yy], DiskOut[diskleftindex - 3, yy],
                                                 DiskOut[diskleftindex - 4, yy], DiskOut[diskleftindex - 5, yy]]
                                        rlist = [DiskOut[diskrightindex, yy], DiskOut[diskrightindex + 1, yy],
                                                 DiskOut[diskrightindex + 2, yy], DiskOut[diskrightindex + 3, yy],
                                                 DiskOut[diskrightindex + 4, yy], DiskOut[diskrightindex + 5, yy]]
                                                 
                                                 
                                        diskleft = max(llist)
                                        diskright = max(rlist)
                                        adddiskleftindex = argmax(llist)
                                        adddiskrightindex = argmax(rlist)
                                        
                                        diskleftindex -= adddiskleftindex
                                        diskrightindex += adddiskrightindex
                                        iscolength = diskrightindex - diskleftindex
                                        
                                        difference = diskright - diskleft
                                        for ii in range(iscolength):
                                                value = diskleft + (ii / iscolength) * difference
                                                if DiskOut[diskleftindex + ii, yy] < value:               
                                                        DiskOut[diskleftindex + ii, yy] = value
                                                        filledinpix += 1
        return(DiskOut)

def CreateWindLine(launchrad, launchangle, maxzheight, zslices, characteristicdistance, centralBHmass = 10**8, launchheight = 0, maxvel = 10**6, launchspeed = 0, alpha=1):
        '''
        This creates a simple line of wind, divided up as vertical slabs, assuming conservation of ang
        momentum. This wind line will use a model for poloidal velocity, using a definable alpha parameter.
        Wind holds velocity parameters, while future absorption / emission is defined in AddWindEffects function.
        Launchrad, maxzheight, characteristicdistance should be inserted in meters.
        launchangle should be inserted as a degree angle from tne normal.
        launchspeed can be inserted as m/s, or a quantity value.
        centralBHmass should be inserted in solar masses.
        zslices is an integer number of slices to calculate. Make sure this matches any wind it gets paired with!
        Output will contain an array with values (r_positions, z_positions, v_r, v_phi, v_z, v_pol) in cylindrical geometry
        '''
        import QuasarModelFunctions as QMF
        import numpy as np


        launchangle *= np.pi/180
        
        phi_init_velocity = QMF.KepVelocity(launchrad.to(u.m).value, centralBHmass)
        
        pol_init_velocity = launchspeed
        rad_init_velocity = pol_init_velocity * np.sin(launchangle)
        z_init_velocity   = pol_init_velocity * np.cos(launchangle)
        pol_end_velocity  = maxvel
        init_vels = [phi_init_velocity, pol_init_velocity, rad_init_velocity, z_init_velocity, pol_end_velocity]

        rad_init_pos = launchrad
        z_init_pos   = launchheight
        init_pos = [rad_init_pos, z_init_pos, characteristicdistance, maxzheight]

        for ii in range(len(init_vels)):
                if type(init_vels[ii]) == u.Quantity:
                        dummy = init_vels[ii].to(u.m / u.s).value   # Standardize everything as u.m/u.s, then strip the units
                        init_vels[ii] = dummy
        for ii in range(len(init_pos)):
                if type(init_pos[ii]) == u.Quantity:
                        dummy = init_pos[ii].to(u.m).value          # Sreamline positions to u.m, then strip units
                        init_pos[ii] = dummy

        if zslices == 1:
                zslices = 2 

        z_dist_traveled_slice = (init_pos[3] - init_pos[1]) / (zslices - 1)

        output_streamline = np.zeros([6, zslices])
        for ii in range(zslices):                
                output_streamline[0, ii] = init_pos[0] + np.tan(launchangle) * ii * z_dist_traveled_slice
                output_streamline[1, ii] = init_pos[1] + z_dist_traveled_slice * ii
                poloidal_dist = (output_streamline[0, ii]**2 + output_streamline[1, ii]**2)**0.5
                v_pol = init_vels[1] + (init_vels[4] - init_vels[1]) * ( (poloidal_dist / init_pos[2])**alpha / ((poloidal_dist / init_pos[2])**alpha + 1) )

                output_streamline[2, ii] = v_pol * np.sin(launchangle)
                output_streamline[4, ii] = v_pol * np.cos(launchangle)
                output_streamline[3, ii] = QMF.KepVelocity(output_streamline[0, ii], centralBHmass).value 
                output_streamline[5, ii] = v_pol

                assert(output_streamline[2, ii] < 3e8)
                assert(output_streamline[3, ii] < 3e8)
                assert(output_streamline[4, ii] < 3e8)
                assert(output_streamline[5, ii] < 3e8)
        return(output_streamline)


def CreateWindRegion(sl1, sl2, r_res = 100, z_res = 100, phi_res = 30, centralBHmass=10**8, r0 = 10e15, sigma = 10e7, function=1, power=1):
        '''
        This function takes in two wind lines and outputs the region of space which they bound. This is in preperation
        for sending a reverberation signal through to observe how it appears with time delays.
        Streamlines should be a 6-dim array with values (r_positions, z_positions, v_r, v_phi, v_z, v_pol) in cylindrical geometry.
        phi_res should be input as a number of times which the full 2pi grid will be diced into.
        The grid contains values equal to f(r)/pol_velocity at each position r, z, phi.
        f(r) is some function of distance which determines the emission profile along poloidal distance
        function = 1 or invalid choice leads to function(r) = (r/r0)**p
        function = 2 leads to function(r) = exp(-(r-r0)**2/sigma)
        function = 3 is a step-like function, between r0 and r0+sigma
        r0 is the characteristic distance in function(r)
        sigma is width of gaussian in function 2, or is the length of the tophat in function 3
        The other output values are used to pass into the reverberatiewind function, to keep positional information.

        '''
        import QuasarModelFunctions as QMF
        import numpy as np

        inputvalues = [centralBHmass, r0, sigma]   #Standardize units and keep values
        for lll in range(len(inputvalues)):
                if type(inputvalues[lll]) == u.Quantity:
                        if lll == 0:
                                dummy = inputvalues[lll].to(u.kg) / const.M_sun.to(u.kg)
                                inputvalues[lll] = dummy.value
                        else:
                                dummy = inputvalues[lll].to(u.m)
                                inputvalues[lll] = dummy.value

        if function==1:
                def fun(r):
                        return (r/inputvalues[1])**power
        elif function == 2:
                def fun(r):
                        return exp(-(r - inputvalues[1])**2 / inputvalues[2]**2)
        elif function == 3:
                def fun(r):
                        if r >= inputvalues[1] and r <= inputvalues[1] + inputvalues[2]:
                                return 1
                        else:
                                return 0
        else:
                print("Invalid function choice, reverting to default")
                def fun(r):
                        return (r/r0)**power
        assert (sl1[1, -1] == sl2[1, -1]) # Require that streamlines are of same height
        
        phi_length = (2*np.pi/phi_res)  # We will need to know how much volume is taken up by one 'cell' in this cylindrical space.
                                
        r_max = max(sl1[0, -1], sl2[0, -1])
        r_min = min(sl1[0, 0], sl2[0, 0])
        r_length = (r_max-r_min)/r_res
        z_max = sl1[1, -1]
        z_length = z_max/z_res

        if sl1[0, -1] > sl2[0, -1]:
                slg = sl1.copy()
                sll = sl2.copy()
        elif sl1[0, -1] < sl2[0, -1]:
                slg = sl2.copy()
                sll = sl1.copy()

        outputgrid = np.empty([r_res, z_res, phi_res, 4]) #outputgrid will be a density, where it's modeled by 1/v_pol and r, phi, z velocities
        for ii in range(np.size(outputgrid, 0)):
                for jj in range(np.size(outputgrid, 1)):
                        if ((ii * r_length + r_min) < sl1[0, jj] and ((ii * r_length + r_min) > sl2[0, jj])) or ((ii * r_length + r_min) > sl1[0, jj] and ((ii * r_length + r_min) < sl2[0, jj])):
                                r_greater = slg[0, jj]
                                r_lesser = sll[0, jj]
                                fracgreater = (ii*r_length + r_min - r_lesser)/(r_greater - r_lesser)
                                fraclesser = 1 - fracgreater
                                assert(fraclesser <= 1)
                                assert(fraclesser >= 0)

                                radius = ((ii * r_length + r_min)**2 + (jj * z_length)**2)**0.5
                                              
                                outputgrid[ii, jj, :, 0] = fun(radius) * ((1/sll[5, jj]) * fraclesser + (1/slg[5, jj]) * fracgreater) #Linear interpolate
                                outputgrid[ii, jj, :, 1] = (sll[2, jj]) * fraclesser + (slg[2, jj]) * fracgreater 
                                outputgrid[ii, jj, :, 2] = QMF.KepVelocity((ii*r_length + r_min), inputvalues[0]).value  
                                outputgrid[ii, jj, :, 3] = (sll[4, jj]) * fraclesser + (slg[4, jj]) * fracgreater 
            
        return(outputgrid, r_length, z_length, phi_length, r_min)


def ProjectWind(windgrid, rlen, zlen, philen, rmin, viewingang, xres, SL1, SL2, velocities, geounits = 4000, mass = 10e8 * const.M_sun.to(u.kg), reverberating=False):
        '''
        This function takes a wind region and projects it into a screen which may be used in microlensing simulations.
        windgrid is from QMF.CreateWindRegion, SL1 + SL2 are from QMF.CreateWindLine
        '''
        import numpy as np
        import QuasarModelFunctions as QMF
        
        viewingang *= np.pi/180
        #Start defining the screen
        #Choose the furthest point away from us to start the screen as the min value
        zmin = min(SL1[1, -1] * np.cos(viewingang) - SL1[0, -1] * np.sin(viewingang),
                        SL2[1, -1] * np.cos(viewingang) - SL2[0, -1] * np.sin(viewingang),
                        0)


        xmax = max(SL1[0, -1], SL2[0, -1])
        xmin = -xmax
        ymax = max(SL1[1, -1] * np.sin(viewingang) + SL1[0, -1] * np.cos(viewingang),
                   SL2[1, -1] * np.sin(viewingang) + SL2[0, -1] * np.cos(viewingang))
        ymin = min(SL1[1, -1] * np.sin(viewingang) - SL1[0, -1] * np.cos(viewingang),
                   SL2[1, -1] * np.sin(viewingang) - SL2[0, -1] * np.cos(viewingang),
                   0)

        xstep = 2*xmax / xres
        yres = int(((ymax - ymin) / (xmax - xmin)) * xres)

        ystep = xstep
        boxres = max(xres, yres)
        
        zstep = min(xstep, ystep)
        zsteps = int((abs(zmin) + max(SL1[1, -1] * np.cos(viewingang) + SL1[0, -1] * np.sin(viewingang),
                        SL2[1, -1] * np.cos(viewingang) + SL2[0, -1] * np.sin(viewingang),
                        0)) // zstep)
        zmax = zstep * zsteps

        fov = xstep * boxres / (geounits * QMF.GetGeometricUnit(mass))

        if reverberating == True:
                screen = np.zeros([boxres, boxres, np.size(windgrid, 3), len(velocities)])
        else:
                screen = np.zeros([boxres, boxres]) #CONTINUE (Add los velocity dimension. Carry info to reverb == false output.

        radii = np.linspace(0, SL2[0, -1], int(SL2[0, -1] / rlen)) #Useful to calculate closest arguments
        angles = np.linspace(0, 2*np.pi, int(2*np.pi / philen))
        heights = np.linspace(0, SL2[1, -1], int(SL2[1, -1] / zlen))


        for ii in range(xres):
            xwind = (xmin + ii * xstep)
            for jj in range(yres):
                for zz in range(zsteps):
                    zwind = (zmin + zz * zstep) * np.cos(viewingang) + (ymin + jj * ystep) * np.sin(viewingang)
                    ywind = (ymin + jj * ystep) * np.cos(viewingang) - (zmin + zz * zstep) * np.sin(viewingang)

                    if zwind >= 0 and zwind <= zmax:
                        r, phi = QMF.ConvertToPolar(xwind, ywind)
                        rarg = np.argmin(abs(r - radii))
                        phiarg = np.argmin(abs(phi - angles))
                        zarg = np.argmin(abs(zwind - heights))

                        if zarg >= np.size(windgrid, 1):
                                continue
                        if rarg >= np.size(windgrid, 0):
                                continue
                        if phiarg >= np.size(windgrid, 2):
                                continue


                        if reverberating == True:
                                
                                screen [ii, jj, :, :] += windgrid[rarg, zarg, phiarg, :, :]
                        else:
                                
                                screen[ii, jj] += windgrid[rarg, zarg, phiarg, 0]
        
        return screen, fov


def CreateBLRTransferFunction(BLR, rlen, zlen, philen, inc_ang, xres, SL1, SL2, mass = 10e8 * const.M_sun.to(u.kg), geounits = 4000, units = 'days', return_grid = False):
        '''
        Similarly to CreateTimeDelayMap, this sums up the time delays of the BLR and weights them with the density of the simulated material in order to
        simplify the reprocessing step.
        '''
        import numpy as np
        import QuasarModelFunctions as QMF
        if type(units) == str:
                if units == 'days' or units == 'Days':
                        steptimescale = 3e8*60*60*24
                elif units == 'hours' or units == 'Hours':
                        steptimescale = 3e8*60*60 #units light hours / days
                elif units == 'weeks' or units == 'Weeks':
                        steptimescale = 3e8*60*60*24*7
                
                else:
                        print('Invalid string deteted. Try "days", "hours", "weeks", or an astropy.unit.\nReverting to days.')
                        steptimescale = 3e8*60*60*24
        elif type(units) == astropy.units.core.Unit or type(units) == astropy.units.core.IrreducibleUnit:
                steptimescale = 3e8 * u.s.to(unit)

        viewingang = inc_ang * np.pi/180
        #Start defining the screen
        #Choose the furthest point away from us to start the screen as the min value
        zmin = min(SL1[1, -1] * np.cos(viewingang) - SL1[0, -1] * np.sin(viewingang),
                        SL2[1, -1] * np.cos(viewingang) - SL2[0, -1] * np.sin(viewingang),
                        0)


        xmax = max(SL1[0, -1], SL2[0, -1])
        xmin = -xmax
        ymax = max(SL1[1, -1] * np.sin(viewingang) + SL1[0, -1] * np.cos(viewingang),
                   SL2[1, -1] * np.sin(viewingang) + SL2[0, -1] * np.cos(viewingang))
        ymin = min(SL1[1, -1] * np.sin(viewingang) - SL1[0, -1] * np.cos(viewingang),
                   SL2[1, -1] * np.sin(viewingang) - SL2[0, -1] * np.cos(viewingang),
                   0)

        xstep = 2*xmax / xres
        yres = int(((ymax - ymin) / (xmax - xmin)) * xres)

        ystep = xstep
        boxres = max(xres, yres)
        
        zstep = min(xstep, ystep)
        zsteps = int((abs(zmin) + max(SL1[1, -1] * np.cos(viewingang) + SL1[0, -1] * np.sin(viewingang),
                        SL2[1, -1] * np.cos(viewingang) + SL2[0, -1] * np.sin(viewingang),
                        0)) // zstep)
        zmax = zstep * zsteps

        fov = xstep * boxres / (geounits * QMF.GetGeometricUnit(mass))


        maxdelay = (rlen * np.size(BLR, 0) + zlen * np.size(BLR, 1)) / steptimescale

        tfgrid = np.zeros([xres, yres, int(maxdelay + 1)])

        radii = np.linspace(0, SL2[0, -1], int(SL2[0, -1] / rlen + 1)) #Useful to calculate closest arguments
        angles = np.linspace(0, 2*np.pi, int(2*np.pi / philen + 1))
        heights = np.linspace(0, SL2[1, -1], int(SL2[1, -1] / zlen + 1))

        for ii in range(xres):
            xwind = (xmin + ii * xstep)
            for jj in range(yres):
                for zz in range(zsteps):
                    zwind = (zmin + zz * zstep) * np.cos(viewingang) + (ymin + jj * ystep) * np.sin(viewingang)
                    ywind = (ymin + jj * ystep) * np.cos(viewingang) - (zmin + zz * zstep) * np.sin(viewingang)

                    if zwind >= 0:
                        rp, phi = QMF.ConvertToPolar(xwind, ywind)
                        rarg = np.argmin(abs(rp - radii))
                        phiarg = np.argmin(abs(phi - angles))
                        zarg = np.argmin(abs(zwind - heights))

                        if zarg >= np.size(BLR, 1):
                                continue
                        if rarg >= np.size(BLR, 0):
                                continue
                        if phiarg >= np.size(BLR, 2):
                                continue
                        if ii >= np.size(tfgrid, 0):
                                continue
                        if jj >= np.size(tfgrid, 1):
                                continue
                        _, theta = QMF.ConvertToPolar(zwind, rp)
                        delay = rp * (1 - np.cos(theta*np.sin(phi)+viewingang)) / steptimescale
                        density = BLR[rarg, zarg, phiarg, 0]
                        if int(delay) >= np.size(tfgrid, 2):
                                continue

                        tfgrid[ii, jj, int(delay)] += density
        
        if return_grid == True:
                return tfgrid
        else:
                tf = np.sum(tfgrid, axis = (0, 1))
                return tf
        

def ReverberateWindLine(windgrid, illumination, wavelengths, inc_angle, r_length, z_length, phi_length, r_min,
                        line_wavelength = 500, line_width = 1, timescale = 1, emissionmultiplier = 1,
                        returnpulse = False, returnvelocitygrid = False, returntimedelay = False):
        '''
        This aims to follow a similar idea above, but actually simulating an emission line in the broad line region.
        For now, this assumes a constant fraction of the emitting species throughout the wind (windgrid)
        Illumination is the illuminating light curve, which should vary with time. It should be 2-dim, other dimension being wavelength.
        Wavelengths is the mapping between which wavelengths we are positioned on the wavelength axis of illumination (array
        inc_angle, lengths all have usual definitions
        line_wavelength is the center of the line profile
        line_width is the gaussian width of line profile
        timescale is the ratio between pixels and physical time units simulated
        emissionmultiplier is a multiplier for the strength of the emission line
        THIS IS A WORK IN PROGRESS
        '''
        import QuasarModelFunctions as QMF
        import numpy as np

        if np.ndim(illumination) == 1:
                dummywaves = np.ones([len(wavelengths), len(illumination)]) # If not given explicity, assume all wavelengths in continuum are equal.
                for ii in range(len(illumination)):
                        dummywaves[:, ii] *= illumination[ii]
                illumination = dummywaves

        inc_angle *= np.pi/180                  #Convert to rads
        timescale *= 3e8                        #Units light-something


        assert(np.size(wavelengths) == np.size(illumination, 0)) #Make sure there's a one-to-one mapping of the wavelength labels to the indexes

        outputreverb = illumination.copy()
        timedelay = np.zeros([np.size(windgrid, 0), np.size(windgrid, 1), np.size(windgrid,2)])

        outputreverb[:, :] = 0

        dradial = (r_length**2 + z_length**2)**0.5
        
        if returnpulse == True:                 #For a pulse map to check timescales
                pulsemap = np.zeros([np.size(windgrid, 0), np.size(windgrid, 1), np.size(illumination, 1)])
                
        if returnvelocitygrid == True:          #For LoS velocities
                vgrid = np.zeros([np.size(windgrid, 0), np.size(windgrid, 2), np.size(windgrid, 1)])
        
        for ii in range(np.size(timedelay, 0)):
                for kk in range(np.size(timedelay, 2)):
                        r = (ii * r_length + r_min)
                        x = r * np.cos(kk * phi_length)
                        y = r * np.sin(kk * phi_length)
                        
                        xp = x

                        for jj in range(np.size(timedelay, 1)):
                                                            
                                if windgrid[ii, jj, kk, 0] != 0:

                                        z = jj * z_length

                                        _, theta1 = QMF.ConvertToPolar(r, z) 
                                        _, theta2 = QMF.ConvertToPolar(((ii+1)*r_length + r_min), z)
                                        dtheta=abs(theta1-theta2)


                                        yp = np.cos(inc_angle)*z - np.sin(inc_angle)*y  #In prime coords, positive y faces to us now (previously, y faced away from us)
                                        zp = np.cos(inc_angle)*y + np.sin(inc_angle)*z

                                        rp = (xp**2 + zp**2)**0.5

                                        timedelay[ii, jj, kk] = abs(yp - (yp**2 + rp**2)**0.5)
                                        
                                        betaaway = (1/windgrid[ii, jj, kk, 0]) / 3e8

                                        gammatoward = ((1-((windgrid[ii, jj, kk, 1] * np.sin(inc_angle) * (-np.sin(kk*phi_length)))/3e8)**2)**(-0.5) +
                                                (1-((windgrid[ii, jj, kk, 2] * np.sin(inc_angle) * np.cos(kk*phi_length))/3e8)**2)**(-0.5) +
                                                (1-((windgrid[ii, jj, kk, 3] * np.cos(inc_angle)) / 3e8)**2)**(-0.5))
                                        

                                        betatoward = (1 - (1/gammatoward)**2)**0.5
                                        delay = timedelay[ii, jj, kk]
                                        delayindex = int(delay//(timescale)+0.5)%np.size(illumination, 1) # Discretize and reduce based on chosen timescale

                                        if returnvelocitygrid == True:
                                                vgrid[ii, kk, jj] = betatoward
                                        
                                
                                else:
                                        betaaway = 0
                                        betatoward = 0
                                        delay = 0
                                        delayindex = 0
                                        
                                assert(betaaway<=1)
                                assert(betatoward <= 1)
                                

                                if windgrid[ii, jj, kk, 0] != 0:
                                        illuminatingwavelength = line_wavelength * ( (1 - betaaway) / (1 + betaaway) )**0.5
                                        wavelengthindex = np.argmin(abs(wavelengths - illuminatingwavelength)) # This is the greatest contributing wavelength to the line. Assume only contribution.
                                        emittedwavelength = line_wavelength * ( (1 - betatoward) / (1 + betatoward) )**0.5                                       
                                        for tt in range(np.size(illumination, 1)):
                                                if tt > delayindex:
                                                        addedoutput = illumination[wavelengthindex, int(tt - delayindex)] * emissionmultiplier *windgrid[ii, jj, kk, 0] * np.exp(-(emittedwavelength - wavelengths)**2 / (2 * line_width)) * r_length * z_length/((x**2 + y**2 + z**2)**0.5* dtheta * dradial)

                                                        outputreverb[:, tt] += addedoutput

                                                if returnpulse == True:
                                                        distance = ((ii * r_length + r_min)**2 + (jj * z_length)**2)**0.5/3e8
                                                        dix = distance//(timescale)
                                                        pulsemap[ii, jj, tt] = illumination[0, int(tt-dix)] #* (ii+r_min/r_length) / ((ii+r_min/r_length)**2 + jj**2)
                                                        
        timedelay *= 1/(timescale)

        if returnvelocitygrid == True:                                                        
                if returnpulse == True:                        
                        if returntimedelay == True:
                                return(outputreverb, pulsemap, vgrid, timedelay)
                        else:
                                return(outputreverb, pulsemap, vgrid)                        
                elif returnpulse == False:
                        if returntimedelay == True:
                                return(outputreverb, vgrid, timedelay)
                        else:
                                return(outputreverb, vgrid)
        else:
                if returnpulse == True:
                        if returntimedelay == True:
                                return(outputreverb, pulsemap, timedelay)
                        else:
                                return(outputreverb, pulsemap)
                if returntimedelay == True:
                        return(outputreverb, timedelay)
                else:
                        return(outputreverb)

        

def CreateReverbDisk(disk, steps, incangle, zquasar = 1.0, mass = 10**8.0*2*10**30, fov = 0.12, diskres = 300, geounits = 4000, init_wavelength=464, compared_wavelength = 658,
                     scale_exponent = 4/3, dampingfactor = 0.1, flickermagnitude = 0.01, steptimescale = 60*60, inputsignal = False, illumination = [],
                     nGRs = 100, sim5 = False, returnscalechange=True):

        '''
        This function takes in a disk image and creates 2 reverberating disk models--one at wavelength init_wavelength, and
        one at wavelength compared_wavelength. The disk should be inserted as a [diskres, diskres] array. Mass is input in kg,
        geounits is a parameter of the Gyoto disk image creation. The scale_exponent is how the disk scales as a function of wavelength.
        steptimescale is how long each timestep is--default is 1 lighthour.
        '''
        import numpy as np
        import QuasarModelFunctions as QMF
        import matplotlib.pyplot as plt
        import numpy.random as random
        
        disk_size_ratio = (compared_wavelength/init_wavelength)**scale_exponent
        
        steptimescale *= 3e8  #Units light-something

        lumdist = (1 + float(zquasar))**2 * QMF.AngDiameterDistance(float(zquasar))
        
        
        if type(disk) == str:
                ii = 0
                dummyhdu = np.empty([int(diskres), int(diskres)])
                with open(disk, 'r') as f:
                    for line in f:
                        line = line.strip()
                        columns = line.split()
                        dummyhdu[:, ii] = np.asarray(columns, dtype=float)
                        ii += 1
                disk = dummyhdu
        else:
                disk = disk
                diskres = np.size(disk, 0)
        hdu = np.empty([int(diskres), int(diskres), steps])

                
        for jj in range(steps):
            hdu[:, :, jj] = (disk != 0)
            hdu[:, :, jj] *= disk
            
        if inputsignal == False:        
                flickering = QMF.DampedWalk(np.ones(steps), dampingfactor = dampingfactor, steppingfactor=flickermagnitude) #This is the input signal for lamppost model
        else:
                flickering = illumination
        hdu2 = hdu.copy()

        if sim5 == True:
                rstep1 = nGRs * QMF.GetGeometricUnit(mass) / diskres
        else:
                rstep1 = (fov) * geounits * QMF.GetGeometricUnit(mass) / diskres
        rstep2 = rstep1 * disk_size_ratio

        delaymap1 = np.empty([int(diskres), int(diskres)])
        delaymap2 = delaymap1.copy()
        dampingmap1 = delaymap1.copy()
        dampingmap2 = delaymap2.copy()
        incangle *= np.pi/180

        for xx in range(diskres):
            for yy in range(diskres):
                z1 = rstep1 * (diskres/2 - yy) * np.sin(incangle)
                z2 = rstep2 * (diskres/2 - yy) * np.sin(incangle)
                x1 = rstep1 * (diskres/2 - yy) * np.cos(incangle)
                x2 = rstep2 * (diskres/2 - yy) * np.cos(incangle)
                y1 = rstep1 * (diskres/2 - xx)
                y2 = rstep2 * (diskres/2 - xx)
                r1 = (x1**2 + y1**2)**0.5
                r2 = (x2**2 + y2**2)**0.5
                delaymap1[xx, yy] = abs(z1 - (z1**2 + r1**2)**0.5)//steptimescale
                delaymap2[xx, yy] = abs(z2 - (z2**2 + r2**2)**0.5)//steptimescale
                dampingmap1[xx, yy] = lumdist.value**2/(lumdist.value**2 + (delaymap1[xx, yy]*steptimescale)**2)
                dampingmap2[xx, yy] = lumdist.value**2/(lumdist.value**2 + (delaymap2[xx, yy]*steptimescale)**2)


        for jj in range(steps):
            for xx in range(diskres):
                for yy in range(diskres):
                    if hdu[xx, yy, jj] != 0:
                        hdu[xx, yy, jj] = flickering[(jj - int(delaymap1[xx, yy]))%steps]
                        hdu2[xx, yy, jj] = flickering[(jj - int(delaymap2[xx, yy]))%steps]
                    if xx == diskres//2 and yy == diskres//2:
                        hdu[xx, yy, jj] = flickering[jj]
                        hdu2[xx, yy, jj] = flickering[jj]
                
        if returnscalechange==True:
                return(hdu, hdu2, flickering, disk_size_ratio)
        else:
                return(hdu, hdu2, flickering)

def CreateTimeDelayMap(disk, incangle, massquasar = 10**8 * const.M_sun.to(u.kg), diskres = 300, fov = 0.12, geounits = 4000,
                       nGRs = 100, sim5 = False, unit='hours'):
        '''
        This aims to create a time delay mapping for the accretion disk for reverberation of the disk itself.
        The input disk should be some accretion disk map which is just used to determine where the accretion disk appears
        The output disk is a time delay mapping in units of lightdays. This can be scaled for larger accretion disks, or for different
        pixel sized maps. Due to linear nature of speed of light, only one needs to be used for any particular viewing angle*
                *if ISCO changes, this will produce more signal similar to input, and won't change overall reverb much!
                *only approximate around shadow of Black hole!
        '''
        import numpy as np
        import QuasarModelFunctions as QMF
        from astropy.io import fits

        incangle *= np.pi/180
        if type(disk) == str and sim5 == False:
                ii = 0
                dummyhdu = np.zeros([int(diskres), int(diskres)])
                with open(disk, 'r') as f:
                    for line in f:
                        line = line.strip()
                        columns = line.split()
                        dummyhdu[:, ii] = np.asarray(columns, dtype=float)
                        ii += 1
                disk = dummyhdu
        elif type(disk) == str and sim5 == True:
                with fits.open(disk) as f:
                        disk = f[0].data
                diskres = np.size(disk, 0)
        else:
                disk = disk
                diskres = np.size(disk, 0)
        if type(unit) == str:
                if unit == 'days' or unit == 'Days':
                        steptimescale = 3e8*60*60*24
                elif unit == 'hours' or unit == 'Hours':
                        steptimescale = 3e8*60*60 #units light hours / days
                elif unit == 'minutes' or unit == 'Minutes':
                        steptimescale = 3e8*60
                elif unit == 'seconds' or unit == 'Seconds':
                        steptimescale = 3e8
                else:
                        print('Invalid string deteted. Try "days", "hours", "minutes", "seconds" or an astropy.unit.\nReverting to hours.')
                        steptimescale = 3e8*60*60
        elif type(unit) == astropy.units.core.Unit or type(unit) == astropy.units.core.IrreducibleUnit:
                steptimescale = 3e8 * u.s.to(unit)
        else:
                print('Invalid unit deteted. Try "days", "hours", "minutes", "seconds" or an astropy.unit.\nReverting to hours.')
                steptimescale = 3e8*60*60
        if sim5 == True:
                rstep = nGRs * QMF.GetGeometricUnit(massquasar) / diskres
        else:
                rstep = fov * geounits * QMF.GetGeometricUnit(massquasar) / diskres
        indexes = disk >= 0.0001
        output = np.ndarray(np.shape(disk.copy()))
        for xx in range(diskres):
            for yy in range(diskres):
                if indexes[xx, yy] != 0:
                        if sim5 == False:
                                z1 = rstep * (diskres/2 - yy) * np.sin(incangle)
                                x1 = rstep * (diskres/2 - yy) * np.cos(incangle)
                                y1 = rstep * (diskres/2 - xx)
                                r1 = (x1**2 + y1**2)**0.5
                                output[xx, yy] = (abs(z1 - (z1**2 + r1**2)**0.5)/steptimescale + 0.5)
                        else:
                                z1 = rstep * (diskres/2 - xx) * np.sin(incangle)
                                x1 = rstep * (diskres/2 - xx) * np.cos(incangle)
                                y1 = rstep * (diskres/2 - yy)
                                r1 = (x1**2 + y1**2)**0.5
                                output[xx, yy] = (abs(z1 - (z1**2 + r1**2)**0.5)/steptimescale + 0.5)                                
        return output


def CreateReverbSnapshots(delaymap, time, illumination = False, massquasar = 10**8 * const.M_sun.to(u.kg), diskres = 300, fov = 0.12, geounits = 4000,
                         dampingfactor = 0.1, steppingfactor = 0.01, DRWseed=False):
        '''
        This uses a delay map created above and applies some random walk to be sampled with the delay mapping.
        It assumes time reversal of some random walk. For time series, please provide illumination so it's consistent!
        If illumination is provided, it will use that instead of the randomly generated DRW.
        
        '''
        import numpy as np
        import QuasarModelFunctions as QMF

        rstep = fov * geounits * QMF.GetGeometricUnit(massquasar) / diskres
        maxtime = int(np.max(delaymap) + 0.5)+time
        if type(illumination) == bool:
                illumination = QMF.DampedWalk(np.ones(maxtime), dampingfactor = dampingfactor, steppingfactor = steppingfactor, seed = DRWseed)
        else:
                assert(len(illumination) >= maxtime)
        output = np.zeros([np.size(delaymap, 0), np.size(delaymap, 1), (time)])
        mask = delaymap != 0
        for jj in range(time):
                delays = (delaymap+jj)
                output[:, :, jj] = illumination[(delays)]*mask
        
        return output
        

def DampedWalk(OrigLC, dampingfactor = 0.1 , steppingfactor = 0.01, seed=False):
        '''
        This function approximates the local variability in the quasar disk, in order to determine how it will echo throughout the
        BELR. Using many random flares originating from the corona / center of the quasar, the result is very similar to random damped
        walks. Additionally, producing a random damped walk is much less computationally expensive, so this will be the method primarily
        use to create these variability light curves.
        The light curve will essentially take a step randomly up to amplitude steppingfactor, then take a step back towards the mean
        with amplitude dampingfactor * distance_from_mean.
        Dampingfactor should be less than 1.
        steppingfactor is a decimal which will multiply the max of the given lightcurve, so the scale stays relatively proper.
        '''
        import numpy as np
        import numpy.random as random

        if type(seed) == int:
                random.seed(seed)

        steppingfactor *= np.average(OrigLC)

        outputLC = OrigLC.copy()
        randomoffset = random.randint(0, len(OrigLC))   #This will shuffle the start point of the damped walk
        for jj in range(len(OrigLC)):
                newval = outputLC[jj-1] + random.normal()*steppingfactor   # Add to the previous step, not original value! normal
                difference = newval - OrigLC[jj]                        # Find the difference with this new step and the original light curve
                outputLC[jj] = newval - dampingfactor * difference

        return(outputLC)
                

def ModelBELRRegion(sl1, sl2, viewingangle, linewavelength, obswavelengths, linewidth, linestrength, diskres = 300, efficiencyparam = 0.1, zshift = 2.0,
                    mass = 1e8 * 2 * 10**30, geounits = 4000, fov = 0.12, absorb=False):
        '''
        This function aims to view the action of the wind region without the underlaying accretion disk model. For absorption, I will
        assume some general continuum value. Diskres is simply here to give a relative feel for the size ratios between my previous
        simulated disks and this simulated wind region.
        WORK IN PROGRESS
        '''
        import numpy as np
        
        assert(viewingangle < 90)
        assert(np.shape(sl1) == np.shape(sl2))

        zstep = sl1[1, 1] - sl1[1, 0]
        dummydistance = GetGeometricUnit(mass)
        rstep = (fov) * geounits * dummydistance / diskres
        viewingangle *= np.pi / 180  #Convert to rads

        yoffset = zstep * np.tan(viewingangle)

        fakedisk = np.ones([diskres * 1, diskres * 1, len(obswavelengths)]) * 1
        screen = np.zeros([diskres * 1, len(sl1[1, :]), len(obswavelengths)])   # This catches wind information which appears above the disk's image


        for ii in range(np.size(fakedisk, 0)):
                x = (ii - diskres*10 // 2) * rstep
                for jj in range(np.size(fakedisk, 1)):
                        for kk in range(len(sl1[1, :])):
                                y = (jj - diskres*10 // 2) * rstep
                                radius, phi = ConvertToPolar(x, y - yoffset * kk)

                                if (radius <= sl1[0, kk] and radius >= sl2[0, kk]) or (radius >= sl1[0, kk] and radius <= sl2[0, kk]):
                                        leftside = abs(sl1[1, kk] - radius)
                                        rightside = abs(sl2[1, kk] - radius)
                                        totallength = abs(sl2[0, kk] - sl1[0, kk])
                                        fracleft = leftside/totallength
                                        fracright = rightside/totallength
                                        stream = fracleft * sl1[:, kk] + fracright * sl2[:, kk]
                                        betaaway = stream[5]/3e8
                                        abs_wavelength = linewavelength * ((1 + betaaway) / (1 - betaaway))**0.5

                                        b1 = stream[4] * np.cos(viewingangle) / 3e8
                                        b2 = stream[2] * np.sin(viewingangle) * (-1) * np.sin(phi) / 3e8
                                        b3 = stream[3] * np.sin(viewingangle) * np.cos(phi) / 3e8
                                        gammatoward = (1/(1-b1**2))**0.5 + (1/(1-b2**2))**0.5 + (1/(1-b3**2))**0.5
                                        betatoward = (1 - (1/gammatoward)**2)**0.5
                                        assert(betatoward < 1)
                                        emit_wavelength = linewavelength * ((1 - betatoward) / (1 + betatoward))**0.5

                                        gaussian_emit = (linestrength / (linewidth * (2 * np.pi)**0.5)) * np.exp(-(emit_wavelength - obswavelengths)**2 / (2 * linewidth**2))
                                        
            
                                        if absorb==True:
                                                gaussian_abs = (efficiencyparam / (linewidth * (2 * np.pi)**0.5)) * np.exp(-(abs_wavelength - obswavelengths)**2 / (2 * linewidth**2))
                                                fakedisk[ii, jj, :] *= (1 - gaussian_abs)
                                        fakedisk[ii, jj, :] += (gaussian_emit)

        return(fakedisk)


def Correlate(LightCurve, LightCurve2 = False):
        '''
        This takes in a curve and returns its autocorrelation along with the FWHM of the peak, assuming there is one peak.
        Including another light curve will then give cross-correlation, along with FWHM and the peak.
        '''
        import numpy as np
        from scipy import signal
        from scipy.interpolate import UnivariateSpline

        if type(LightCurve2) == bool:
                LightCurve2 = LightCurve

        lags = signal.correlation_lags(len(LightCurve), len(LightCurve2))
        correlation = signal.correlate(LightCurve - np.average(LightCurve), LightCurve2 - np.average(LightCurve2))
        spline = UnivariateSpline(lags, correlation - np.max(correlation)/2, s=0)
        r1, r2 = spline.roots()
        FWHM = r2-r1
        peak = np.argmax(correlation)
        
        return lags, correlation, FWHM, peak 


def InsertFlare(Disk, amplitude, theta_inc, decaytime, initialpoint=False, returnmovie=False, returnlightcurve=False, verbose=False):
        '''
        This aims to create a multiplicative field which decays in time in order to model a
        flaring effect as it propagates across the accretion disk. For some initial amplification,
        and at some initial location if given, it will spread out across the 'flat' disk radially
        with time. After being magnified, the multiplicative factor will decrease exponentially
        as exp(-timestep/decaytime). At each time step, the flare will move one pixel outwards.
        
        theta_inc should be inserted in degrees.
        initialpoint should be set as [x, y]

        Setting the returnmovie=True will return a 3-dim image of the disk as the flare propagates.

        Setting the returnlightcurve=True will return the sumtotal of the pixel values * flarefield as
        time progresses to create a simple 2-dim plot.
        '''
        import numpy as np
        assert(returnmovie==True or returnlightcurve==True)
        
        diskxsize = np.size(Disk, 0)
        diskysize = np.size(Disk, 1)
        flarefield = np.ones([diskxsize, diskysize, 2*diskxsize])       #Final dimension is unitless time, and should be long enough to show
        radiusfield = np.zeros([diskxsize, diskysize])                  #the effect even near the edges as time progresses.
        assert(theta_inc < 90)

        flarefieldydim = 1/np.cos(theta_inc*np.pi/180)                  #Distance scaling for the pixels which travel in the compressed direction

        if initialpoint == False:
                initialpoint = [diskxsize//2, diskysize//2]             #Int division so we can choose a pixel index

        for xx in range(diskxsize):
                for yy in range(diskysize):                             #Create a field of values to refer to for radii
                        radiusfield[xx, yy] = ((xx - initialpoint[0])**2 + ((yy - initialpoint[1])*flarefieldydim)**2)**0.5
                        
        decaymask = np.zeros([diskxsize, diskysize])                    # Define so the zero loop doesn't break
        for ii in range(np.size(flarefield, 2)):
                mask = (radiusfield <= ii)                              # Create a mask which shows the actively flaring parts
                if ii > 0:
                        decaymask = (radiusfield <= ii-1)               # This mask shows the decaying parts post-flare
                for xx in range(diskxsize):
                        for yy in range(diskysize):
                                if mask[xx, yy] == 1 and flarefield[xx, yy, ii] != 1:
                                        flarefield[xx, yy, ii] = 1 + amplitude/(1+radiusfield[xx, yy]**2)
                                if decaymask[xx, yy] == 1 and ii > radiusfield[xx, yy]:
                                        flarefield[xx, yy, ii] = 1 + (amplitude/(1+radiusfield[xx, yy]**2)) * np.exp(-(ii - radiusfield[xx, yy]) / decaytime)
 
                if verbose==True:
                        print("Completed", ii, "out of", 2*diskxsize, "steps. ("+str(100*ii/(2*diskxsize))+" %)")

        output = np.empty([diskxsize, diskysize, 2*diskxsize])
        outputmovie = np.empty([diskxsize, diskysize, 2*diskxsize])
        
        for ii in range(2*diskxsize):
                outputmovie[:, :, ii] = Disk * flarefield[:, :, ii]

        if returnmovie==True:
                return(outputmovie)

        if returnlightcurve==True:
                outputgraph = np.empty([2*diskxsize])
                for ii in range(2*diskxsize):
                        outputgraph[ii] = np.sum(outputmovie[:, :, ii])
                return(outputgraph)
        

def ConvertToPolar(x, y):
        '''
        This simply converts x, y coords into r, theta coords
        '''
        import numpy as np
        
        r = (x**2 + y**2)**0.5
        theta=np.arctan2(y, x)
        return(r, theta)

def ConvertToCart(r, theta):
        '''
        This function switches r, theta coords back to x, y coords
        '''
        import numpy as np
        
        x = r * np.sin(theta)
        y = r * np.cos(theta)
        return(x, y)

        
        
        






        
                
























                

        

