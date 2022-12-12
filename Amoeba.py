'''
This file holds various classes for Amoeba.
Also required is "QuasarModelFunctions.py" in the same directory.
There are 3 objects included:
    -ThinDisk object which represents an accretion disk. QMF has a function to help generate the required maps
       assuming Sim5 is installed, named "CreateMapsForDiskClass".
       ThinDisk.CreateRestSurfaceIntensityMap() creates an intensity map without adjusting the wavelengths due to redshift and doppler shift
       ThinDisk.CreateObservedSurfaceIntensityMap() creates an intensity map while adjusting the wavelengths at each pixel due to redshift+doppler
       ThinDisk.CreateReprocessedEmissionMap() creates a map of dB(T; lambda)/dT for the reprocessing model
       ThinDisk.PlanckLaw() calculates the emission of Planck's Law for any given temp + wavelength. Not specific to this model.
       ThinDisk.PlanckTempDerivative() calculates the derivative of Planck's Law w.r.t. temp, for any given temp + wavelength
       ThinDisk.GetGeometricUnit() calculates one geometric unit, defined as G M c^(-2)
       ThinDisk.AngDiameterDistance() calculates the angular diameter distance for given flat Lambda-CDM cosmology.
       ThinDisk.CalculateLuminosityDistance() calculates the luminosity distance for given flat Lambda-CDM cosmology.
       ThinDisk.CreateTimeDelayMap() creates a map of time delays between a point source above the accretion disk and the accretion disk
       ThinDisk.CreateGeometricFactorMap() creates a map of the geometric weights associated with the reprocessing model
       ThinDisk.ConstructDiskTransferFunction() calculates the model transfer function of the accretion disk under the lamppost model geometry

    -MagnificationMap object is an object set up to hold a magnification map with functions relating to microlensing. It is constructed with
       either a binary .dat file or a fits file, and information regarding the redshifts, convergence, shear, mass of microlenses, and number
       of einstein radii must be included.
       MagnificationMap.Convolve() makes the convolution between this magnification map and an accretion disk image, defined with Disk and disk_intensity_map. An optional rotation is allowed.
       MagnificationMap.CalcMapPixSize() calculates the physical size of one pixel
       MagnificationMap.CalcEinsteinRadius() calculates the Einstein Radius of the microlensing system. All distances assume flat lambda-CDM model.
       MagnificationMap.AngDiameterDistanceDifference() calculates the angular diameter distance difference between the lens and source
       MagnificationMap.AngDiameterDistance() calculates the angular diameter distance
       MagnificationMap.PullRandomLightCurve() pulls a random light curve off the convolution, assuming some relative transferse velocity vtrans (km/s) and a time period time (years).

    -BroadLineRegion object is an object defined to create a simulated broad line region. It is initialized with a ThinDisk object, resolutions
       in the z, r, phi directions for calculations, a maximum height maxz, and a projection resolution resolution.
       BroadLineRegion.CreateWindLine() sets up a single wind line which will be used to bound a broad line region area
       BroadLineRegion.AddWindRegion() interpolates information between two bounding streamlines and adds them to the broad line region object
       BroadLineRegion.ProjectBLR() projects the current broad line region density onto the source plane
       BroadLineRegion.CreateBLRTransferFunction() creates a model transfer function which assumes reprocessed photons from the accretion disk are
        further scattered by the BLR density.

'''

from astropy import units as u
from astropy import constants as const
from scipy.integrate import quad
import numpy as np
import QuasarModelFunctions as QMF
from scipy.fft import fft2, ifft2
from scipy.ndimage import rotate
from skimage.transform import rescale 
from astropy.io import fits


c = const.c                                        		
G = const.G	                             
sigma = const.sigma_sb                  # Stefan_Boltzmann Const     
h = const.h                             #Planck Const
k = const.k_B                           #Boltzmann Const
M_Proton = const.m_p                    #Mass of Proton
Thompson_Cross_Section = const.sigma_T  


class ThinDisk:

    def __init__(self, mass_exponent, redshift, nGRs, inc_ang, temp_map, vel_map, g_map, name=''):

        self.name = name            # Label for particularly modelled systems
        self.mass_exponent = mass_exponent
        self.mass = 10**mass_exponent * const.M_sun.to(u.kg)
        self.nGRs = nGRs
        self.redshift = redshift
        self.inc_ang = inc_ang
        self.temp_map = temp_map
        self.vel_map = vel_map      # Line-of-sight velocity map
        self.g_map = g_map          # Relativistic boost map
        self.pxsize = self.GetGeometricUnit() * self.nGRs / np.size(self.temp_map, 0)

    def CreateRestSurfaceIntensityMap(self, wavelength):

        return QMF.PlanckLaw(self.temp_map, wavelength) * pow(self.g_map, 4.)

    def CreateObservedSurfaceIntensityMap(self, wavelength):
        
        if type(wavelength) != u.Quantity:
            wavelength *= u.nm
        else:
            dummy=wavelength.to(u.nm)
            wavelength=dummy
        redshiftedwavelength = wavelength / (1 + self.redshift)
        dopplershiftedwavelength = redshiftedwavelength * (1/(1-(self.vel_map)))
        output = self.PlanckLaw(self.temp_map, dopplershiftedwavelength) * pow(self.g_map, 4.)

        return output, dopplershiftedwavelength

    def CreateReprocessedEmissionMap(self, wavelength):
        
        if type(wavelength) != u.Quantity:
            wavelength *= u.nm
        redshiftedwavelength = wavelength / (1 + self.redshift)
        dopplershiftedwavelength = redshiftedwavelength * (1/(1-(self.vel_map)))
        output = self.PlanckTempDerivative(self.temp_map, wavelength) * pow(self.g_map, 4.)
                
        return np.nan_to_num(output)
                            

    def PlanckLaw(self, T, lam):
        '''
        I plan to pass in lam in units of [nm]. Otherwise, attach the units and it will convert.
        '''
        if type(lam) != u.Quantity:
                lam *= u.nm
        
        return ((2.0 * h * c**(2.0) * (lam.to(u.m))**(-5.0) * ((np.e**(h * c / (lam.to(u.m) * k * T)).decompose().value - 1.0)**(-1.0))).to(u.W/(u.m**3)))  # This will return the Planck Law wavelength function at the temperature input

    def PlanckTempDerivative(self, T, lam): 
        '''
        This is the derivative of Planck's Law, with respect to temperature
        '''
        if type(T) != u.Quantity:
                T *= u.K
        if type(lam) != u.Quantity:
                lam *= u.nm

        a = 2 * h**2 * c**4 / ((lam.to(u.m))**6.0 * k * T**2)
        b = np.e**(h * c / (lam.to(u.m) * k * T)).decompose()

        return a.value * b.value / (b.value - 1)**2
    

    def GetGeometricUnit(self):
        '''
        This function simply returns what the length (in meters) of a geometric unit is for a given mass (in kg)
        '''
        if type(self.mass) != u.Quantity:
                self.mass *= u.kg
        return (G * self.mass / c**2).decompose()

    def AngDiameterDistance(self, Om0=0.3, OmL=0.7):
        '''
        This funciton takes in a redshift value of z, and calculates the angular diameter distance. This is given as the
        output. This assumes LCDM model.
        '''
        z = self.redshift
        multiplier = (9.26* 10 **25) * (10/7) * (1 / (1 + z))               # This need not be integrated over
        integrand = lambda z_p: ( Om0 * (1 + z_p)**(3.0) + OmL )**(-0.5)    # This must be integrated over
        integral, err = quad(integrand, 0, z)
        output = multiplier * integral * u.m
        return output
        
    def CalculateLuminosityDistance(self, Om0=0.3, OmL=0.7):
        '''
        This calculates the luminosity distance using the AngdiameterDistance formula above for flat lam-CDM model
        '''
        z = self.redshift
        
        return ((1 + z)**2 * self.AngDiameterDistance(Om0, OmL))

    def CreateTimeDelayMap(self, coronaheight, axisoffset=0, angleoffset=0, unit='hours'):
        output = QMF.CreateTimeDelayMap(self.temp_map, self.inc_ang, massquasar=self.mass, nGRs=self.nGRs, coronaheight=coronaheight,
                                        axisoffset=axisoffset, angleoffset=angleoffset, sim5=True, unit='hours')
        return output

    def CreateGeometricFactorMap(self, coronaheight, axisoffset=0, angleoffset=0, albedo=0):
        output = QMF.ConstructGeometricDiskFactor(self.temp_map, self.inc_ang, self.mass, coronaheight, r=axisoffset,
                                        phi=angleoffset, albedo=albedo, sim5=True)
        return output

    def ConstructDiskTransferFunction(self, disk_derivative, coronaheight, axisoffset=0, angleoffset=0, units=u.h, albedo=0, weight=False):
        output = QMF.ConstructDiskTransferFunction(disk_derivative, self.inc_ang, self.mass, coronaheight, units=units,
                                        r=axisoffset, phi=angleoffset, albedo=albedo, weight=weight, nGRs=self.nGRs)
        return output

class MagnificationMap:

    def __init__(self, redshift_quasar, redshift_lens, file_name, convergence, shear,
                 m_lens = 1 * const.M_sun.to(u.kg), n_einstein = 25, name = ''):

        self.name = name
        self.zq = redshift_quasar
        self.zl = redshift_lens
        self.file_name = file_name
        self.convergence = convergence
        self.shear = shear
        self.n_einstein = n_einstein
        self.m_lens = m_lens

        if file_name[-4:] == 'fits':
            with fits.open(file_name) as f:
                self.ray_map = f[0].data
        elif file_name[-4:] == '.dat':
            with open(file_name, 'rb') as f:
                MagMap = np.fromfile(f, 'i', count=-1, sep='')
                self.ray_map = QMF.ConvertMagMap(MagMap)
        else:
            print("Invalid file name. Please pass in a .fits or .dat file")
            
        self.resolution = np.size(self.ray_map, 0)
        self.ray_to_mag_ratio = (1 / ((1 - self.convergence)**2.0 - self.shear**2.0)) / (np.sum(self.ray_map) / self.resolution**2.0)
        self.mag_map = self.ray_map * self.ray_to_mag_ratio
                
    
            
    def Convolve(self, Disk, disk_intensity_map, rotation=False):

        output, _ = QMF.ConvolveSim5Map(self.mag_map, disk_intensity_map, zlens=self.zl, zquasar=self.zq, mquasarexponent=Disk.mass_exponent,
                            mlens=self.m_lens, nmapERs=self.n_einstein, numGRs=Disk.nGRs, rotation=rotation)
        return output
        



    def CalcMapPixSize(self):
        ERsize = self.CalcEinsteinRadius() * self.AngDiameterDistance()
        pxsize = ERsize / self.resolution
        return pxsize


    def CalcEinsteinRadius(self, Om0=0.3, OmL=0.7):
        '''
        This function takes in values of z_lens and z_source (not simply by finding 
        the difference of the two! See AngDiameterDistanceDifference function above!). The output is the
        Einstein radius of the lens, in radians. This assumes LCDM model.
        '''
        D_lens = self.AngDiameterDistance(Om0, OmL)
        D_source = self.AngDiameterDistance(Om0, OmL)
        D_LS = self.AngDiameterDistanceDifference(Om0, OmL)
        output = ((( 4 * G * self.m_lens / c**2) * D_LS / (D_lens * D_source))**(0.5)).value
        return output
            

    def AngDiameterDistanceDifference(self, Om0=0.3, OmL=0.7):
        '''
        This function takes in 2 redshifts, designed to be z1 = redshift (lens) and z2 = redshift (source). It then
        integrates the ang. diameter distance between the two. This assumes LCDM model.
        '''
        multiplier = (9.26* 10 **25) * (10/7) * (1 / (1 + self.zq))
        integrand = lambda z_p: ( Om0 * (1 + self.zl)**(3.0) + OmL )**(-0.5)               # This must be integrated over
        integral1, err1 = quad(integrand, 0, self.zl)
        integral2, err2 = quad(integrand, 0, self.zq)
        output = multiplier * (integral2 - integral1) * u.m
        return output

    def AngDiameterDistance(self, Om0=0.3, OmL=0.7):
        '''
        This funciton takes in a redshift value of z, and calculates the angular diameter distance. This is given as the
        output. This assumes LCDM model.
        '''
        z = self.zq
        multiplier = (9.26 * 10**25) * (10/7) * (1 / (1 + z))               # This need not be integrated over
        integrand = lambda z_p: ( Om0 * (1 + z_p)**(3.0) + OmL )**(-0.5)    # This must be integrated over
        integral, err = quad(integrand, 0, z)
        output = multiplier * integral * u.m
        return output

    def PullRandomLightCurve(self, convolution, vtrans, time):
        output = QMF.PullRandLC(convolution, self.CalcMapPixSize(), vtrans, time)
        return output
        

class BroadLineRegion:

    def __init__(self, Disk, zres, rres, phires, maxz, resolution=100):
        self.massexp = Disk.mass_exponent
        self.mass = 10**self.massexp * const.M_sun.to(u.kg)
        self.inc_ang = Disk.inc_ang
        self.zres = zres
        self.rres = rres
        self.phires = phires
        self.windgrid = np.zeros((rres, zres, phires, 4))
        self.maxz = maxz
        self.windparams = []
        self.projection_resolution = resolution

    def CreateWindLine(self, launch_radius, launch_angle, r0, launch_height=0, maxvel=10**6, launchspeed=0, alpha=1):
        output = QMF.CreateWindLine(launch_radius, launch_angle, self.maxz, self.zres, r0, centralBHmassexp=self.massexp,
                                    launchheight=launch_height, maxvel=maxvel, launchspeed=launchspeed, alpha=alpha)
        return output

    def AddWindRegion(self, sl1, sl2, r0=False, sigma=False, function=1, power=1, overwrite=False, weight=1):
        if r0 == False:
            r0 = self.maxz/2
        if sigma == False:
            sigma = self.maxz/4
        output, rlen, zlen, philen, rmin = QMF.CreateWindRegion(sl1, sl2, r_res=self.rres, z_res=self.zres, phi_res=self.phires, centralBHmassexp=self.massexp,
                                      r0=r0, sigma=sigma, function=function, power=power)
        mask = self.windgrid == 0
        if overwrite == False:
            self.windgrid += mask * output * weight
        elif overwrite == True:
            mask = output == 0
            output += mask * self.windgrid * weight
            self.windgrid = output
        self.windparams.append(rlen)
        self.windparams.append(zlen)
        self.windparams.append(philen)
        self.windparams.append(rmin)

    def ProjectBLR(self, sl1, sl2):
        output = QMF.ProjectWind(self.windgrid, self.windparams[-4], self.windparams[-3], self.windparams[-2],
                                 self.windparams[-1], self.inc_ang, self.projection_resolution, sl1, sl2, [], mass = self.mass)
        return output

    def CreateBLRTransferFunction(self, sl1, sl2, units='days', grid=False):
        output = QMF.CreateBLRTransferFunction(self.windgrid, self.windparams[-4], self.windparams[-3], self.windparams[-2],
                                               self.inc_ang, self.projection_resolution, sl1, sl2, mass=self.mass, units=units,
                                               returngrid=grid)
        return output
                                               
                      



















