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
sigma = const.sigma_sb                  #Stefan_Boltzmann Const     
h = const.h                             #Planck Const
k = const.k_B                           #Boltzmann Const
M_Proton = const.m_p                    #Mass of Proton
Thompson_Cross_Section = const.sigma_T  


class ThinDisk:

    def __init__(self, mass_exp, redshift, numGRs, inc_ang, coronaheight, temp_map, vel_map, g_map, r_map, spin=0, omg0=0.3, omgl=0.7, H0=70, name=''):

        self.name = name            # Label space for particularly modelled systems
        self.mass_exp = mass_exp
        self.mass = 10**mass_exp * const.M_sun.to(u.kg)
        self.numGRs = numGRs
        self.redshift = redshift
        self.inc_ang = inc_ang
        self.spin = spin
        self.temp_map = temp_map
        self.vel_map = vel_map      
        self.g_map = g_map          
        self.r_map = r_map
        self.omg0 = omg0
        self.omgl = omgl
        self.little_h = H0/100
        self.lum_dist = QMF.CalculateLuminosityDistance(self.redshift, Om0=self.omg0, OmL=self.omgl, little_h=self.little_h)
        self.rg = QMF.GetGeometricUnit(self.mass)
        self.pxsize = self.rg * self.numGRs * 2 / np.size(self.temp_map, 0)
        self.c_height = coronaheight
        


    def CreateObservedSurfaceIntensityMap(self, wavelength, returnwavelengths=False):
        
        if type(wavelength) != u.Quantity:
            wavelength *= u.nm
        else:
            dummy=wavelength.to(u.nm)
            wavelength=dummy
        
        event_horizon = (1 + (1 - abs(self.spin))**0.5)
        radius = self.r_map 

        redshiftfactor = 1/(1+self.redshift)
        gravshiftfactor = (1 - event_horizon/radius)**0.5
        reldopplershiftfactor = ((1+self.vel_map)/(1-self.vel_map))**0.5
        totalshiftfactor = redshiftfactor * reldopplershiftfactor * gravshiftfactor

        emittedwavelength = totalshiftfactor * wavelength.value
        
        output = np.nan_to_num(QMF.PlanckLaw(self.temp_map, emittedwavelength) * pow(self.g_map, 4.))
        if returnwavelengths == True: return output, emittedwavelength
        return output

    def CreateReprocessedEmissionMap(self, wavelength):
        
        event_horizon = (1 + (1 - abs(self.spin))**0.5)
        radius = self.r_map
            
        redshiftfactor = 1/(1+self.redshift)
        gravshiftfactor = (1 - event_horizon/radius)**0.5
        reldopplershiftfactor = ((1+self.vel_map)/(1-self.vel_map))**0.5
        totalshiftfactor = redshiftfactor * reldopplershiftfactor * gravshiftfactor

        emittedwavelength = totalshiftfactor * wavelength
        output = QMF.PlanckTempDerivativeNumeric(self.temp_map, emittedwavelength) * pow(self.g_map, 4.)
                
        return np.nan_to_num(output)
        

    def CreateTimeDelayMap(self, coronaheight=None, axisoffset=0, angleoffset=0, unit='hours'):
        
        if coronaheight: override = coronaheight
        else: override = 0
        coronaheight = self.c_height
        if override > 0: coronaheight = override
        
        output = QMF.CreateTimeDelayMap(self.temp_map, self.inc_ang, massquasar=self.mass, redshift = self.redshift, numGRs=self.numGRs*2, coronaheight=coronaheight,
                                        axisoffset=axisoffset, angleoffset=angleoffset, sim5=True, unit=unit)
        return output


    def MakeDTDLxMap(self, wavelength, coronaheight=None, axisoffset=0, angleoffset=0):
        if coronaheight: override = coronaheight
        else: override = 0
        coronaheight = self.c_height
        if override > 0: coronaheight = override
        
        disk_derivative = self.CreateReprocessedEmissionMap(wavelength)
        output = QMF.MakeDTDLx(disk_derivative, self.temp_map, self.inc_ang, self.mass, coronaheight, axisoffset=axisoffset, angleoffset=angleoffset)
        return output


    def ConstructDiskTransferFunction(self, wavelength, coronaheight=None, axisoffset=0, angleoffset=0, maxlengthoverride=4800, units='hours', albedo=0, weight=True,
                                      smooth=True, sim5=True, fixedwindowlength=None, spacing='linear'):
        if coronaheight: override = coronaheight
        else: override = 0
        coronaheight = self.c_height
        if override > 0: coronaheight = override
        
        disk_derivative = self.CreateReprocessedEmissionMap(wavelength)
        if spacing=='linear':
            output = QMF.ConstructDiskTransferFunction(disk_derivative, self.temp_map, self.inc_ang, self.mass, self.redshift, coronaheight, maxlengthoverride=maxlengthoverride, units=units,
                                        axisoffset=axisoffset, angleoffset=angleoffset, albedo=albedo, weight=weight, numGRs=self.numGRs*2, smooth=smooth, sim5=sim5, fixedwindowlength=fixedwindowlength, spacing=spacing)
            return output
        elif spacing=='log':
            times, output = QMF.ConstructDiskTransferFunction(disk_derivative, self.temp_map, self.inc_ang, self.mass, self.redshift, coronaheight, maxlengthoverride=maxlengthoverride, units=units,
                                        axisoffset=axisoffset, angleoffset=angleoffset, albedo=albedo, weight=weight, numGRs=self.numGRs*2, smooth=smooth, sim5=sim5, fixedwindowlength=fixedwindowlength, spacing=spacing)
            return times, output

class MagnificationMap:

    def __init__(self, redshift_quasar, redshift_lens, file_name, convergence, shear,
                 m_lens = 1 * const.M_sun.to(u.kg), n_einstein = 25, Om0=0.3, OmL=0.7, H0 = 70, ismagmap=False, name = ''):

        self.name = name
        self.zq = redshift_quasar
        self.zl = redshift_lens
        self.file_name = file_name
        self.convergence = convergence
        self.shear = shear
        self.n_einstein = n_einstein
        self.m_lens = m_lens
        self.ein_radius = QMF.CalcEinsteinRadius(self.zl, self.zq, M_lens=self.m_lens, Om0=Om0, OmL=OmL, little_h=H0/100)*QMF.AngDiameterDistance(self.zq, Om0=Om0, OmL=OmL, little_h=H0/100).to(u.m).value

        if file_name[-4:] == 'fits':
            with fits.open(file_name) as f:
                self.ray_map = f[0].data
        elif file_name[-4:] == '.dat':
            with open(file_name, 'rb') as f:
                MagMap = np.fromfile(f, 'i', count=-1, sep='')
                self.ray_map = QMF.ConvertMagMap(MagMap)
        elif type(file_name) == np.ndarray:
            if file_name.ndim == 1:
                self.ray_map = QMF.ConvertMagMap(file_name)
            elif file_name.ndim == 2:
                self.ray_map = file_name
        else:
            print("Invalid file name. Please pass in a .fits or .dat file")
            
        self.resolution = np.size(self.ray_map, 0)
        self.ray_to_mag_ratio = (1 / ((1 - self.convergence)**2.0 - self.shear**2.0)) / (np.sum(self.ray_map) / self.resolution**2.0)
        self.mag_map = self.ray_map
        if ismagmap == False:
            self.mag_map = self.ray_map * self.ray_to_mag_ratio
        self.px_size = self.ein_radius * self.n_einstein / self.resolution
        self.px_shift = 0
            
    def Convolve(self, Disk, obs_wavelength, rotation=False):

        output, px_size, px_shift = QMF.ConvolveSim5Map(self.mag_map, Disk.CreateObservedSurfaceIntensityMap(obs_wavelength), redshift_lens=self.zl, redshift_source=self.zq, mass_exp=Disk.mass_exp,
                            mlens=self.m_lens, nmapERs=self.n_einstein, numGRs=Disk.numGRs, rotation=rotation)
        return output, px_size, px_shift

    def PullValue(self, x_val, y_val):

        return QMF.MeasureMLAmp(self.mag_map, x_val+self.px_shift, y_val+self.px_shift)
    

    def PullRandomLightCurve(self, vtrans, time):
        
        return QMF.PullRandLC(self.mag_map, self.px_size, vtrans, time, px_shift = self.px_shift)

    
    def GenerateMicrolensedResponse(self, Disk, wavelength, coronaheight=None, rotation=False, x_position=None,
                            y_position=None, axisoffset=0, angleoffset=0, unit='hours', smooth=False, returnmaps=False):

        if coronaheight: override = coronaheight
        else: override = 0
        coronaheight = Disk.c_height
        if override > 0: coronaheight = override
        
        return QMF.MicrolensedResponse(self, Disk, wavelength, coronaheight, rotation=rotation, x_position=x_position,
                            y_position=y_position, axisoffset=axisoffset, angleoffset=angleoffset, unit=unit,
                            smooth=smooth, returnmaps=returnmaps)


class MagnifiedConvolution(MagnificationMap):

    def __init__(self, MagMap, Disk, obs_wavelength, rotation=False):

        self.px_size = MagMap.px_size
        self.n_einstein = MagMap.n_einstein
        self.m_lens = MagMap.m_lens
        self.resolution = MagMap.resolution
        self.mag_map, self.px_shift = MagMap.Convolve(Disk, obs_wavelength, rotation=rotation)
        self.disk_mass_exp = Disk.mass_exp
        self.disk_inc_angle = Disk.inc_ang
        self.disk_rg = Disk.rg
        self.disk_obs_wavelength = obs_wavelength
        self.rotation = rotation

    


        
















