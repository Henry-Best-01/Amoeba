'''
This file holds various classes for Amoeba.
Also required is "QuasarModelFunctions.py" in the same directory.
There are 3 objects included:
    -FlatDisk object which represents an accretion disk. QMF has a function to help generate the required maps
       assuming Sim5 is installed, named "CreateMaps".
       FlatDisk.MakeSurfaceIntensityMap() creates an intensity map while adjusting the wavelengths at each pixel due to redshift+doppler
       FlatDisk.MakeDBDTMap() creates a map of dB(T; lambda)/dT for the reprocessing model
       FlatDisk.MakeTimeDelayMap() creates a map of time delays between a point source above the accretion disk and the accretion disk
       FlatDisk.MakeDTDLxMap() creates a map of the geometric weights associated with the reprocessing model
       FlatDisk.ConstructDiskTransferFunction() calculates the model transfer function of the accretion disk under the lamppost model geometry

    -MagnificationMap object is an object set up to hold a magnification map with functions relating to microlensing. It is constructed with
       either a binary .dat file or a fits file, and information regarding the redshifts, convergence, shear, mass of microlenses, and number
       of einstein radii must be included.
       MagnificationMap.Convolve() makes the convolution between this magnification map and an accretion disk image, defined with Disk and disk_intensity_map. An optional rotation is allowed.
       MagnificationMap.PullLightCurve() pulls a light curve off the convolution, assuming some relative transferse velocity vtrans (km/s) and a time period time (years).
       MagnificationMap.GenerateMicrolensedResponse() magnifies a FlatDisk response function.

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


class FlatDisk:

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
        self.g_map = g_map          #redshift map
        self.r_map = r_map
        self.omg0 = omg0
        self.omgl = omgl
        self.little_h = H0/100
        self.lum_dist = QMF.CalcLumDist(self.redshift, Om0=self.omg0, OmL=self.omgl, little_h=self.little_h)
        self.rg = QMF.CalcRg(self.mass)
        self.pxsize = self.rg * self.numGRs * 2 / np.size(self.temp_map, 0)
        self.c_height = coronaheight
        

    def MakeSurfaceIntensityMap(self, wavelength, approxshift=False, returnwavelengths=False):
        
        if type(wavelength) != u.Quantity:
            wavelength *= u.nm
        else:
            dummy=wavelength.to(u.nm)
            wavelength=dummy
        
        event_horizon = (1 + (1 - (self.spin)**2)**0.5)
        radius = self.r_map 

        redshiftfactor = 1/(1+self.redshift)
        totalshiftfactor = redshiftfactor * self.g_map 
        if approxshift == True:
            gravshiftfactor = (1 - event_horizon/radius)**0.5
            reldopplershiftfactor = ((1+self.vel_map)/(1-self.vel_map))**0.5
            totalshiftfactor = redshiftfactor * gravshiftfactor * reldopplershiftfactor
        emittedwavelength = totalshiftfactor * wavelength.value
        
        output = np.nan_to_num(QMF.PlanckLaw(self.temp_map, emittedwavelength) * pow(self.g_map, 4.))
        if returnwavelengths == True: return output, emittedwavelength
        return output

    def MakeDBDTMap(self, wavelength, approxshift=False):
        
        if type(wavelength) != u.Quantity:
            wavelength *= u.nm
        else:
            dummy=wavelength.to(u.nm)
            wavelength=dummy
            
        event_horizon = (1 + (1 - (self.spin)**2)**0.5)
        radius = self.r_map
            
        redshiftfactor = 1/(1+self.redshift)
        totalshiftfactor = redshiftfactor * self.g_map 
        if approxshift == True:
            gravshiftfactor = (1 - event_horizon/radius)**0.5
            reldopplershiftfactor = ((1+self.vel_map)/(1-self.vel_map))**0.5
            totalshiftfactor = redshiftfactor * gravshiftfactor * reldopplershiftfactor
        emittedwavelength = totalshiftfactor * wavelength.value
        
        output = QMF.PlanckDerivative(self.temp_map, emittedwavelength) * pow(self.g_map, 4.)
                
        return np.nan_to_num(output)
        

    def MakeTimeDelayMap(self, coronaheight=None, axisoffset=0, angleoffset=0, unit='hours', jitters=True):
        
        if coronaheight: override = coronaheight
        else: override = 0
        coronaheight = self.c_height
        if override > 0: coronaheight = override
        
        output = QMF.MakeTimeDelayMap(self.temp_map, self.inc_ang, massquasar=self.mass, redshift = self.redshift, numGRs=self.numGRs*2, coronaheight=coronaheight,
                                        axisoffset=axisoffset, angleoffset=angleoffset, unit=unit, jitters=jitters, radiimap=self.r_map)
        return output


    def MakeDTDLxMap(self, wavelength, coronaheight=None, axisoffset=0, angleoffset=0, approxshift=False):
        if coronaheight: override = coronaheight
        else: override = 0
        coronaheight = self.c_height
        if override > 0: coronaheight = override
        
        disk_derivative = self.MakeDBDTMap(wavelength, approxshift=approxshift)
        output = QMF.MakeDTDLx(disk_derivative, self.temp_map, self.inc_ang, self.mass, coronaheight, numGRs=self.numGRs*2, axisoffset=axisoffset, angleoffset=angleoffset, radiimap=self.r_map)
        return output


    def ConstructDiskTransferFunction(self, wavelength, coronaheight=None, axisoffset=0, angleoffset=0, maxlengthoverride=4800, units='hours', albedo=0,
                                      smooth=False, scaleratio=1, fixedwindowlength=None, approxshift=False):
        if coronaheight: override = coronaheight
        else: override = 0
        coronaheight = self.c_height
        if override > 0: coronaheight = override
        
        disk_derivative = self.MakeDBDTMap(wavelength, approxshift=approxshift)

        output = QMF.ConstructDiskTransferFunction(disk_derivative, self.temp_map, self.inc_ang, self.mass, self.redshift, coronaheight, maxlengthoverride=maxlengthoverride, units=units, scaleratio=scaleratio,
                                        axisoffset=axisoffset, angleoffset=angleoffset, albedo=albedo, numGRs=self.numGRs*2, smooth=smooth, fixedwindowlength=fixedwindowlength, radiimap=self.r_map)
        return output


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
        self.ein_radius = QMF.CalcRe(self.zl, self.zq, M_lens=self.m_lens, Om0=Om0, OmL=OmL, little_h=H0/100)*QMF.CalcAngDiamDist(self.zq, Om0=Om0, OmL=OmL, little_h=H0/100).to(u.m).value

        if type(file_name) == np.ndarray:
            if file_name.ndim == 1:
                self.ray_map = QMF.ConvertMagMap(file_name)
            elif file_name.ndim == 2:
                self.ray_map = file_name
        elif file_name[-4:] == 'fits':
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

        print(self.resolution)
        print(self.ray_to_mag_ratio)
        print(np.sum(self.ray_map)/ self.resolution**2)
        print(np.sum(self.ray_map))
        
        self.mag_map = self.ray_map
        if ismagmap == False:
            self.mag_map = self.ray_map * self.ray_to_mag_ratio
        self.px_size = self.ein_radius * self.n_einstein / self.resolution
        self.px_shift = 0
            
    def Convolve(self, Disk, obs_wavelength, rotation=False):

        output, px_size, px_shift = QMF.ConvolveMaps(self.mag_map, Disk.MakeSurfaceIntensityMap(obs_wavelength), redshift_lens=self.zl, redshift_source=self.zq, mass_exp=Disk.mass_exp,
                            mlens=self.m_lens, nmapERs=self.n_einstein, numGRs=Disk.numGRs, rotation=rotation)
        return output, px_size, px_shift

    def PullValue(self, x_val, y_val):

        return QMF.PullValue(self.mag_map, x_val+self.px_shift, y_val+self.px_shift)
    

    def PullLightCurve(self, vtrans, time, x_start=None, y_start=None, phi_angle=None, returntrack=False):
        
        return QMF.PullLC(self.mag_map, self.px_size, vtrans, time, px_shift = self.px_shift, x_start=x_start, y_start=y_start, phi_angle=phi_angle, returntrack=returntrack)

    
    def GenerateMicrolensedResponse(self, Disk, wavelength, coronaheight=None, rotation=0, x_position=None,
                            y_position=None, axisoffset=0, angleoffset=0, unit='hours', scaleratio=1, smooth=False, returnmaps=False):

        if coronaheight: override = coronaheight
        else: override = 0
        coronaheight = Disk.c_height
        if override > 0: coronaheight = override
        
        return QMF.MicrolensedResponse(self, Disk, wavelength, coronaheight, rotation=rotation, x_position=x_position,
                            y_position=y_position, axisoffset=axisoffset, angleoffset=angleoffset, unit=unit,
                            smooth=smooth, returnmaps=returnmaps, scaleratio=scaleratio)


class ConvolvedMap(MagnificationMap):

    def __init__(self, MagMap, Disk, obs_wavelength, rotation=False):

        self.px_size = MagMap.px_size
        self.n_einstein = MagMap.n_einstein
        self.m_lens = MagMap.m_lens
        self.resolution = MagMap.resolution
        self.mag_map, self.px_size, self.px_shift = MagMap.Convolve(Disk, obs_wavelength, rotation=rotation)
        self.disk_mass_exp = Disk.mass_exp
        self.disk_inc_angle = Disk.inc_ang
        self.disk_rg = Disk.rg
        self.disk_obs_wavelength = obs_wavelength
        self.rotation = rotation

    


        

















