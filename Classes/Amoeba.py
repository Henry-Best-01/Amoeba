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

    -ConvolvedMap object is a child of Magnification map allowing the user to store the convolution between a Disk and Map object (thus locking in
       both inclination and orientation angles). Relative pixel shift is stored and applied to every light curve returned by PullLightCurve()

    -BroadLineRegion object is an object designed to store a BLR. The max height and resolutions must be defined at initialization to avoid issues.
       Add_SL_Bounded_Region() takes in two streamline objects to use as boundaries to populate the BLR. Conservation of matter is considered. V_r
       and V_z are interpolated between streamlines, while V_phi is calculated assuming Keplerian velocity.
       Project_BLR_Density() produces a column density along line of sight through the BLR down to the plane of the accretion disk. 
       Project_BLR_Velocity_Slice() produces a similar projection while making out regions not falling within a predefined velocity slice (velocities
       normalized by speed of light, positive velocities are towards the observer).
       Scattering_BLR_TF() calculates a "transfer function" by assuming photons from the accretion disk may be scattered from the BLR particles. This
       scattering is proportional to the density of the BLR and is calculated from the position of the black hole (e.g. the "average" emission from the disk).
       We note this is a similar approximation to the lamppost approximation although the accretion disk is significantly larger than the size of the corona.
       The BLR is assumed to be optically thin except for the first scattering event. 
       Scattering_Vel_Line_BLR_TF() is similar to Scattering_BLR_TF(), but now masks out velocities not included by the predefined velocity slice. 
       Check_Line_Contamination() takes in some inclination angle, emission line, and bounds of a passband in order to let the user know if the
       emission line would contribute to the simulated observation. If it does, this method returns the v_0 and delta_v required to project the
       column density (for BALs) or calculate the reverberated emission line (for BELs).
       

'''

from astropy import units as u
from astropy import constants as const
from scipy.integrate import quad
import numpy as np
import sys
sys.path.append("../Functions/")
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

    def __init__(self, mass_exp, redshift, numGRs, inc_ang, coronaheight, temp_map, vel_map, g_map, r_map, spin=0, omg0=0.3, omgl=0.7, H0=70, R_out=None, name=''):

        self.name = name            # Label space for particularly modelled systems
        self.mass_exp = mass_exp
        self.mass = 10**mass_exp * const.M_sun.to(u.kg)
        self.numGRs = numGRs
        self.redshift = redshift
        self.inc_ang = inc_ang
        self.spin = spin
        self.r_map = r_map
        if R_out is None:
            R_out = r_map[np.size(r_map, 0)//2, -1]
        rmask = r_map <= R_out
        self.temp_map = temp_map * rmask
        self.vel_map = vel_map * rmask      
        self.g_map = g_map * rmask          
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
        

    def MakeTimeDelayMap(self, coronaheight=None, axisoffset=0, angleoffset=0, unit='hours', jitters=True, source_plane=True):
        
        if coronaheight: override = coronaheight
        else: override = 0
        coronaheight = self.c_height
        if override > 0: coronaheight = override
        
        output = QMF.MakeTimeDelayMap(self.temp_map, self.inc_ang, massquasar=self.mass, redshift = self.redshift, numGRs=self.numGRs*2, coronaheight=coronaheight,
                                        axisoffset=axisoffset, angleoffset=angleoffset, unit=unit, jitters=jitters, radiimap=self.r_map, source_plane=source_plane)
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
                                      smooth=False, scaleratio=1, fixedwindowlength=None, approxshift=False, jitters=False, source_plane=True):
        if coronaheight: override = coronaheight
        else: override = 0
        coronaheight = self.c_height
        if override > 0: coronaheight = override
        
        disk_derivative = self.MakeDBDTMap(wavelength, approxshift=approxshift)

        output = QMF.ConstructDiskTransferFunction(disk_derivative, self.temp_map, self.inc_ang, self.mass, self.redshift, coronaheight, maxlengthoverride=maxlengthoverride, units=units,
                                                   scaleratio=scaleratio, axisoffset=axisoffset, angleoffset=angleoffset, albedo=albedo, numGRs=self.numGRs*2, smooth=smooth,
                                                   fixedwindowlength=fixedwindowlength, radiimap=self.r_map, jitters=jitters, source_plane=source_plane)
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
                            y_position=None, axisoffset=0, angleoffset=0, unit='hours', scaleratio=1, smooth=False,
                                    returnmaps=False, jitters=False, source_plane=True):

        if coronaheight: override = coronaheight
        else: override = 0
        coronaheight = Disk.c_height
        if override > 0: coronaheight = override
        
        return QMF.MicrolensedResponse(self, Disk, wavelength, coronaheight, rotation=rotation, x_position=x_position,
                            y_position=y_position, axisoffset=axisoffset, angleoffset=angleoffset, unit=unit,
                            smooth=smooth, returnmaps=returnmaps, scaleratio=scaleratio, jitters=jitters, source_plane=source_plane)


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



class BroadLineRegion():

    def __init__(self, BHmassexp, max_z, r_res=10, z_res=10, max_r=0):
        # res holds the number of R_g each pixel is

        self.mass_exp = BHmassexp
        self.max_z = max_z
        self.r_res = r_res
        self.z_res = z_res
        self.max_r = max_r
        self.density_grid = np.zeros((max_r//r_res + 1, max_z//z_res + 1))                                  # Holds densities
        self.z_velocity_grid = np.zeros((max_r//r_res + 1, max_z//z_res + 1))                                 # Holds z velocities
        self.r_velocity_grid = np.zeros((max_r//r_res + 1, max_z//z_res + 1))                                 # Holds r velocities
        self.r_vals = np.linspace(0, max_r, max_r//r_res + 1)
        self.z_vals = np.linspace(0, max_z, max_z//z_res + 1)                                               

        self.mass = 10**(BHmassexp) * const.M_sun.to(u.kg)

    def Add_SL_bounded_region(self, Streamline_inner, Streamline_outer, density_init_weight=1):

        assert Streamline_inner.z_res == Streamline_outer.z_res
        assert Streamline_inner.max_z == Streamline_outer.max_z
        assert Streamline_inner.z_res == self.z_res
        assert Streamline_inner.max_z == self.max_z
        assert density_init_weight > 0
        
        if np.max([np.max(Streamline_inner.radii), np.max(Streamline_outer.radii)]) > self.max_r:
            prevmaxlen = self.max_r//self.r_res + 1
            self.max_r = int(np.max([np.max(Streamline_inner.radii), np.max(Streamline_outer.radii)]))
            dummygrid = np.zeros((self.max_r//self.r_res + 1, self.max_z//self.z_res + 1))
            dummygrid[:prevmaxlen, :] = self.density_grid
            self.density_grid = dummygrid
            dummygrid = np.zeros((self.max_r//self.r_res + 1, self.max_z//self.z_res + 1))
            dummygrid[:prevmaxlen, :] = self.z_velocity_grid
            self.z_velocity_grid = dummygrid
            dummygrid = np.zeros((self.max_r//self.r_res + 1, self.max_z//self.z_res + 1))
            dummygrid[:prevmaxlen, :] = self.r_velocity_grid
            self.r_velocity_grid = dummygrid
            
            self.r_vals = np.linspace(0, self.max_r, self.max_r//self.r_res + 1)

        for hh in range(np.size(self.z_vals)):
            low_mask = self.r_vals >= min(Streamline_inner.radii[hh], Streamline_outer.radii[hh])
            high_mask = self.r_vals <= max(Streamline_inner.radii[hh], Streamline_outer.radii[hh])
            mask = np.logical_and(low_mask, high_mask) + 0
            if hh == 0: norm = sum(mask)
            self.density_grid[np.argmax(mask):np.argmax(mask)+sum(mask), hh] *= 0  # overwrites these cells only
            self.r_velocity_grid[np.argmax(mask):np.argmax(mask)+sum(mask), hh] *= 0
            self.z_velocity_grid[np.argmax(mask):np.argmax(mask)+sum(mask), hh] *= 0
            kkspace = np.linspace(0, sum(mask), sum(mask))
            self.z_velocity_grid[np.argmax(mask):np.argmax(mask)+sum(mask), hh] = Streamline_inner.poloidal_vel[hh] * np.cos(Streamline_inner.launch_theta) + (kkspace / sum(mask)) * (Streamline_outer.poloidal_vel[hh] * np.cos(Streamline_outer.launch_theta) - Streamline_inner.poloidal_vel[hh] * np.cos(Streamline_inner.launch_theta))
            self.r_velocity_grid[np.argmax(mask):np.argmax(mask)+sum(mask), hh] = Streamline_inner.poloidal_vel[hh] * np.sin(Streamline_inner.launch_theta) + (kkspace / sum(mask)) * (Streamline_outer.poloidal_vel[hh] * np.sin(Streamline_outer.launch_theta) - Streamline_inner.poloidal_vel[hh] * np.sin(Streamline_inner.launch_theta))
            
            del_pol_vels_on_vels = Streamline_inner.dpol_vel_dz_on_vel[hh] + (kkspace / sum(mask)) * (Streamline_outer.dpol_vel_dz_on_vel[hh] - Streamline_inner.dpol_vel_dz_on_vel[hh])
            self.density_grid[np.argmax(mask):np.argmax(mask)+sum(mask), hh] += density_init_weight * del_pol_vels_on_vels * self.r_vals[np.argmax(mask):np.argmax(mask)+sum(mask)]**(-1)

    def Project_BLR_density(self, inc_ang, grid_size=100, R_out=None):
        # grid_size is the TOTAL number of grid points along an axis.
        # R_out is an override to truncate the radial boundary of the BLR, and by default is set to self.max_r

        return QMF.Project_BLR_density(self, inc_ang, grid_size=grid_size, R_out=R_out)


    def Project_BLR_velocity_slice(self, inc_ang, v_0, delta_v, grid_size=100, R_out=None, density_weighting=True):
        # Similar to above's density calculation, but this time only includes cells with line of sight veloicty
        # within v_0 +/- delta_v (dimensionless)

        return QMF.Project_BLR_velocity_slice(self, inc_ang, v_0, delta_v, grid_size=grid_size, R_out=R_out,
                                density_weighting=density_weighting)


    def Scattering_BLR_TF(self, inc_ang, grid_size=100, redshift=0, unit='hours', jitters=True, axisoffset = 0,
                                angleoffset = 0, scaleratio=10, source_plane=True):

        return QMF.Scattering_BLR_TF(self, inc_ang, grid_size=grid_size, redshift=redshift, unit=unit,
                                jitters=jitters, axisoffset = axisoffset, angleoffset = angleoffset,
                                scaleratio=scaleratio, source_plane=source_plane)


    def Scattering_Vel_Line_BLR_TF(self, inc_ang, v_0, delta_v, grid_size=100, redshift=0, unit='hours', jitters=True,
                                axisoffset = 0, angleoffset = 0, scaleratio=10, source_plane=True):

        return QMF.Line_BLR_TF(self, inc_ang, v_0, delta_v, grid_size=grid_size, redshift=redshift, unit=unit,
                                jitters=jitters, axisoffset = axisoffset, angleoffset = angleoffset,
                                scaleratio=scaleratio, source_plane=source_plane)


    def Check_Line_Contamination(self, inc_ang, emit_wavelength, passband_min, passband_max, redshift=0):

        return QMF.Check_EL_Contamination(self, inc_ang, emit_wavelength, passband_min, passband_max, redshift=redshift)


class Streamline():

    def __init__(self, launch_r, launch_theta, max_z, char_dist, BHmassexp, asympt_vel, z_res=10,
                 launch_z=1, launch_vel=1e-3, alpha=1, v_vec=None, r_vec=None):

        self.launch_radius = launch_r
        self.launch_theta = launch_theta * np.pi / 180
        self.launch_z = launch_z
        self.launch_vel = launch_vel
        self.z_res = z_res
        self.pol_init_vel = launch_vel
        self.radial_init_vel = launch_vel * np.sin(self.launch_theta)
        self.asympt_vel = asympt_vel
        self.max_z = max_z

        assert launch_vel >= 0 and launch_vel < 1 and asympt_vel < 1
        assert launch_r > 1
        assert launch_theta < 90 and launch_theta >= 0
        assert abs(asympt_vel) < 1

        length = max_z // z_res + 1 
        self.z_vals = np.linspace(0, max_z, length)
        if v_vec is not None:
            self.poloidal_vel = v_vec
            vector = np.zeros(length)
            for jj in range(length):
                if jj > 0:
                    vector[jj] = v_vec[jj] - v_vec[jj-1] / v_vec[jj]
                else:
                    vector[jj] = (v_vec[jj+1] - v_vec[jj]) / v_vec[jj+1]
            self.dpol_vel_dz_on_vel = vector
        else:
            vector = np.zeros(length)
            for jj in range(length):
                if jj * self.z_res >= launch_z:
                    pol_position = (((jj+0.5) * z_res * np.tan(self.launch_theta))**2 + (((jj+0.5) * z_res)**2))**0.5
                    vector[jj] = self.launch_vel + (self.asympt_vel - self.launch_vel) * ((pol_position / char_dist)**alpha / ((pol_position / char_dist)**alpha + 1))
            self.poloidal_vel = vector
            vector = np.zeros(length)
            for jj in range(length):
                if jj * self.z_res >= launch_z:
                    pol_position = (((jj+0.5) * z_res * np.tan(self.launch_theta))**2 + (((jj+0.5) * z_res)**2))**0.5
                    vector[jj] = ((self.asympt_vel - self.launch_vel) * ((pol_position / char_dist)**(alpha-1) * alpha / (((pol_position / char_dist)**alpha + 1)**2 * np.cos(self.launch_theta)))) / self.poloidal_vel[jj]
            self.dpol_vel_dz_on_vel = vector
        if r_vec is not None:
            assert len(r_vec) == len(v_vec)
            assert len(r_vec) == length
            self.radii = r_vec
        else:
            vector = np.linspace(launch_r, max_z * np.tan(self.launch_theta) + launch_r, length)
            self.radii = vector

        
        



        

















