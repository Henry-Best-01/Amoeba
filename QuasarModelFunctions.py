'''
This file simply holds all functions from QuasarModel
'''
from numpy import *
import numpy as np
from astropy import units as u
from astropy import constants as const
from scipy.integrate import quad
import QuasarModelFunctions as QMF
from astropy.io import fits
from numpy.random import rand
import astropy



c = const.c                     		
G = const.G	                         	
sigma = const.sigma_sb                   
h = const.h                                  
k = const.k_B                             
M_Proton = const.m_p                           
Thompson_Cross_Section = const.sigma_T
pi = np.pi


def CreateMaps(mass_exp, redshift, numGRs, inc_ang, resolution, spin=0, disk_acc = const.M_sun.to(u.kg)/u.yr, temp_beta=0, coronaheight=6,
                           albedo=1, eta=0.1, genericbeta=False, eddingtons=None):
        '''
        This function sets up maps required for the FlatDisk class in Amoeba. The following parameters are required:
         -mass_exp, the exponent to how many solar masses the black hole is modeled to have. M_bh = 10**mass_exp * M_sun
         -redshift, the position of the modeled quasar
         -numGRs, the amount of gravitational radii the disk is calculated out to
         -inc_ang, the inclination angle of the accretion disk
         -resolution, the amount of pixels along 1 axis. All images are created square.
         -spin, the dimensionless spin parameter of the modeled black hole, bounded on [-1, 1]
         -disk_acc, the amount of mass accreted by the accretion disk per time. If a number is given, units of solar_masses/year are assumed.
         -temp_beta, a wind parameter which serves to adjust the temperature profile (genericbeta==True will force r^-beta dependence instead by calculating required "beta")
         -coronaheight, number of grav. radii above the accretion disk the assumed lamppost is
         -albedo, reflectivity of disk. Setting to 0 will make the disk absorb emission, heating it up
         -eta, lamp post source luminosity coefficient. Defined as Lx = eta * M_dot * c^2
        The output is 6 values (mass_exp, redshift, numGRs, inc_ang, coronaheight, spin) and 3 surface maps (img_temp, img_vel, img_g, img_r)
        These are all recorded for conveninence, as they all get put into the FlatDisk constructor in order.
        '''
        import sim5

        assert redshift >= 0
        assert inc_ang > 0
        assert inc_ang < 90
        assert abs(spin) < 1
        assert temp_beta >= 0
        bh_mass = 10**mass_exp*const.M_sun.to(u.kg) 
        bh_rms = sim5.r_ms(spin)
        gravrad = QMF.CalcRg(bh_mass)
        img_temp = np.zeros((resolution, resolution))
        img_vel = img_temp.copy()
        img_g = img_temp.copy()
        img_r = img_temp.copy()
        for iy in range(resolution):
                for ix in range(resolution):
                        alpha = ((ix + 0.5)/resolution - 0.5) * 2.0*numGRs
                        beta = ((iy + 0.5)/resolution - 0.5) * 2.0*numGRs
                        gd = sim5.geodesic()
                        error = sim5.intp()
                        sim5.geodesic_init_inf(inc_ang * np.pi/180, abs(spin), alpha, beta, gd, error)
                        if error.value(): continue
                        P = sim5.geodesic_find_midplane_crossing(gd, 0)
                        if isnan(P): continue
                        r = sim5.geodesic_position_rad(gd, P)
                        if isnan(r): continue
                        if r >= QMF.SpinToISCO(spin):
                                phi = np.arctan2((ix-resolution/2), (iy-resolution/2))
                                img_vel[ix, iy] = -QMF.KepVel(r * gravrad, bh_mass) * np.sin(inc_ang * np.pi/180) * np.sin(phi) 
                                img_g[ix, iy] = sim5.gfactorK(r, abs(spin), gd.l)
                        img_r[ix, iy] = r
        nISCOs = QMF.SpinToISCO(spin)
        img_temp = QMF.AccDiskTemp(img_r*gravrad, nISCOs*gravrad, bh_mass, disk_acc, beta=temp_beta, coronaheight=coronaheight, albedo=albedo, eta=eta, genericbeta=genericbeta, eddingtons=eddingtons)
        return mass_exp, redshift, numGRs, inc_ang, coronaheight, spin, img_temp, img_vel, img_g, img_r



def KepVel (r, M):
        '''
        This calculates the magnitude of Keplerian Velocity at a distance r, on the Acc. Disk
        r should be in meters
        M should be in solar masses
        '''
       
        if type(M) != u.Quantity:
                M *= const.M_sun.to(u.kg)
        if type(r) != u.Quantity:
                r *= u.m
        if r == 0: return(0)
        else:
                return ((G * M.to(u.kg) / r.to(u.m) )**(0.5))/c
        

def SpinToISCO(spin):
        '''
        This function converts the dimensionless spin parameter into the ISCO size in units R_g
        '''
        z1 = 1 + (1-spin**2)**(1/3) * ( (1 + spin)**(1/3) + (1 - spin)**(1/3))
        z2 = (3 * spin**2 + z1**2)**(1/2)
        return 3 + z2 - np.sign(spin) * ( (3 - z1) * (3 + z1 + 2 * z2) )**(1/2)


def EddingtonRatioToMDot(mass, eddingtons, efficiency = 0.1):
        '''
        This function converts an Eddington Ratio (i.e. 0.15) into the corresponding accretion rate in physical units
        '''
        if type(mass) != u.Quantity:
                mass *= u.kg
        LEdd = 4*np.pi*G*mass*M_Proton*c/Thompson_Cross_Section
        L = LEdd * eddingtons
        return L / (efficiency * c**2)


def AccDiskTemp (R, R_min, M, M_acc, beta=0, coronaheight=6, albedo=1, eta=0.1, genericbeta=False, eddingtons=None):
        '''
        This function aims to take the viscous Thin disk and allows multiple additional modifications.
        Base Thin disk requires:
                R = radius [m]
                R_min = ISCO radius [m]
                M = mass [kg]
                M_acc = Accretion rate at ISCO [M_sun / yr]
        Keep all other defaults to output Thin disk model.
        
        A wind effect which acts to remove accreting material and adjusts the slope may add:
                beta > 0  is the wind strength. This directly effects the temperature gradient. See Sun+ 2018
        A corona heating effect due to lamp post geometry (Cacket+ 2007) may add:
                coronaheight = lamp post height in gravitational radii. Default is 6, the Schwarzschild ISCO case.
                albedo = reflection coefficent, 0 being perfect absorption and 1 being perfect reflectivity. Default is 1, meaning no heating from lamp post term.
                eta = strength coefficient of lamp post source, defined as Lx = eta * L_bol, where L_bol = M_acc c^2.
        Two further arguments are included:
                genericbeta = True if you want your profile to take the form r^(-beta). The beta relating to the disk+wind model will be worked out and used.
                eddingtons = ratio. If included, M_acc will be calculated in order to produce the desired eddington ratio.                
        '''
        if genericbeta == True:
                dummy = 3 - 4*beta
                beta = dummy
        if eddingtons:
                M_acc = QMF.EddingtonRatioToMDot(M, eddingtons)
        if type(R) == u.Quantity:
                R = R.to(u.m)
        if type(R_min) == u.Quantity:
                R_min = R_min.to(u.m)
        if type(M_acc) == u.Quantity:
                M_acc = M_acc.to(u.kg/u.s)
        else:
                M_acc *= const.M_sun.to(u.kg)/u.yr.to(u.s)  #Assumed was in M_sun / year
                M_acc = M_acc.value
        
        if type(M) == u.Quantity:
                M = M.to(u.kg)
        else:
                M *= const.M_sun.to(u.kg)  #Assumed was in M_sun
                M = M.value
        Rs = 2 * QMF.CalcRg(M)
        r = R/Rs
        r_in = R_min/Rs
        m0_dot = M_acc / (r_in**beta)
        coronaheight += 0.5               # Avoid singularities
        coronaheight *= QMF.CalcRg(M)

        zeroes = R>R_min
        
        tempmap = ( ( (3.0 * G * M * m0_dot * (1.0 - ((r_in) / r)**(0.5))) / (8.0 * pi * sigma * Rs**3) )**(0.25)).decompose().value * (r**(-(3-beta)/4))
        visc_temp = tempmap

        geometric_term = ((1-albedo)*coronaheight/(4*pi*sigma*(R**2+coronaheight**2)**(3/2))).decompose().value
        Lx = (eta * M_acc * c**2).decompose().value

        temp = (visc_temp**4 + geometric_term * Lx)**(1/4) * zeroes
        return np.nan_to_num(temp)
        

def PlanckLaw (T, lam): 
        '''
        I plan to pass in lam in units of [nm]. Otherwise, attach the units and it will convert.
        '''
        if type(lam) == u.Quantity:
                dummyval = lam.to(u.m)
                lam = dummyval.value
        elif type(lam) != u.Quantity:
                dummyval = lam * u.nm.to(u.m)
                lam = dummyval
        
        return np.nan_to_num(2.0 * h.value * c.value**2 * (lam)**(-5.0) * ((e**(h.value * c.value / (lam * k.value * T)) - 1.0)**(-1.0)))  # This will return the Planck Law wavelength function at the temperature input


def PlanckDerivative(T, lam):
        '''
        Numerical calculation
        '''
        PlanckA = QMF.PlanckLaw(T, lam)
        PlanckB = QMF.PlanckLaw(T+1, lam)
        return PlanckB-PlanckA
        

def PullValue (MagMap2d, X, Y):
        '''
        This takes a generic point (X,Y) off the magnification map and returns its value and includes
        decimal values
        '''
        assert X > 0 and Y > 0
        assert X < np.size(MagMap2d, 0) and Y < np.size(MagMap2d, 1)
        x = X//1
        y = Y//1
        decx = X%1
        decy = Y%1
        baseval = MagMap2d[int(x), int(y)]
        dx = (MagMap2d[int(x)+1, int(y)] - MagMap2d[int(x), int(y)]) * decx
        dy = (MagMap2d[int(x), int(y)+1] - MagMap2d[int(x), int(y)]) * decy
        return (MagMap2d[int(x), int(y)] + dx + dy)


def CalcRg(mass):
        '''
        This function simply returns what the length (in meters) of a geometric unit is for a given mass (in kg)
        '''
        if type(mass) != u.Quantity:
                mass *= u.kg
        return (G * mass / c**2).decompose().value
        

def ConvertMagMap(MagMap):
        '''
        The aim of this is to convert the 1-dim form of the lensing maps into a 2-dim image.
        Once it's done, the 2d map can (and should) be saved as a fits file for future use
        '''
        res = int(size(MagMap)**0.5)
        MapXY = zeros([res, res])
        for i in range(res):
                for j in range(res):
                        MapXY[i, j] = MagMap[res*i + j]

        return(MapXY)


def CalcAngDiamDist(redshift, Om0=0.3, OmL=0.7, little_h = 0.7): 
        '''
        This funciton takes in a redshift value of z, and calculates the angular diameter distance. This is given as the
        output. This assumes LCDM model.
        little_h is the value such that H_0 = little_h * 100 km s^-1 Mpc^-1. Leaving little_h=0.7 sets H_0 = 70 km s^-1 Mpc^-1.
        '''
        multiplier = (9.26* 10 **25) * (little_h)**(-1) * (1 / (1 + redshift))          # This need not be integrated over
        integrand = lambda z_p: ( Om0 * (1 + z_p)**(3.0) + OmL )**(-0.5)                # This must be integrated over
        integral, err = quad(integrand, 0, redshift)
        value = multiplier * integral * u.m
        return(value)


def CalcAngDiamDistDiff(redshift1, redshift2, Om0=0.3, OmL=0.7, little_h = 0.7):
        '''
        This function takes in 2 redshifts, designed to be z1 = redshift (lens) and z2 = redshift (source). It then
        integrates the ang. diameter distance between the two. This assumes LCDM model.
        h is defined as in CalcAngDiamDist
        '''
        multiplier = (9.26* 10 **25) * (little_h)**(-1) * (1 / (1 + redshift2))
        integrand = lambda z_p: ( Om0 * (1 + z_p)**(3.0) + OmL )**(-0.5)               # This must be integrated over
        integral1, err1 = quad(integrand, 0, redshift1)
        integral2, err2 = quad(integrand, 0, redshift2)
        value = multiplier * (integral2 - integral1) * u.m
        return(value)


def CalcLumDist(redshift, Om0=0.3, OmL=0.7, little_h = 0.7): 
        '''
        This calculates the luminosity distance using the CalcAngDiamDist formula above for flat lam-CDM model
        '''
        return (1 + redshift)**2 * QMF.CalcAngDiamDist(redshift, Om0=Om0, OmL=OmL, little_h = little_h)


def CalcRe (redshift_lens, redshift_source, M_lens=((1)) * const.M_sun.to(u.kg), Om0=0.3, OmL=0.7, little_h=0.7):#
        '''
        This function takes in values of z_lens and z_source. The output is the
        Einstein radius of the lens, in radians. This assumes LCDM model.
        '''
        D_lens = CalcAngDiamDist(redshift_lens, Om0=Om0, OmL=OmL, little_h=little_h)
        D_source = CalcAngDiamDist(redshift_source, Om0=Om0, OmL=OmL, little_h=little_h)
        D_LS = CalcAngDiamDistDiff(redshift_lens, redshift_source, Om0=Om0, OmL=OmL, little_h=little_h)
        value =( (( 4 * G * M_lens / c**2) * D_LS / (D_lens * D_source))**(0.5)).value
        return(value)


def ConvolveMaps(MagMap, disk, redshift_lens = 0.5, redshift_source = 2.1, mass_exp = 8.0, mlens = 1.0*const.M_sun.to(u.kg),
                nmapERs = 25, numGRs = 100, rotation=0, verbose=False, returnmag2d=False): 
        '''
        This makes the convolution between a disk and a magnification map. The difference is we physically know the screen size
        in physical units, as opposed to the field of view calculation required for GYOTO disks.
        
        '''
        from scipy.fft import fft2, ifft2
        from scipy.ndimage import rotate
        from skimage.transform import rescale 
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
        if type(rotation) != int and type(rotation) != float:
                rotation = np.random.rand() * 360
        newimg = rotate(disk, rotation, axes=(0, 1), reshape=False)
        disk = newimg
        if verbose==True: print("Disk Rotated by "+str(rotation)[:4]+" degrees")
        if MagMap.ndim == 2:
                MagMap2d = MagMap
        else:                   
                MagMap2d = QMF.ConvertMagMap(MagMap)
                if verbose==True: print('Magnification Map Changed. Shape =', np.shape(MagMap2d))
        mquasar = 10**mass_exp*const.M_sun.to(u.kg)
        diskpxsize = numGRs * QMF.CalcRg(mquasar)*u.m / diskres
        pixelsize = QMF.CalcRe(redshift_lens, redshift_source, M_lens = mlens) * QMF.CalcAngDiamDist(redshift_source) * nmapERs / np.size(MagMap2d, 0)
        if verbose==True: print('A pixel on the mag map is', pixelsize)
        if verbose==True: print('A pixel on the disk map is', diskpxsize)

        pixratio = diskpxsize.value/pixelsize.value
        dummydiskimg = rescale(disk, pixratio)
        disk = dummydiskimg
        if verbose==True: print("The disk's shape is now:", np.shape(disk))    
        
        dummymap = np.zeros(np.shape(MagMap2d))
        dummymap[:np.size(disk, 0), :np.size(disk, 1)] = disk
        convolution = ifft2(fft2(dummymap) * fft2(MagMap2d))
        output = convolution.real

        pixel_shift = np.size(disk, 0)//2
                        
        if verbose==True: print("Convolution Completed")
        
        if returnmag2d==True:
                return output, pixelsize, pixel_shift, MagMap2d
        return output, pixelsize, pixel_shift

        
def PullLC(convolution, pixelsize, vtrans, time, px_shift=0, x_start=None, y_start=None, phi_angle=None, returntrack=False): 
        '''
        Returning the track will allow both plotting tracks on the magnification map and also comparing different
        models along identical tracks.
        '''
        if type(vtrans) == u.Quantity:
                vtrans = vtrans.to(u.m/u.s) 
        else:
                vtrans *= u.km.to(u.m)
        if type(time) == u.Quantity:
                time = time.to(u.s)
        else:
                time *= u.yr.to(u.s)
        length_traversed = vtrans * time
        
        if type(length_traversed/pixelsize) == u.Quantity:
                length_traversed = (length_traversed/pixelsize).value
                pixelsize=1
        px_traversed = int(length_traversed / pixelsize + 0.5)

        xbounds = [abs(px_traversed)+2*px_shift, np.size(convolution, 0)-abs(px_traversed)]
        ybounds = [abs(px_traversed)+2*px_shift, np.size(convolution, 1)-abs(px_traversed)]

        if x_start:
                assert x_start >= xbounds[0] and x_start <= xbounds[1]
                xstart = x_start
        else: xstart = xbounds[0] + rand() * (xbounds[1] - xbounds[0])
        if y_start:
                assert y_start >= ybounds[0] and y_start <= ybounds[1]
                ystart = y_start
        else: ystart = ybounds[0] + rand() * (ybounds[1] - ybounds[0])
        if phi_angle:
                angle = phi_angle*np.pi/180
        else: angle = rand() * 2*np.pi
                
        startposition = [xstart, ystart]

        xtraversed = px_traversed * np.cos(angle)
        ytraversed = px_traversed * np.sin(angle)

        xpositions = np.linspace(startposition[0], startposition[0]+xtraversed, px_traversed) + px_shift
        ypositions = np.linspace(startposition[1], startposition[1]+ytraversed, px_traversed) + px_shift
        light_curve = []
        for tt in range(px_traversed):
                light_curve.append(PullValue(convolution, xpositions[tt], ypositions[tt]))

        track = [xpositions, ypositions]

        if returntrack==True:
                return light_curve, track
        else:
                return light_curve
        

def MakeTimeDelayMap(disk, inc_ang, massquasar = 10**8 * const.M_sun.to(u.kg), redshift = 0.0, diskres = 300,
                       numGRs = 100, coronaheight = 5, axisoffset=0, angleoffset=0, unit='hours', jitters=True, radiimap=None): 
        '''
        This aims to create a time delay mapping for the accretion disk for reverberation of the disk itself.
        The input disk should be some accretion disk map which is just used to determine where the accretion disk appears
        The output disk is a time delay mapping in units of lightdays. This can be scaled for larger accretion disks, or for different
        pixel sized maps. Due to linear nature of speed of light, only one needs to be used for any particular viewing angle*
                *if ISCO changes, this will produce more signal similar to input, and won't change overall reverb much!
                *only approximate around shadow of Black hole!
        axisoffset is the offset from the axis of rotation in r_g of the source
        angleoffset is the phi angle offset--0 deg is towards observer while 180 deg is away.
        '''
        inc_ang *= np.pi/180
        angleoffset *= np.pi/180

        if type(disk) == str:
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
                elif unit == 'GR' or unit == 'Gr' or unit == 'gr' or unit == 'Rg' or unit == 'rg' or unit == 'RG' or unit == 'R_g' or unit == 'r_g' or unit == 'R_G':
                        steptimescale = QMF.CalcRg(massquasar)
                else:
                        print('Invalid string deteted. Try "days", "hours", "minutes", "seconds" or an astropy.unit.\nReverting to hours.')
                        steptimescale = 3e8*60*60
        elif type(unit) == astropy.units.core.Unit or type(unit) == astropy.units.core.IrreducibleUnit:
                steptimescale = 3e8 / u.s.to(unit)
        else:
                print('Invalid unit deteted. Try "days", "hours", "minutes", "seconds" or an astropy.unit.\nReverting to hours.')
                steptimescale = 3e8*60*60

        gr = QMF.CalcRg(massquasar)        
        rstep = numGRs * gr / diskres
        time_resolution = rstep/steptimescale
        
        if jitters==True:
                timeshifts = np.random.uniform(size=np.shape(disk)) * time_resolution/np.cos(inc_ang)  # Helps smooth the transfer function by essentially randomizing position in pixel
        else:
                timeshifts = np.zeros(np.shape(disk))

        xoffset = axisoffset*gr*np.sin(angleoffset)
        yoffset = axisoffset*gr*np.cos(angleoffset)
        zp = ((coronaheight+0.5)*gr)

        output = np.zeros(np.shape(disk))

        mask1 = disk>0

        xx = np.linspace(-diskres/2 * rstep, diskres/2 * rstep, diskres) 
        yy = np.linspace(-diskres/2 * rstep, diskres/2 * rstep, diskres) / np.cos(inc_ang)
        xxx, yyy = np.meshgrid(xx, yy)
        rrr, phiphiphi = QMF.CartToPolar(xxx-xoffset, yyy-yoffset)
        phiphiphi += np.pi/2                                                    # 0 degrees should be to observer, aimed at negative y direction
        if radiimap is not None:
                rrr = radiimap * gr
        output = ((rrr**2 + zp**2)**0.5 + zp*np.cos(inc_ang) - rrr*np.cos(phiphiphi)*np.sin(inc_ang))/steptimescale
        safe_mask = output < (np.max(output) - time_resolution/np.cos(inc_ang))  #We don't want to *sometimes* increase the maximum time lag based on random variance.
        output += timeshifts * safe_mask 

        mask2 = output >= 0
        mask = np.logical_and(mask1, mask2)
        output *= mask
        output *= (1+redshift)
        return output


def ConstructGeometricDiskFactor(disk, inc_ang, massquasar, coronaheight, axisoffset=0, angleoffset=0, numGRs=100, albedo=0, radiimap=None):
        '''
        This function creates the geometric factor of the accretion disk:
        (1-A)cos(theta_x)/(4*pi*sigma*R_{*}^{2})  #Additional term which multiplies Lx in Eq. 2 of Cackett+ 2007
        This takes parameters:
        disk is a two dimensional map
        inc_ang is the inclination angle of the disk
        massquasar is the mass of the central black hole
        coronaheight is the height of the flaring source in Rg
        axisoffset / angleoffset determines how off axis the flaring source is, in Rg and degrees respectively
        numGRs is the number of gravitational radii the accretion disk map is calculated out to (radially)
        albedo is the reflectivity of the accretion disk in the range 0, 1
        '''
        if disk.ndim == 2:
                output = np.zeros(np.shape(disk))
        elif disk.ndim == 3:
                output = np.zeros(np.shape(disk[:,:,-1]))
        assert albedo <= 1 and albedo >= 0
        inc_ang *= np.pi/180
        gr = QMF.CalcRg(massquasar)
        rstep = numGRs * gr / np.size(disk, 0)

        axisoffset = axisoffset
        angleoffset = angleoffset * np.pi/180
        sig = const.sigma_sb.value
        diskres = np.size(disk, 0)
        
        xoffset = axisoffset * gr * np.sin(angleoffset) 
        yoffset = axisoffset * gr * np.cos(angleoffset)

        mask1 = disk>0
        xx = np.linspace(-diskres/2 * rstep, diskres/2 * rstep, diskres)
        yy = np.linspace(-diskres/2 * rstep, diskres/2 * rstep, diskres) / np.cos(inc_ang)
        xxx, yyy = np.meshgrid(xx, yy)
        rrr, phiphiphi = QMF.CartToPolar(xxx-xoffset, yyy-yoffset)
        phiphiphi += np.pi/2
        if radiimap is not None:                                                            # allow passing in radii map if precalculated
                rrr = radiimap * gr
        coronaheight += 1/2                                                     # adding 0.5 Rg to h avoids dividing by zero
        distance = (rrr**2 + ((coronaheight) * gr)**2)**0.5
        costheta =((coronaheight) * gr)/distance
        output = (1-albedo) * costheta / (4 * np.pi * sig * distance**2)
        mask2 = output >= 0
        mask = np.logical_and(mask1, mask2)
        output *= mask
        return output


def MakeDTDLx(disk_der, temp_map, inc_ang, massquasar, coronaheight, numGRs = 100, axisoffset=0, angleoffset=0, radiimap=None):
        '''
        Approximates DT/DLx on the accretion disk assuming the irradiated disk model, such that:
        T_disk**4 = T_viscous**4 + T_irradiation
        (T_disk + delta_T)**4 = T_viscous**4 + T_irradiation + delta_Lx * geometric_factor
        T_disk**4 + 4 * delta_T * T_disk**3 + (order delta_T**2+)... = T_disk**4 + delta_Lx * geometric_factor
        delta_T / delta_Lx ~ geometric_factor / (4*T_disk**3)
        '''
        T_orig = temp_map
        weightingmap = QMF.ConstructGeometricDiskFactor(disk_der, inc_ang, massquasar, coronaheight, numGRs=numGRs, axisoffset=axisoffset, angleoffset=angleoffset, radiimap=radiimap)
        output = weightingmap/(4*(T_orig)**3)
        
        return output
                                

def ConstructDiskTransferFunction(image_der_f, temp_map, inc_ang, massquasar, redshift, coronaheight, maxlengthoverride=4800, units='hours', axisoffset=0, angleoffset=0, numGRs =100, 
                                  albedo=0, smooth=True, fixedwindowlength=None, radiimap=None): 
        '''
        This takes in disk objects and parameters in order to construct its transfer function
        The disk should be the derivative of the Planck function w.r.t Temperature
        coronaheight is the position of the point source
        r, phi are for off-axis point sources
        all distances are input in R_g
        '''
        
        from scipy.signal import savgol_filter
        import numpy as np
        import QuasarModelFunctions as QMF
        import astropy.units as u
        import astropy.constants as const

        dummy=np.nan_to_num(image_der_f)
        image_der_f = dummy

        if image_der_f.ndim == 2:
                diskdelays = QMF.MakeTimeDelayMap(image_der_f, inc_ang, massquasar = massquasar, redshift = redshift, coronaheight = coronaheight, unit = units,
                                                    axisoffset = axisoffset, angleoffset = angleoffset, numGRs = numGRs, radiimap=radiimap)
                minlength = int(np.min(diskdelays * (diskdelays>0)))
                maxlength = int(np.max(diskdelays)+10)
                dummy = min(maxlength, maxlengthoverride)
                maxlength = dummy
                windowlength = int((maxlength-minlength)/50)+5
                output = np.zeros((maxlength))
                
        elif image_der_f.ndim == 3:
                diskdelays = QMF.MakeTimeDelayMap(image_der_f[:,:,-1], inc_ang, massquasar = massquasar, redshift = redshift, coronaheight = coronaheight, unit = units,
                                                    axisoffset = axisoffset, angleoffset = angleoffset, numGRs = numGRs, radiimap=radiimap)
                minlength = int(np.min(diskdelays * (diskdelays>0)))
                maxlength = int(np.max(diskdelays)+10)
                dummy = min(maxlength, maxlengthoverride)
                maxlength = dummy
                windowlength = int((maxlength-minlength)/50)+5
                output = np.zeros((maxlength, np.size(image_der_f, 2)))
        else:
                print("Invalid dimensionality of input disk")
                return null
        if fixedwindowlength: windowlength = fixedwindowlength
        if windowlength %2 == 0: windowlength += 1                                                              # Savgol filter window must be odd
        weight = np.nan_to_num(QMF.MakeDTDLx(image_der_f, temp_map, inc_ang, massquasar, coronaheight, numGRs=numGRs, axisoffset=axisoffset, angleoffset=angleoffset, radiimap=radiimap))

        output = np.histogram(diskdelays, range=(0, np.max(diskdelays)+1), bins=int(np.max(diskdelays)+1), weights=np.nan_to_num(weight*image_der_f), density=True)[0]

        if smooth==True:

                zeromask = np.ones(np.shape(output))
                if image_der_f.ndim==2:
                        
                        if np.argmax(output) > 1:
                                dummy_slice = output[:np.argmax(output)] * (output[:np.argmax(output)] > 0)
                                last_zero = np.size(dummy_slice) - np.argmin(np.flip(dummy_slice))              # last_zero is defined as the shortest time delay in the transfer function
                                zeromask[:last_zero] = 0
                                
                                                                             
                        smoothedoutput = savgol_filter(output, windowlength, 3)
                        smoothedoutput *= (smoothedoutput>0) * zeromask
                        smoothedoutput /= np.sum(smoothedoutput)
                else:
                        smoothedoutput = np.zeros(np.shape(output))
                        
                                
                        for band in range(np.size(image_der_f, 2)):
                                if np.argmax(output[band, :]) > 1:
                                        dummy_slice = output[band, :np.argmax(output)] * (output[band, np.argmax(output[band, :])] > 0)
                                        last_zero = np.size(dummy_slice) - np.argmin(np.flip(dummy_slice))
                                        zeromask[band, :last_zero] = 0
                                smoothedoutput[:,band] = savgol_filter(output[:,band], windowlength, 3)
                        smoothedoutput *= (smoothedoutput>0) * zeromask
                        for band in range(np.size(image_der_f, 2)):
                                smoothedoutput[:,band] /= np.sum(smoothedoutput[:,band])
                return smoothedoutput
        if image_der_f.ndim==2:
                output /= np.sum(output)
        else:
                for band in range(np.size(image_der_f, 2)):
                        output[:,band] /= np.sum(output[:,band])
        return output


def MicrolensedResponse(MagMap, AccDisk, wavelength, coronaheight, rotation=0, x_position=None, y_position=None,
                        axisoffset=0, angleoffset=0, unit='hours', smooth=False, returnmaps=False, radiimap=None):
        '''
        This function aims to microlens the response from a fluctuation in the lamppost geometry at some position
        on the magnification map
        '''
        from skimage.transform import rescale
        from scipy.ndimage import rotate
        from numpy.random import rand
        from scipy.signal import savgol_filter
        reprocessedmap = AccDisk.MakeDBDTMap(wavelength)
        pxratio = AccDisk.pxsize/MagMap.px_size
        adjusteddisk = np.nan_to_num(rescale(reprocessedmap*AccDisk.MakeDTDLxMap(wavelength, axisoffset=axisoffset,
                                                           angleoffset=angleoffset), pxratio))
        if returnmaps == True:
                adjustedtimedelays = rescale(AccDisk.MakeTimeDelayMap(axisoffset=axisoffset, 
                                                            angleoffset=angleoffset, unit=unit, jitters=False), pxratio)
        else:
                adjustedtimedelays = rescale(AccDisk.MakeTimeDelayMap(axisoffset=axisoffset, 
                                                            angleoffset=angleoffset, unit=unit), pxratio)
        maxrange = np.max(adjustedtimedelays)+1
        edgesize = np.size(adjusteddisk, 0)                                             # This is the edge length we must avoid

        if type(rotation) != int and type(rotation) != float:
                rotation = np.random.rand() * 360
        newimg1 = rotate(adjusteddisk, rotation, axes=(0, 1), reshape=False)
        adjusteddisk = newimg1
        newimg2 = rotate(adjustedtimedelays, rotation, axes=(0, 1), reshape=False)
        adjustedtimedelays = newimg2
        
        while 3*edgesize > MagMap.resolution:
                print("Disk too large, or Magnification Map must be larger! Adjusting...")
                smallerdisk = adjusteddisk[edgesize//2-edgesize//4:edgesize//2+edgesize//4, edgesize//2-edgesize//4:edgesize//2+edgesize//4]
                smallerTDs = adjustedtimedelays[edgesize//2-edgesize//4:edgesize//2+edgesize//4, edgesize//2-edgesize//4:edgesize//2+edgesize//4]
                edgesize = np.size(smallerdisk, 0)
                adjusteddisk = smallerdisk
                adjustedtimedelays = smallerTDs
    
        if x_position: 
                if x_position - edgesize > 0 and x_position + edgesize//2 < MagMap.resolution:
                        xposition = x_position
                else: 
                        xposition = int(edgesize + rand() * (MagMap.resolution - 2*edgesize))
        else: 
                xposition = int(edgesize + rand() * (MagMap.resolution - 2*edgesize))

        if y_position: 
                if y_position - edgesize > 0 and y_position + edgesize//2 < MagMap.resolution:
                        yposition = y_position
                else: 
                        yposition = int(edgesize + rand() * (MagMap.resolution - 2*edgesize))
        else: 
                yposition = int(edgesize + rand() * (MagMap.resolution - 2*edgesize))
        
        xposition -= edgesize//2
        yposition -= edgesize//2
    
        magnifiedresponse = np.nan_to_num(adjusteddisk * MagMap.mag_map[yposition:yposition+edgesize, xposition:xposition+edgesize])
    
        if returnmaps==True:
                return adjustedtimedelays, magnifiedresponse, xposition+edgesize//2, yposition+edgesize//2

        output = np.histogram(adjustedtimedelays, range=(0, maxrange), bins=int(maxrange), weights=np.nan_to_num(magnifiedresponse), density=True)[0]
        
        if smooth==True:
                if unit=='hours': windowlength = np.size(output)//50 + 5
                elif unit=='days': windowlength = np.size(output)//2 + 5
                if windowlength%2 == 0: windowlength += 1
                zeromask = np.ones(np.shape(output))
                        
                if np.argmax(output) > 1:
                        dummy_slice = output[:np.argmax(output)] * (output[:np.argmax(output)] > 0)
                        last_zero = np.size(dummy_slice) - np.argmin(np.flip(dummy_slice)) # last_zero is defined as the shortest time delay in the transfef function
                        zeromask[:last_zero] = 0
                                                                             
                smoothedoutput = savgol_filter(output, windowlength, 3)
                smoothedoutput *= (smoothedoutput>0) * zeromask
                smoothedoutput /= np.sum(smoothedoutput)
                
                output = smoothedoutput
        return output
        
                                

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
        

def MakeDRW(t_max, delta_t, SF_inf, tau):
        '''
        This sets up a Damped Random Walk for input into an intrinsic variability model. It uses the recursion formula
        found in Kelly, Bechtold & Siemiginowska (2009). This gets added to the continuum light curve after convoultion with a transfer function.
        Even spacing between observations is assumed.
        t_max is the full light curve time period. input as u.Quantity, or the assumption of years is made.
        delta_t is the spacing between observations. input as u.Quantity, or the assumption of days is made.
        SF_inf is the structure function asymptote
        tau is the characteristic time scale. input as u.Quantity, or the assumption of days is made.
        '''

        import numpy as np

        if type(t_max) != u.Quantity:
                t_max *= u.year
        if type(delta_t) != u.Quantity:
                delta_t *= u.day
        if type(tau) != u.Quantity:
                tau *= u.day
        n_obs = int((t_max / delta_t).decompose().value) + 1

        variability_lc = np.zeros(n_obs)

        for nn in range(n_obs-1):
                variability_lc[nn + 1] = (variability_lc[nn] * np.exp(-abs((delta_t)/tau).decompose().value) +
                        (SF_inf / 2**(1/2)) * random.normal() * ( 1 - (np.exp(-2 * abs((delta_t)/tau).decompose().value)))**(1/2))

        return variability_lc
        

def CartToPolar(x, y):
        '''
        This simply converts x, y coords into r, theta coords
        '''
        
        import numpy as np
        
        r = (x**2 + y**2)**0.5
        theta=np.arctan2(y, x)
        return(r, theta)


def PolarToCart(r, theta):
        '''
        This function switches r, theta coords back to x, y coords
        Theta is in radians
        '''
        
        import numpy as np
        
        x = r * np.sin(theta)
        y = r * np.cos(theta)
        return(x, y)

        

def MakeSnapshots(DiskEmission, DiskReprocess, DiskLags, SnapshotTimesteps, SignalWeighting=1, Signal=None,
                  SF_inf=None, tau=None, returnsignal = False):
        '''
        This function takes in maps of the emission, response, and timelags. Additionally, a
        list of timestamps are required. SignalWeighting is the relative strength of the
        heating signal. Signal, if given, is the driving signal. If not given, a DRW will
        be created.
        '''
        
        maxtime = np.max(DiskLags) + SnapshotTimesteps[-1] + 100  # The longest signal we need for desired timesteps
        diskmask = np.nan_to_num(DiskReprocess) > 1e-25
        print(np.sum(diskmask))

        if Signal is not None:
                if len(Signal) < maxtime:
                        print("Input signal is not long enough to cover all snapshots!")
                        return
        else:
                maxtime /= (365*24)  #Convert to hours
                from QuasarModelFunctions import MakeDRW
                if SF_inf is None: SF_inf = np.random.rand() * 300  # if not SF_inf, generate a random one between 0 to 300
                if tau is None: tau = np.random.rand() * 100  #if not tau, generate a random one between 0 and 100
                Signal = MakeDRW(maxtime, 1/24, SF_inf, tau)
                Signal -= np.mean(Signal)
                Signal /= np.std(Signal)
                Signal -= np.min(Signal)
                                 
                
        output = []

        for jj in SnapshotTimesteps:
                timestamps = DiskLags.astype(int) + jj
                output.append(np.nan_to_num(DiskEmission + SignalWeighting * DiskReprocess * Signal[timestamps]) * diskmask)
        if returnsignal == True:
                return output, Signal
        return output






        






        
                
























                

        

