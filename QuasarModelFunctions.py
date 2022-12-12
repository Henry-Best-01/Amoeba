'''
This file simply holds all functions from QuasarModel
'''

from numpy import *
from astropy import units as u
from astropy import constants as const
from scipy.integrate import quad


c = const.c                                                     # [m/s]			
G = const.G	                                     # [m^(3) kg ^(-1) s^(-2)]		
sigma = const.sigma_sb                                    # [W m^(-2) K^(-4)
h = const.h                                                      # [J s]		#Planck Const
k = const.k_B                                                   # [J/K]	#Boltzmann Const
M_Proton = const.m_p                                      # [kg]
Thompson_Cross_Section = const.sigma_T         # [m^2]

M = ((1.0*10**8)) * const.M_sun.to(u.kg)			#Mass of Black Hole (first number is in solar masses. 2e30 converts solar masses into kilograms)
M_acc = ((0.3323)) * (const.M_sun / u.yr).to(u.kg/u.s)                     #Mass Accreted/second (as above) Double par. is how many solar masses/year.
M_lens = ((1)) * const.M_sun.to(u.kg)                                             #Mass of the lensing object (first number is in solar masses. 
rho_0 = 10.0**(-10) * u.kg / (u.m)**3			#This is the maximum density of particles in the Acc. Disk. Parameterizes the length of the AD.
z = 0.0		                                                	#This is the redshift z scaling, describing how far away the quasar is.
H0 = 70                                                                                     # Hubble Constant
Om0 = 0.3                                                                                 # Omega_0 is percent matter
OmL = 0.7                                                                                 # Omega_L is percent cos. constant
OmK = 0                                                                                   # Omega_K is percent curvature (0 is flat)
Rotational_Parameter = ((0.0))                                                      # This is a/M. 0 reduces to Schwarzschild, while this asymptotes to either 1.5 or 9.0 x Schw. Rad. Negative corresponds to retrograde spin.
Eddingtons = 0.15                                                                      # This is how many Eddingtons the QSO is radiating at


def CreateMapsForDiskClass(mass_exp, redshift, nGRs, inc_ang, resolution, spin=0, disk_acc = const.M_sun.to(u.kg)/u.yr, temp_beta=0):
        '''
        This function sets up maps required for the ThinDisk class in Amoeba. The following parameters are required:
         -mass_exp, the exponent to how many solar masses the black hole is modeled to have. M_bh = 10**mass_exp * M_sun
         -redshift, the position of the modeled quasar
         -nGRs, the amount of gravitational radii the disk is calculated out to
         -inc_ang, the inclination angle of the thin disk
         -resolution, the amount of pixels along 1 axis. All images are created square.
         -spin, the dimensionless spin parameter of the modeled black hole, bounded on [-1, 1]
         -disk_acc, the amount of mass accreted by the accretion disk
         -temp_beta, a wind parameter which serves to adjust the temperature profile
        The output is 4 values, mass_exp, redshift, nGRs, inc_ang and 3 surface maps, img_temp, img_vel, img_g
        These are all recorded for conveninence, as they all get put into the ThinDisk constructor in order.
        '''
        import sim5
        import numpy as np
        import QuasarModelFunctions as QMF
        bh_mass = 10**mass_exp
        bh_rms = sim5.r_ms(spin)
        img_temp = np.zeros((resolution, resolution))
        img_vel = img_temp.copy()
        img_g = img_temp.copy()
        for iy in range(resolution):
                for ix in range(resolution):
                        alpha = ((ix + 0.5)/resolution - 0.5) * 2.0*nGRs
                        beta = ((iy + 0.5)/resolution - 0.5) * 2.0*nGRs
                        gd = sim5.geodesic()
                        error = sim5.intp()
                        sim5.geodesic_init_inf(inc_ang * np.pi/180, spin, alpha, beta, gd, error)
                        if error.value(): continue
                        P = sim5.geodesic_find_midplane_crossing(gd, 0)
                        if isnan(P): continue
                        r = sim5.geodesic_position_rad(gd, P)
                        if isnan(r): continue
                        if r >= QMF.SpinToISCO(spin):
                                phi = np.arctan2((ix-resolution/2), (iy-resolution/2))
                                img_temp[iy, ix] = QMF.AccDiskTempBeta(r*QMF.GetGeometricUnit(bh_mass*const.M_sun.to(u.kg)), QMF.SpinToISCO(spin)*QMF.GetGeometricUnit(bh_mass), disk_acc, bh_mass, temp_beta)
                                img_vel[iy, ix] = -QMF.KepVelocity(r * QMF.GetGeometricUnit(bh_mass*const.M_sun.to(u.kg)), bh_mass * const.M_sun.to(u.kg)) * np.sin(inc_ang * np.pi/180) * np.sin(phi) #Try to find a way to extract actual LoS velocity from this geodesic if possible!
                                img_g[iy, ix] = sim5.gfactorK(r, spin, gd.l)
        return mass_exp, redshift, nGRs, inc_ang, img_temp, img_vel, img_g


def KepVelocity (r, M):
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
        import numpy as np
        z1 = 1 + (1-spin**2)**(1/3) * ( (1 + spin)**(1/3) + (1 - spin)**(1/3))
        z2 = (3 * spin**2 + z1**2)**(1/2)
        return 3 + z2 - np.sign(spin) * ( (3 - z1) * (3 + z1 + 2 * z2) )**(1/2)
        
        

def AddVels (v1 = 0, v2 = 0, v3 = 0, output = 0):
        '''
        This Approximately adds relativistic velocities by converting to gamma factors, then adding together and returning beta.
        If units were not included, m/s units are assumed.
        output gives final velocity in units of c.
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
        gamma1 = (1 - beta1**2)**(-0.5) - 1  # The deviation of each gamma factor from 1 is added in this approx.
        gamma2 = (1 - beta2**2)**(-0.5) - 1  # Splitting velocities into components is not a great stratergy in the first place however.
        gamma3 = (1 - beta3**2)**(-0.5) - 1

        thresh = 0.01
        
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
        else: r *= u.m  #Assumed was in meters
        if type(R_min) == u.Quantity:
                R_min = R_min.to(u.m)
        else: R_min *= u.m  #Assumed was in meters
        if type(M_acc) == u.Quantity:
                M_acc = M_acc.to(u.kg/u.s)
        else: M_acc *= (const.M_sun/u.yr).to(u.kg/u.s)  #Assumed was in M_sun / year
        
        if type(M) == u.Quantity:
                M = M.to(u.kg)
        else: M *= const.M_sun.to(u.kg)  #Assumed was in M_sun
        if r < R_min:
                return 0*u.K
        else:
                return (((3.0 * G * M * M_acc * (1.0 - (R_min / r)**(0.5))) / (8.0 * pi * sigma * (r**3.0)) )**(0.25)).decompose()  # This calculates the temperature of the fluid on the Acc. Disk 

def AccDiskTempBeta (R, R_min, M_acc, M, beta):
        '''
        This similarly calculates the temp. of the Acc Disk, though with an additional beta parameter to modify dependence.
        This follows Eq 4 in Sun+ 2018.
        M_acc defines the accretion rate at the ISCO, as in Eq 1 of Sun+ for R = R_min.
        '''
        import QuasarModelFunctions as QMF
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
        Rs = 2 * QMF.GetGeometricUnit(M)
        r = R/Rs
        r_in = R_min/Rs
        m0_dot = M_acc / (r_in**beta)
        
        if R < R_min:
                return 0
        return( ( (3.0 * G * M * m0_dot * (1.0 - ((r_in) / r)**(0.5))) / (8.0 * pi * sigma * Rs**3) )**(0.25)).decompose().value * (r**((beta-3)/4))


def PlanckLaw (T, lam): #
        '''
        I plan to pass in lam in units of [nm]. Otherwise, attach the units and it will convert.
        '''
        if type(lam) == u.Quantity:
                dummyval = lam.to(u.m)
                lam = dummyval.value
        elif type(lam) != u.Quantity:
                dummyval = lam * u.nm.to(u.m)
                lam = dummyval
                
        
        return (2.0 * h.value * c.value**(2.0) * (lam)**(-5.0) * ((e**(h.value * c.value / (lam * k.value * T)) - 1.0)**(-1.0)))  # This will return the Planck Law wavelength function at the temperature input

def PlanckTempDerivative (T, lam): #
        '''
        This is the derivative of Planck's Law, with respect to temperature
        '''
        if type(lam) != u.Quantity:
                lam *= u.nm

        a = 2 * h**2 * c**4 / ((lam.to(u.m))**6.0 * k * T**2)
        b = e**(h * c / (lam.to(u.m) * k * T)).decompose()

        return a.value * b.value / (b.value - 1)**2

def MeasureMLAmp (MagMap2d, X, Y):
        '''
        This takes a generic point (X,Y) off the magnification map and returns its value and includes
        decimal values
        '''
        x = X//1
        y = Y//1
        decx = X%1
        decy = Y%1
        baseval = MagMap2d[int(x), int(y)]
        dx = (MagMap2d[int(x)+1, int(y)] - MagMap2d[int(x), int(y)]) * decx
        dy = (MagMap2d[int(x), int(y)+1] - MagMap2d[int(x), int(y)]) * decy
        return (MagMap2d[int(x), int(y)] + dx + dy)


        

def GetGeometricUnit(mass):#
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
        Once it's done, the 2d map can (and should) be saved as a fits file for future use
        '''
        res = int(size(MagMap)**0.5)
        MapXY = zeros([res, res])
        for i in range(res):
                for j in range(res):
                        MapXY[i, j] = MagMap[res*i + j]

        return(MapXY)


def ReadThroughput(file):
        
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


def AngDiameterDistance(z, Om0=0.3, OmL=0.7, h = 0.7): #
        '''
        This funciton takes in a redshift value of z, and calculates the angular diameter distance. This is given as the
        output. This assumes LCDM model.
        h is the value such that H_0 = h * 100 km s^-1 Mpc^-1. Leaving h=0.7 sets H_0 = 70 km s^-1 Mpc^-1.
'''
        multiplier = (9.26* 10 **25) * (h)**(-1) * (1 / (1 + z))                                      # This need not be integrated over
        integrand = lambda z_p: ( Om0 * (1 + z_p)**(3.0) + OmL )**(-0.5)               # This must be integrated over
        integral, err = quad(integrand, 0, z)
        value = multiplier * integral * u.m
        return(value)


def AngDiameterDistanceDifference(z1, z2, Om0=0.3, OmL=0.7, h = 0.7):#
        '''
        This function takes in 2 redshifts, designed to be z1 = redshift (lens) and z2 = redshift (source). It then
        integrates the ang. diameter distance between the two. This assumes LCDM model.
        h is defined as in AngDiameterDistance
'''
        multiplier = (9.26* 10 **25) * (h)**(-1) * (1 / (1 + z2))
        integrand = lambda z_p: ( Om0 * (1 + z_p)**(3.0) + OmL )**(-0.5)               # This must be integrated over
        integral1, err1 = quad(integrand, 0, z1)
        integral2, err2 = quad(integrand, 0, z2)
        value = multiplier * (integral2 - integral1) * u.m
        return(value)



def CalculateLuminosityDistance(z, Om0=0.3, OmL=0.7): #
        '''
        This calculates the luminosity distance using the AngdiameterDistance formula above for flat lam-CDM model
'''
        return((1 + z)**2 * AngDiameterDistance(z, Om0, OmL))


def CalcEinsteinRadius (z1, z2, M_lens=((1)) * const.M_sun.to(u.kg), Om0=0.3, OmL=0.7):#
        '''
        This function takes in values of z_lens and z_source (not simply by finding 
        the difference of the two! See AngDiameterDistanceDifference function above!). The output is the
        Einstein radius of the lens, in radians. This assumes LCDM model.
'''
        D_lens = AngDiameterDistance(z1, Om0, OmL)
        D_source = AngDiameterDistance(z2, Om0, OmL)
        D_LS = AngDiameterDistanceDifference(z1, z2, Om0, OmL)
        value =( (( 4 * G * M_lens / c**2) * D_LS / (D_lens * D_source))**(0.5)).value
        return(value)




def SaveCurve(time, brightness, ext, input_file):
        '''
        This funciton saves a light curve created in a .fits format. The header will contain information about how
        many high magnification events are detected, how long the curve lasts, and information regarding the
        brightness model input.
        I will define high magnification event as a 50% increase of magnification within 300 time units.
        Using 100% increase of magnification within 100 time units did not capture all the observed crossing events.
        Using 50% increase of magnification within 100 time units captured 2237's events, but not 0435's events.
        Using 50% increase within 1% of the time units

        time is the time axis of the light curve
        brightness is the light curve
        ext is the extention to be given to the file name
        input_file is the brightness map .fits file, used to extract some header information
'''
        with fits.open(input_file) as hdul:
                hdu = hdul[0].data
                ER = hdul[0].header['ER']
                wavelength = hdul[0].header['wavelen']
                velocity = hdul[0].header['velocity'] 
                pix_size = hdul[0].header['pixsize']

        events = 0
        max_time = max(time)                    # This simply stores the duration of the light curve
        for i in range(int(size(time)/50)):
                if (i) * int(size(time)/50) < size(brightness):
                        if max(brightness[(i * int(size(time)/50 + 0.5)):(i + 1) * int(size(time)/50 + 0.5)]) > min(brightness[i * int(size(time)/50):(i + 1) * int(size(time)/50 + 0.5)]) * 1.5 :
                                events += 1

        data = array([time, brightness])
        
        hdu2 = fits.PrimaryHDU(data)
        hdu2.header['ER'] = ER                            # In number
        hdu2.header['wavelen'] = wavelength         # In nm
        hdu2.header['velocity'] = velocity                # In pixels/year
        hdu2.header['pixsize'] = pix_size                # In meters/pixel
        hdu2.header['events'] = events                  # in number
        hdu2.header['duration'] = max_time              # In years
        hdu2.writeto('LightCurve'+str(ER)+'ER'+str(int(max_time))+'years'+str(wavelength)+'nm'+str(ext)+'.fits')

def CreateMagStrip(XYMap, startx, starty, endx, endy, width, steps):

        positions = zeros([steps + 1, 2])
        for i in range(steps + 1):                                                                  # These are the (x, y) positions at each time step, stored as [:, 0], [:, 1]
                positions[i, 0] = int((endx - startx) * i / steps + 0.5) + startx       
                positions[i, 1] = int((endy - starty) * i / steps + 0.5) + starty

        output = zeros([steps, width])
        halfwidth = int(width/ 2.0)
                
        if abs(endy - starty) > abs(endx - startx):
                # Moving in y direction primarily!
                for i in range(steps):
                        output[i, :] = XYMap[int(positions[i, 0]-halfwidth) : int(positions[i, 0]+halfwidth), int(positions[i, 1]) ]

        else:
                # Moving in x direction primarily!
                for i in range(steps):
                        output[i, :] = XYMap[int(positions[i, 0]), int((positions[i, 1]-halfwidth)) : int((positions[i, 1]+halfwidth))] 
                        

        return(output)

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
        else:
                nconvos = 1

        if sim5==True:
                for xxx in range(nconvos):
                        convo, pxsize = QMF.ConvolveSim5Map(MagMap, disk[:, :, xxx], zlens = zlens, zquasar = zquasar, mquasarexponent = mquasarexponent, mlens = mlens,
                                                            nmapERs = nmapERs, numGRs = numGRs, rotation=rotation, verbose=verbose)
                        if xxx == 0:
                                LCs = []
                                tracks = []
                                for nn in range(ncurves):
                                        LC, track = QMF.PullRandLC(convo.real, pxsize, vtrans, time, returntrack=True)  # Get initial LCs and tracks based on input parameters
                                        LCs.append(LC)
                                        tracks.append(track)
                                lencurve = len(LC)
                                pointsperslice = int(lencurve/nconvos)
                        else:
                                for nn in range(ncurves):
                                        for jjj in range(pointsperslice):
                                                LCs[nn][int(xxx*pointsperslice)+jjj] = MeasureMLAmp(convo, tracks[nn][0][int(xxx*pointsperslice)+jjj], tracks[nn][1][int(xxx*pointsperslice)+jjj])
        else:
                for xxx in range(nconvos):
                        convo, pxsize = QMF.ConvolveMap(MagMap, disk[:, :, xxx], diskres = diskres, zlens = zlens, zquasar = zquasar, mquasarexponent = mquasarexponent, mlens = mlens,
                                                        nmapERs = nmapERs, diskfov = diskfov, diskposition = diskposition, rotation=rotation, verbose=verbose)
                        if xxx == 0:
                                LCs = []
                                tracks = []
                                for nn in range(ncurves):
                                        LC, track = QMF.PullRandLC(convo.real, pxsize, vtrans, time, returntrack=True)  # Get initial LCs and tracks based on input parameters
                                        LCs.append(LC)
                                        tracks.append(track)
                                lencurve = len(LC)
                                pointsperslice = int(lencurve/nconvos)
                        else:
                                for nn in range(ncurves):
                                        for jjj in range(pointsperslice):
                                                LCs[nn][int(xxx*pointsperslice)+jjj] = MeasureMLAmp(convo, tracks[nn][0][int(xxx*pointsperslice)+jjj], tracks[nn][1][int(xxx*pointsperslice)+jjj])

        return (LCs, tracks)
                                

        

def ConvolveSim5Map(MagMap, disk, zlens = 0.5, zquasar = 2.1, mquasarexponent = 8.0, mlens = 1.0*const.M_sun.to(u.kg),
                nmapERs = 25, numGRs = 100, rotation=False, verbose=False, returnmag2d=False): #
        '''
        This makes the convolution between a Sim5 disk and a magnification map. The difference is we physically know the screen size
        in physical units, as opposed to the field of view calculation required for GYOTO disks.
        
        '''
        import numpy as np
        import QuasarModelFunctions as QMF
        import matplotlib.pyplot as plt
        from scipy.fft import fft2, ifft2
        from scipy.ndimage import rotate
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
                        
        if verbose==True: print("Convolution Completed")
        
        if returnmag2d==True:
                return output, pixelsize, MagMap2d
        return output, pixelsize


def ConvolveMap(MagMap, disk, diskres = 300, zlens = 0.5, zquasar = 2.1, mquasarexponent = 8.0, mlens = 1.0*const.M_sun.to(u.kg),
                nmapERs = 25, diskfov = 0.12, diskposition = 4000, rotation=False, verbose=False, returnmag2d=False):
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
        
        if verbose==True: print("Convolution Completed")
        
        if returnmag2d==True:
                return output, pixelsize, MagMap2d
        return output, pixelsize
                            
def PullLightCurve(convolution, pixelsize, vtrans, time, startposition = (1000, 1000), angle = 0):
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

        xpositions = np.linspace(startposition[0], startposition[0]+xtraversed, px_traversed)
        ypositions = np.linspace(startposition[1], startposition[1]+ytraversed, px_traversed)
        light_curve = []
        for tt in range(px_traversed):
                light_curve.append(MeasureMLAmp(convolution, xpositions[tt], ypositions[tt]))
                
        return light_curve

        

def PullRandLC(convolution, pixelsize, vtrans, time, returntrack=False): #
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

        xpositions = np.linspace(startposition[0], startposition[0]+xtraversed, px_traversed)
        ypositions = np.linspace(startposition[1], startposition[1]+ytraversed, px_traversed)
        light_curve = []
        for tt in range(px_traversed):
                light_curve.append(MeasureMLAmp(convolution, xpositions[tt], ypositions[tt]))

        track = [xpositions, ypositions]


        if returntrack==True:
                return light_curve, track
        else:
                return light_curve
        



def FastLensNoFitsMismatchedRes(MagMap, file, startX, startY, endX, endY, steps, wavelength, zquasar = 2.0, mass = 10**8.0*2*10**30, ratio = False,
                                diskres = 131, geounits = 4000, fov = 0.12, 
                                masklowerx=0, maskupperx=0, masklowery=0, maskuppery=0, verbose=False, zlens = 0.46, nummapER = 1, mapres = 10000,
                                deres=False, ReturnTimeAxis=False, Rotation=False, ReturnYScannedImage=False, TestFile=False, Plot=False):
        '''
        This loads the disk image (from a .disk file) and lenses it through a magnification map.
        Make sure it is the proper size for the resolution and Einstein Radii! The header should contain information
        about the number of ER, the wavelength, and the velocity.
        Similar to my other lensing functions, this accepts the MagMap, initial, and final positions, as well as the
        number of steps to take between those points. The difference is the image is already calculated and this
        allows for rapid tests to be performed.
        MagMap may be input as either a 1-dim or 2-dim array.
        The file image expects a mapping of 10000 x 10000
        If Rotate is input as a degree value, the image of the file will be rotated counter-clockwise by that degree amount.
        If TestFile is input (as file path), this program will take the header info from "file" and apply those parameters to the "TestFile"
        to map the TestFile even though it doesn't have things like wavelength/velocity/pixsize defined already. Use with Gyoto outputs.

        This also adjusts the resolution in order to match pixels properly when mass and zshifts are included.
        Dependencies:
        Numpy
        Astropy.fits
        PyPlot
        scikit-image
'''
        import numpy as np
        import matplotlib.pyplot as plt
        from astropy.nddata import block_reduce
        from QuasarModelFunctions import AngDiameterDistance, GetGeometricUnit, CalcEinsteinRadius, AngDiameterDistanceDifference
        from astropy import units as u
        from scipy.ndimage import rotate

        if type(file) == str:
        
                with open(file+".comp", 'r') as f:
                    line1 = f.readline()
                    line2 = f.readline()
                    line3 = f.readline()

                    line = line1.split()
                    velocity = float(np.asarray(line)[1])
                    line = line2.split()
                    radius = float(np.asarray(line)[1])
                    line = line3.split()
            
        
                hdu = empty([int(diskres), int(diskres)])
                ii = 0
                with open(file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        columns = line.split()
                        hdu[:, ii] = np.asarray(columns, dtype=float)
                        ii += 1
        else:
                hdu = file


        if ReturnYScannedImage==True: scanned_image_y = array([steps+1, max(size(hdu, 0), size(hdu, 1))])

        if type(rotation) != bool:
                newimg = rotate(disk, rotation, axes=(0, 1), reshape=False)
                disk = newimg

        if MagMap.ndim == 1:
                MapXY = ConvertMagMap(MagMap)
                res = int(size(MagMap)**0.5)
        elif MagMap.ndim == 2:
                MapXY = MagMap
                res = int(size(MagMap[0, :]))
        else:
                print("invalid dimensionality of lensing map")
                return(null)

        if type(file) == str:
                if deres == True:
                        if res != 10000:
                                resratio = res/10000
                                velocity *= resratio
                                startX *= resratio
                                startY *= resratio
                                endX *= resratio
                                endY *= resratio
                                hdu = block_reduce(hdu, 1/resratio)
                                pix_size *= 1/resratio
                                intsteps = steps * resratio
                                steps = int(intsteps)
                                print("The resolution of the source has been modified as the expected map's resolution did not match.\nThe new shape is:", shape(hdu))


        if maskupperx > 0 and maskuppery > 0:
                Mask = zeros([size(MapXY, 0), size(MapXY, 1)])
                Mask[masklowerx:maskupperx, masklowery:maskuppery] = 1
                MapXY *= Mask
                print("The lensing map has been masked and updated!")

        if type(file) == str:
                pix_traveled = ( (startX - endX)**2 + (startY - endY)**2)**(0.5)           # This is how many pixels are travled by the QSO
                years = (pix_traveled / velocity)                                                          # This gives how long the travel time is in years, for the distance (startx, starty) to (endx, endy)
                x_axis = linspace(0, years , steps + 1)
              
        positions = zeros([steps + 1, 2])
        output = zeros(steps + 1)
        for i in range(steps + 1):                                                                  # These are the (x, y) positions at each time step, stored as [:, 0], [:, 1]
                positions[i, 0] = int((endX - startX) * i / steps + 0.5) + startX       
                positions[i, 1] = int((endY - startY) * i / steps + 0.5) + startY

        

        lumdist = (1 + float(zquasar))**2 * AngDiameterDistance(float(zquasar))                  # luminosity distance
        dummydistance = GetGeometricUnit(mass)                                          # This is the distance between screen and object for gyoto image
        imgpixsize = (fov) * geounits * dummydistance / diskres                               # This defines the size of each pixel on the object

        ERsize = lumdist * CalcEinsteinRadius(zlens, zquasar)
        mappixsize = (nummapER * ERsize / mapres).value                                                 # Calculates the size of each pixel on the mag map

        imgpixislarger = 1
        
        if ratio == False:
                if imgpixsize > mappixsize:
                        ratio = int(imgpixsize / mappixsize)
                        xlen = int((size(hdu[:, 0])/2.0))*ratio                         # these are radii from black hole, on the image of the caustic
                        ylen =  int((size(hdu[0, :])/2.0))*ratio
                elif imgpixsize < mappixsize:
                        ratio = int(mappixsize / imgpixsize)
                        imgpixislarger = 0
                        xlen = int((size(hdu[:, 0])/2.0))/ratio
                        ylen =  int((size(hdu[0, :])/2.0))/ratio
                else:
                        ratio = 1
                        xlen = int((size(hdu[:, 0])/2.0))*ratio
                        ylen =  int((size(hdu[0, :])/2.0))*ratio
        else:
                xlen = int((size(hdu[:, 0])//2))
                ylen = int(size(hdu[0, :])//2)
                        
        
        for step in range(steps + 1):
                lensingimage = MapXY[int(positions[step, 0] - (xlen)) : int(positions[step, 0] + xlen + 1), int(positions[step, 1] - (ylen)) : int(positions[step, 1] + ylen + 1)]
                diskimage = hdu.copy()
                
                if imgpixislarger == 1:
                        adjustedlensingimage = block_reduce(lensingimage, ratio)        # This block fixes the resolution ratios
                        lensingimage = adjustedlensingimage
                elif imgpixislarger == 0:
                        adjustedmap = block_reduce(diskimage, ratio)
                        diskimage = adjustedmap
                        
                if shape(lensingimage) != shape(diskimage):                                   # This block fixes any mismatched sizes before convolution which occur due to rounding
                        if len(lensingimage[:, 0]) > len(diskimage[:, 0]):
                                lensingimage = lensingimage[:(len(lensingimage[:, 0]) - 1), :]
                        if len(lensingimage[0, :]) > len(diskimage[0, :]):
                                lensingimage = lensingimage[:, :(len(lensingimage[0, :]) - 1)]
                        if len(diskimage[0, :]) > len(lensingimage[0, :]):
                                diskimage = diskimage[:, : (len(diskimage[0, :]) - 1)]
                        if len(diskimage[:, 0]) > len(lensingimage[:, 0]):
                                diskimage = diskimage[: (len(diskimage[:, 0]) - 1), :]

                                
                output[step] = nansum(diskimage * lensingimage) 
                if ReturnYScannedImage==True: scanned_image_y[step, :] = lensingimage[:, int(len(lensingimage)/2)]
                if verbose == True: print((step + 1) / (steps + 1) * 100, "% complete")

        dummyout = output * (2*np.pi*(1-cos(fov))/diskres**2) * 1e-6 * (dummydistance * geounits/lumdist.value)**2  # output is in W/m^3/ster, 2.63e-6 factor comes from ster/pixel. 1e-6 converts to W/m^2/um. (geounits*dummydistance/lumdist)^ is r^-2 dependence of luminosity.
        output = dummyout
        x_axis = np.linspace(0, (endY - startY)/10000, (endY-startY)+1)
        if Plot==True:
                

                brightness = output
                fig = plt.figure()
                ax = plt.axes()
                plt.plot(x_axis, brightness)
                ax.set_xlabel('Time [ER]')
                ax.set_ylabel('Brightness observed [W/m^2/μm]')
                ax.set_title('Light curve of disk over lensing map')

        if ReturnTimeAxis==False and ReturnYScannedImage==True:
                return(scanned_image_y, output)
        if ReturnTimeAxis==True and ReturnYScannedImage==True:
                return(x_axis, scanned_image_y, output)
        if ReturnTimeAxis==True and ReturnYScannedImage==False:
                return(x_axis, output)
        else:
                return(output)


def TimeDependentLensing(MagMap, file, startX, startY, endX, endY, steps, wavelength, velocity, zquasar = 2.0, mass = 10**8.0*2*10**30, ratio = False,
                                diskres = 131, geounits = 4000, fov = 0.12, sim5=False, diskGRs = 100, 
                                masklowerx=0, maskupperx=0, masklowery=0, maskuppery=0, verbose=False, zlens = 0.46, nummapER = 25, mapres = 10000,
                                deres=False, ReturnTimeAxis=False, Rotate=False, ReturnYScannedImage=False, TestFile=False, Plot=False):
        '''
        This loads the disk image (from a .fits file) and lenses it through a magnification map.
        Make sure it is the proper size for the resolution and Einstein Radii! 
        Similar to my other lensing functions, this accepts the MagMap, initial, and final positions, as well as the
        number of steps to take between those points. The difference is the image is already calculated and this
        allows for rapid tests to be performed. It also takes a time-varying image (EG: for reverberation).
        MagMap may be input as either a 1-dim or 2-dim array.
        The file image expects a mapping of 10000 x 10000
        velocity is expected in units km/s, and is transverse velocity
        If Rotate is input as a degree value, the image of the file will be rotated counter-clockwise by that degree amount.
        If TestFile is input (as file path), this program will take the header info from "file" and apply those parameters to the "TestFile"
        to map the TestFile even though it doesn't have things like wavelength/velocity/pixsize defined already. Use with Gyoto outputs.

        This also adjusts the resolution in order to match pixels properly when mass and zshifts are included.
        Dependencies:
        Numpy
        Astropy.fits
        PyPlot
        scikit-image
'''
        import numpy as np
        import matplotlib.pyplot as plt
        from astropy.nddata import block_reduce
        from QuasarModelFunctions import AngDiameterDistance, GetGeometricUnit, CalcEinsteinRadius, AngDiameterDistanceDifference
        from astropy import units as u

        if type(file) == str:

                hdu = empty([int(diskres), int(diskres), int(steps)])
                dummyhdu = empty([int(diskres), int(diskres)])
                ii = 0
                with open(file, 'r') as f:
                        for line in f:
                                line = line.strip()
                                columns = line.split()
                                dummyhdu[:, ii] = np.asarray(columns, dtype=float)
                                ii += 1

                for jj in range(steps):
                        hdu[:, :, jj] = dummyhdu

                        
        else:
                hdu = file
        if type(velocity) == u.Quantity:
                newvel = velocity.to(u.m/u.d)
        else:
                newvel = velocity * 1000 * 60 * 60 * 24  #Convert manually to m/day. intrinsic variation calculated in light days.

        if ReturnYScannedImage==True: scanned_image_y = array([steps+1, max(size(hdu, 0), size(hdu, 1))])

        if Rotate != False:
                
                length = int(max(size(hdu, axis=0), size(hdu, axis=1)))
                new_image = zeros([length, length, size(hdu, axis=2)])

                for x in range(length):
                        for y in range(length):
                                i = x - length/2.0
                                j = y - length/2.0
                                rad, phi = QMF.ConvertToPolar(i, j)

                                phi += (-1) * Rotate * pi / 180.0
                                
                                i = int( rad * np.cos(phi) + size(hdu, axis=0) / 2.0 + 0.5 )
                                j = int( rad * np.sin(phi) + size(hdu, axis=1) / 2.0 + 0.5 )

                                if i < size(hdu, axis=0) and j < size(hdu, axis=1) and i > 0 and j > 0: new_image[x, y, :] = hdu[i, j, :]

                hdu = new_image
                                

        if MagMap.ndim == 1:
                MapXY = ConvertMagMap(MagMap)
                res = int(size(MagMap)**0.5)
        elif MagMap.ndim == 2:
                MapXY = MagMap
                res = int(size(MagMap[0, :]))
        else:
                print("invalid dimensionality of lensing map")
                return(null)

        if type(file) == str:
                if deres == True:
                        if res != 10000:
                                resratio = res/10000
                                startX *= resratio
                                startY *= resratio
                                endX *= resratio
                                endY *= resratio
                                hdu = block_reduce(hdu, 1/resratio)
                                pix_size *= 1/resratio
                                intsteps = steps * resratio
                                steps = int(intsteps)
                                print("The resolution of the source has been modified as the expected map's resolution did not match.\nThe new shape is:", shape(hdu))



        if maskupperx > 0 and maskuppery > 0:
                Mask = zeros([size(MapXY, 0), size(MapXY, 1)])
                Mask[masklowerx:maskupperx, masklowery:maskuppery] = 1
                MapXY *= Mask
                print("The lensing map has been masked and updated!")

              
        positions = zeros([steps + 1, 2])
        output = zeros(steps + 1)
        for i in range(steps + 1):                                                                  # These are the (x, y) positions at each time step, stored as [:, 0], [:, 1]
                positions[i, 0] = int((endX - startX) * i / steps + 0.5) + startX       
                positions[i, 1] = int((endY - startY) * i / steps + 0.5) + startY

        

        lumdist = (1 + float(zquasar))**2 * AngDiameterDistance(float(zquasar))                  # luminosity distance
        dummydistance = GetGeometricUnit(mass)                                          # This is the distance between screen and object for gyoto image
        if sim5==True:
                imgpixsize = diskGRs * GetGeometricUnit(mass) / diskres
        else:
                imgpixsize = (fov) * geounits * dummydistance / diskres                               # This defines the size of each pixel on the object

        ERsize = lumdist * CalcEinsteinRadius(zlens, zquasar)
        mappixsize = (nummapER * ERsize / mapres).value                                                 # Calculates the size of each pixel on the mag map
        daysperpix = int(mappixsize / newvel + 0.5)  # Each pixel movement in mag map moves this many frames ahead in the disk image
        if steps*daysperpix > np.size(hdu, -1):
                print("Intrinsic variability curve not long enough, cycling through data given")
                print("To avoid this with this configuration, use:", steps * daysperpix, "time units")
        imgpixislarger = 1
        
        if ratio == False:
                if imgpixsize > mappixsize:
                        ratio = int(imgpixsize / mappixsize)
                        xlen = int((size(hdu[:, 0, 0])/2.0))*ratio                         # these are radii from black hole, on the image of the caustic
                        ylen =  int((size(hdu[0, :, 0])/2.0))*ratio
                elif imgpixsize < mappixsize:
                        ratio = int(mappixsize / imgpixsize)
                        imgpixislarger = 0
                        xlen = int((size(hdu[:, 0, 0])/2.0))/ratio
                        ylen =  int((size(hdu[0, :, 0])/2.0))/ratio
                else:
                        ratio = 1
                        xlen = int((size(hdu[:, 0, 0])/2.0))*ratio
                        ylen =  int((size(hdu[0, :, 0])/2.0))*ratio
        else:
                xlen = int((size(hdu[:, 0, 0])//2))
                ylen = int(size(hdu[0, :, 0])//2)
                        
        for step in range(steps):
                lensingimage = MapXY[int(positions[step, 0] - (xlen)) : int(positions[step, 0] + xlen + 1), int(positions[step, 1] - (ylen)) : int(positions[step, 1] + ylen + 1)]
                diskimage = hdu.copy()
                
                
                if imgpixislarger == 1:
                        adjustedlensingimage = block_reduce(lensingimage, ratio)        # This block fixes the resolution ratios
                        lensingimage = adjustedlensingimage
                        dummy = diskimage[:, :, step*daysperpix % np.size(hdu, -1)]
                        diskimage = dummy
                elif imgpixislarger == 0:
                        adjustedmap = block_reduce(diskimage[:, :, step*daysperpix % np.size(hdu, -1)], ratio)
                        diskimage = adjustedmap
                        
                if shape(lensingimage) != shape(diskimage):                                   # This block fixes any mismatched sizes before convolution which occur due to rounding
                        boxlength = (min(np.size(lensingimage, 0), np.size(diskimage, 0)))
                
                        lensingimage = lensingimage[:boxlength, :boxlength]
                        diskimage = diskimage[:boxlength, :boxlength]
    
                output[step] = nansum(diskimage * lensingimage) 
                if ReturnYScannedImage==True: scanned_image_y[step, :] = lensingimage[:, int(len(lensingimage)/2)]
                if verbose == True: print((step + 1) / (steps) * 100, "% complete")

        dummyout = output * (2*np.pi*(1-cos(fov))/diskres**2) * 1e-6 * (dummydistance * geounits/lumdist.value)**2  # output is in W/m^3/ster, 2.63e-6 factor comes from ster/pixel. 1e-6 converts to W/m^2/um. (geounits*dummydistance/lumdist)^ is r^-2 dependence of luminosity.
        output = dummyout
        x_axis = np.linspace(0, (endY - startY)/10000, (endY-startY)+1)
        if Plot==True:
                

                brightness = output
                fig = plt.figure()
                ax = plt.axes()
                plt.plot(x_axis, brightness)
                ax.set_xlabel('Time [ER]')
                ax.set_ylabel('Brightness observed [W/m^2/μm]')
                ax.set_title('Light curve of disk over lensing map')

        if ReturnTimeAxis==False and ReturnYScannedImage==True:
                return(scanned_image_y, output)
        if ReturnTimeAxis==True and ReturnYScannedImage==True:
                return(x_axis, scanned_image_y, output)
        if ReturnTimeAxis==True and ReturnYScannedImage==False:
                return(x_axis, output)
        else:
                return(output)


def SaveCurveNoFits(time, brightness, ext, input_file, wavelength):
        '''
        This funciton saves a light curve created in a .fits format. The header will contain information about how
        many high magnification events are detected, how long the curve lasts, and information regarding the
        brightness model input.
        I will define high magnification event as a 50% increase of magnification within 300 time units.
        Using 100% incresae of magnification within 100 time units did not capture all the observed crossing events.
        Using 50% increase of magnification within 100 time units captured 2237's events, but not 0435's events.
        Using 50% increase within 1% of the time units

        time is the time axis of the light curve
        brightness is the light curve
        ext is the extention to be given to the file name
        input_file is the brightness map .fits file, used to extract some header information
'''
        import numpy as np
        from astropy.io import fits

        with open(input_file+'.comp', 'r') as f:
            line1 = f.readline()
            line2 = f.readline()

            line = line1.split()
            velocity = float(np.asarray(line)[1])
            line = line2.split()
            radius = (np.asarray(line)[1])
        
        max_time = max(time)                    # This simply stores the duration of the light curve

        data = array([time, brightness])
        
        hdu2 = fits.PrimaryHDU(data)
        hdu2.header['wavelen'] = wavelength         # In nm
        hdu2.header['velocity'] = velocity                # In pixels/year
        hdu2.header['duration'] = max_time              # In years
        hdu2.writeto('LightCurve'+str(int(max_time))+'years'+str(wavelength)+'nm'+str(ext)+'.fits')

def SaveManyCurveNoFits(time, brightness, input_file, wavelength, mass, viewang, impactang, eddingtons, redshift, orignote = True, note=''):
        '''
        This funciton saves a light curve created in a .fits format. The header will contain information about how
        many high magnification events are detected, how long the curve lasts, and information regarding the
        brightness model input.

        time is the time axis of the light curve
        brightness is the light curve
        ext is the extention to be given to the file name
        input_file is the brightness map .fits file, used to extract some header information
'''
        import numpy as np
        from astropy.io import fits

        if orignote == True:
                with open(input_file+str(mass)+str(eddingtons)+str(viewang)+str(note)+'.disk.comp', 'r') as f:
                    line1 = f.readline()
                    line2 = f.readline()

                    line = line1.split()
                    velocity = float(np.asarray(line)[1])
                    line = line2.split()
                    radius = (np.asarray(line)[1])
        else:
                with open(input_file+str(mass)+str(eddingtons)+str(viewang)+str(orignote)+'.disk.comp', 'r') as f:
                    line1 = f.readline()
                    line2 = f.readline()

                    line = line1.split()
                    velocity = float(np.asarray(line)[1])
                    line = line2.split()
                    radius = (np.asarray(line)[1])        
        
        max_time = max(time)                    # This simply stores the duration of the light curve

        data = array([time, brightness])
        
        hdu2 = fits.PrimaryHDU(data)
        hdu2.header['wavelen'] = wavelength         # In nm
        hdu2.header['velocity'] = velocity                # In pixels/year
        hdu2.header['duration'] = max_time              # In years
        hdu2.writeto('/Users/henrybest/PythonStuff/Data/RawLightCurves/RAW'+str(mass)+'M'+str(viewang)+'VA'+str(impactang)+'IA'+str(eddingtons)+'ER'+str(redshift)+'z'+str(note)+'.lightcurve.fits', overwrite=True)

def CreateFlatCaustic(XYMap, distance, XSlice=False):
        '''
        This function takes in an XY representation of a caustic map (such as those created by GERLUMPH and
        converted to x-y coordinates, and then selects a vertical (or horizontal) slice from this map. This slice is then
        propagated to fill the entire size of the original XYMap, and outputs a new caustic map of same size.
        By default, this selects a slice of the map along the y-direction. However, setting XSclice=True will output
        a map created with a slice along the x-direction.
        Distance is a pixel number along the map, and must be less than the resolution of the input map.
        '''
        assert type(distance) == (int)
        if XSlice==False:
                assert distance < size(XYMap, 1)
        else:
                assert distance < size(XYMap, 0)

        output = zeros([size(XYMap, 0), size(XYMap, 1)])

        if XSlice==False:
                for ii in range(size(XYMap, 1)):
                        output[:, ii] = XYMap[distance, ii]
        else:
                for ii in range(size(XYMap, 0)):
                        output[ii, :] = XYMap[ii, distance]
        return(output)

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

def CreateWindLine(launchrad, launchangle, maxzheight, zslices, characteristicdistance, centralBHmassexp = 8.0, launchheight = 0, maxvel = 10**6, launchspeed = 0, alpha=1):
        '''
        This creates a simple line of wind, divided up as vertical slabs, assuming conservation of ang
        momentum. This wind line will use Suk Yee's model for poloidal velocity, using a definable alpha parameter.
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
        
        phi_init_velocity = QMF.KepVelocity(launchrad.to(u.m).value, 10**centralBHmassexp)
        
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
                output_streamline[3, ii] = QMF.KepVelocity(output_streamline[0, ii], 10**centralBHmassexp).value 
                output_streamline[5, ii] = v_pol

                assert(output_streamline[2, ii] < 3e8)
                assert(output_streamline[3, ii] < 3e8)
                assert(output_streamline[4, ii] < 3e8)
                assert(output_streamline[5, ii] < 3e8)
        return(output_streamline)


def CreateWindRegion(sl1, sl2, r_res = 100, z_res = 100, phi_res = 30, centralBHmassexp=8.0, r0 = 10e15, sigma = 10e7, function=1, power=1):
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

        inputvalues = [10**centralBHmass, r0, sigma]   #Standardize units and keep values
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


def ProjectWind(windgrid, rlen, zlen, philen, rmin, viewingang, xres, SL1, SL2, velocities, geounits = 4000, mass = 10e8 * const.M_sun.to(u.kg), reverberating=False): #
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
                screen = np.zeros([boxres, boxres])

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

                        if zarg >= (len(heights)-2):
                                continue
                        if rarg >= (len(radii) - 2):
                                continue
                        if zarg < np.size(windgrid, 1) and rarg < np.size(windgrid, 0) and phiarg < np.size(windgrid, 2):

                                if reverberating == True:
                                        
                                        screen [ii, jj, :, :] += windgrid[rarg, zarg, phiarg, :, :]
                                else:
                                        
                                        screen[ii, jj] += windgrid[rarg, zarg, phiarg, 0]
        
        return screen, fov



def ReverberateWind(windgrid, illumination, inc_angle, r_length, z_length, phi_length, continuumpowerslope = 1.0, timescale = 60*60*24, reflectfactor = 1,
                    velocityres = 21, resetreverb=False):
        '''
        This function aims to propogate some illuminating signal through the wind, assuming the wind has no frequency dependence.
        The illumination should be some brightness array used as the input signal.
        This does not do any radiative transfer calculations, it simply determines the time delay to the source, and reflects it.
        inc_angle is the angle of inclination [deg].
        r_length and z_length are the ammount of distance each pixel travels in one time unit
        phi_res is the number of radians in a resolution unit.
        velocityres is how many bins to split up the line of sight velocity into
        Hopefully, the echo will be captured in the resulting output
        timescale is a parameter which determines how much 1 'timedelay' shifts the reverberation along 1 'illumination unit'
        '''
        import numpy as np
        from QuasarModelFunctions import ConvertToPolar, AddVels


        inc_angle *= np.pi/180
        timescale *= 3e8

        timedelay = np.zeros([np.size(windgrid, 0), np.size(windgrid, 1), np.size(windgrid,2)])
        maxvel = max(np.max(windgrid[:, :, :, 2])*np.sin(inc_angle), np.max(windgrid[:, :, :, 1])*np.sin(inc_angle), np.max(windgrid[:, :, :, 3])*np.cos(inc_angle))
        velocities = np.linspace(-maxvel/3e8, maxvel/3e8, velocityres)

        outputreverb = illumination.copy()
        wind = np.zeros([np.size(windgrid, 0), np.size(windgrid, 1), np.size(windgrid, 2), np.size(illumination), velocityres])

        if resetreverb == True:
                outputreverb[:] = 0

        for ii in range(np.size(timedelay, 0)):
                rp = ii * r_length
                for kk in range(np.size(timedelay, 2)):
                        phi = kk * phi_length
                        for jj in range(np.size(timedelay, 1)):
                                z = jj * z_length
                                _, theta = ConvertToPolar(z, rp)
                                
                                timedelay[ii, jj, kk] = rp * (1 - np.cos(theta*np.sin(phi)+inc_angle))

                                if windgrid[ii, jj, kk, 0] != 0:
                                        vpol = ((windgrid[ii, jj, kk, 1]**2 + windgrid[ii, jj, kk, 3]**2)**0.5)
                                        beta = vpol / 3e8
                                else:
                                        beta = 0
                                        continue
                                if beta >= 1:
                                        print(beta)
                                        print(vpol)
                                        print(windgrid[ii, jj, kk, 1], windgrid[ii, jj, kk, 3])
                                        continue
                                assert(beta<=1)
                                
                                delay = timedelay[ii, jj, kk]      
                                delayindex = int(delay//timescale)
                                betaadjustment = (( (1 + beta) / (1 - beta))**0.5 - 1) ** continuumpowerslope
                
                                if windgrid[ii, jj, kk, 0] != 0:
                                        for tt in range(len(illumination)):
                                              
                                                betatowards = AddVels(windgrid[ii, jj, kk, 1]*np.sin(inc_angle)*np.sin(kk*phi_length)*(-1),
                                                                          windgrid[ii, jj, kk, 2]*np.sin(inc_angle)*np.cos(kk*phi_length)*(-1),
                                                                          windgrid[ii, jj, kk, 3]*np.cos(inc_angle))
                                                        
                                                varg = np.argmin(abs(velocities - betatowards))
                                                addedoutput = (1 + betaadjustment) * illumination[int(tt - delayindex)%len(illumination)] * reflectfactor 
                                                        
                                                outputreverb[tt] += addedoutput
                                                wind[ii, jj, kk, tt, varg] += addedoutput

        return(outputreverb, wind, velocities)


def CreateBLRTransferFunction(BLR, rlen, zlen, philen, inc_ang, xres, SL1, SL2, mass = 10e8 * const.M_sun.to(u.kg), geounits = 4000, units = 'days', return_grid = False):#

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


        maxdelay = 2*((rlen * np.size(BLR, 0))**2 + (zlen * np.size(BLR, 1))**2)**0.5 / steptimescale

        tfgrid = np.zeros([xres, yres, int(maxdelay + 1)])

        radii = np.linspace(0, SL2[0, -1], int(SL2[0, -1] / rlen)) #Useful to calculate closest arguments
        angles = np.linspace(0, 2*np.pi, int(2*np.pi / philen))
        heights = np.linspace(0, SL2[1, -1], int(SL2[1, -1] / zlen))

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

                        if zarg < np.size(BLR, 1) and rarg < np.size(BLR, 0) and phiarg < np.size(BLR, 2):
                                
                                _, theta = QMF.ConvertToPolar(zwind, rp)
                                delay = rp * (1 - np.cos(theta*np.sin(phi)+viewingang)) / steptimescale
                                density = BLR[rarg, zarg, phiarg, 0]
                                if int(delay) < (np.size(tfgrid, 2)):
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
        '''
        import QuasarModelFunctions as QMF
        import numpy as np

        if np.ndim(illumination) == 1:
                dummywaves = np.ones([len(wavelengths), len(illumination)]) # If not given explicity, assume all wavelengths in continuum are equal.
                for ii in range(len(illumination)):
                        dummywaves[:, ii] *= illumination[ii]
                illumination = dummywaves

        inc_angle *= np.pi/180 #Convert to rads
        timescale *= 3e8 #Units light-something


        assert(np.size(wavelengths) == np.size(illumination, 0)) #Make sure there's a one-to-one mapping of the wavelength labels to the indexes

        outputreverb = illumination.copy()
        timedelay = np.zeros([np.size(windgrid, 0), np.size(windgrid, 1), np.size(windgrid,2)])

        outputreverb[:, :] = 0

        dradial = (r_length**2 + z_length**2)**0.5
        
        if returnpulse == True: #For a pulse map to check timescales
                pulsemap = np.zeros([np.size(windgrid, 0), np.size(windgrid, 1), np.size(illumination, 1)])
                
        if returnvelocitygrid == True: #For LoS velocities
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

        


def CreateTimeDelayMap(disk, incangle, massquasar = 10**8 * const.M_sun.to(u.kg), diskres = 300, fov = 0.12, geounits = 4000,
                       nGRs = 100, coronaheight = 5, axisoffset=0, angleoffset=0, sim5 = False, unit='hours'): #
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
        import numpy as np
        import QuasarModelFunctions as QMF
        from astropy.io import fits
        import astropy

        incangle *= np.pi/180
        angleoffset *= np.pi/180
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
                steptimescale = 3e8 / u.s.to(unit)
        else:
                print('Invalid unit deteted. Try "days", "hours", "minutes", "seconds" or an astropy.unit.\nReverting to hours.')
                steptimescale = 3e8*60*60
        if sim5 == True:
                rstep = nGRs * QMF.GetGeometricUnit(massquasar) / diskres
        else:
                rstep = fov * geounits * QMF.GetGeometricUnit(massquasar) / diskres

        gr = QMF.GetGeometricUnit(massquasar)
        #offsets are position of X-ray source
        xoffset = axisoffset * gr * np.sin(angleoffset)
        yoffset = - axisoffset * gr * np.cos(angleoffset) * np.cos(incangle) + coronaheight * gr * np.sin(incangle)
        zoffset = - coronaheight * gr * np.cos(incangle) - axisoffset * gr * np.cos(angleoffset) * np.sin(incangle)
        indexes = disk >= 0.0001
        output = np.ndarray(np.shape(disk.copy()))
        for xx in range(diskres):
            for yy in range(diskres):
                if indexes[xx, yy] != 0:
                        if sim5 == False:
                                xxx = diskres/2 - xx  #convert [xx, yy] gridpoints to proper axis
                                yyy = diskres/2 - yy
                                x1 = rstep * xxx
                                y1 = rstep * yyy 
                                z1 = yyy * rstep * np.sin(incangle) / np.cos(incangle)

                                r1 = ((x1 - xoffset)**2 + (y1 - yoffset)**2)**0.5
                                z = zoffset - z1

                                output[xx, yy] = (abs(z - (z**2 + r1**2)**0.5)/steptimescale + 0.5)
                        else:
                                xxx = diskres/2 - xx 
                                yyy = diskres/2 - yy 
                                x1 = rstep * xxx
                                y1 = rstep * yyy # *cos(incangle)/cos(incangle)
                                z1 = yyy * rstep * np.sin(incangle) / np.cos(incangle)

                                r1 = ((x1 - xoffset)**2 + (y1 - yoffset)**2)**0.5
                                z = zoffset - z1

                                output[yy, xx] = (abs(z - (z**2 + r1**2)**0.5)/steptimescale + 0.5)                                
        return output

def ConstructGeometricDiskFactor(disk, inc_ang, massquasar, height, r=0, phi=0, nGRs=100, albedo=0, geounits = 4000, fov=0.12, sim5=True):#
        '''
        This function creates the geometric factor of the accretion disk:
        (1-A)cos(theta_x)/(4*pi*sigma*R_{*}^{2})  #Additional term which multiplies Lx in Eq. 2 of Cackett+ 2007
        
        
        '''
        import numpy as np
        import QuasarModelFunctions as QMF
        import astropy.units as u
        import astropy.constants as const
        if disk.ndim == 2:
                output = np.zeros(np.shape(disk))
        elif disk.ndim == 3:
                output = np.zeros(np.shape(disk[:,:,-1]))
        assert albedo <= 1 and albedo >= 0
        inc_ang *= np.pi/180
        gr = QMF.GetGeometricUnit(massquasar)
        if sim5 == True:
                rstep = nGRs * gr / np.size(disk, 0)
        else:
                rstep = fov * geounits * gr / np.size(disk, 0)
        axisoffset = r
        angleoffset = phi * np.pi/180
        sig = const.sigma_sb.value
        
        xoffset = axisoffset * gr * np.sin(angleoffset) 
        yoffset = - axisoffset * gr * np.cos(angleoffset) * np.cos(inc_ang) + height * gr * np.sin(inc_ang) 
        
        for xx in range(np.size(disk, 0)):
                for yy in range(np.size(disk, 1)):
                        if sim5 == False:
                                yyy = rstep*(np.size(disk, 1)/2 - yy)/np.cos(inc_ang)
                                xxx = rstep*(np.size(disk, 0)/2 - xx)

                                distance2 = (xxx-xoffset)**2 + (yyy-yoffset)**2 + (height * gr)**2
                                costheta = (height*gr)/(distance2**0.5)
                                output[xx, yy] = (1-albedo) * costheta / (4 * np.pi * sig * distance2)
                        else:
                                xxx = rstep*(np.size(disk, 1)/2 - xx)
                                yyy = rstep*(np.size(disk, 0)/2 - yy)/np.cos(inc_ang)

                                distance2 = (xxx-xoffset)**2 + (yyy-yoffset)**2 + (height * gr)**2
                                costheta = (height*gr)/(distance2**0.5)
                                output[yy, xx] = (1-albedo) * costheta / (4 * np.pi * sig * distance2)
                                
                        
        return output
                                



def ConstructDiskTransferFunction(image_der_f, inc_ang, massquasar, height, maxlength=1000, units=u.h, r=0, phi=0, nGRs =100, sim5=True, weight=False,
                                  albedo=0, geounits=4000, fov=0.12): #
        '''
        This takes in disk objects and parameters in order to construct its transfer function
        The disk should be the derivative of the Planck function w.r.t Temperature
        height is the position of the point source
        r, phi are for off-axis point sources
        all distances are input in R_g
        '''
        from scipy.signal import savgol_filter
        import numpy as np
        import QuasarModelFunctions as QMF
        import astropy.units as u
        import astropy.constants as const

        
        if image_der_f.ndim == 2:
                output = np.zeros((maxlength))
                diskdelays = QMF.CreateTimeDelayMap(image_der_f, inc_ang, massquasar = massquasar, coronaheight = height, unit = units,
                                                    axisoffset = r, angleoffset = phi, nGRs = nGRs, sim5=sim5)
        elif image_der_f.ndim == 3:
                output = np.zeros((maxlength, np.size(image_der_f, 2)))
                diskdelays = QMF.CreateTimeDelayMap(image_der_f[:,:,-1], inc_ang, massquasar = massquasar, coronaheight = height, unit = units,
                                                    axisoffset = r, angleoffset = phi, nGRs = nGRs, sim5=sim5)
        else:
                print("Invalid dimensionality of input disk")
                return null
        if weight==True:
                weight = QMF.ConstructGeometricDiskFactor(image_der_f, inc_ang, massquasar, height, r=r, phi=phi, nGRs=nGRs, albedo=albedo,
                                                          geounits = geounits, fov=fov, sim5=sim5)
        else:
                weight = np.ones((np.size(image_der_f, 0), np.size(image_der_f, 1)))

        for jj in range(maxlength):
                maskin = (diskdelays >= jj - 1)
                maskout = (diskdelays <= jj + 1)
                mask = maskin*maskout
                if image_der_f.ndim == 2:
                        output[jj] = np.sum((mask * image_der_f[:,:] * weight), (0, 1))
                else:
                        for band in range(np.size(image_der_f, 2)):
                                print(np.shape(mask), np.shape(image_der_f[:,:,band]), np.shape(weight))
                                output[jj, band] = np.sum((mask * image_der_f[:,:,band] * weight), (0,1))
        if image_der_f.ndim==2:
                smoothedoutput = savgol_filter(output, 7, 3)
                smoothedoutput *= smoothedoutput>0
                smoothedoutput /= np.sum(smoothedoutput)
        else:
                smoothedoutput = np.zeros(np.shape(output))
                for band in range(np.size(image_der_f, 2)):
                        smoothedoutput[:,band] = savgol_filter(output[:,band], 7, 3)
                smoothedoutput *= smoothedoutput>0
                for band in range(np.size(image_der_f, 2)):
                        smoothedoutput[:,band] /= np.sum(smoothedoutput[:,band])
        return smoothedoutput
        
        
        
                                        
                
        
        
                                  

        



def CreateReverbSnapshots(delaymap, time, illumination = False, massquasar = 10**8 * const.M_sun.to(u.kg), diskres = 300, fov = 0.12, geounits = 4000,
                         dampingfactor = 0.1, steppingfactor = 0.01, DRWseed=False):
        '''
        This uses a delay map created above and applies some damped random walk to be sampled with the delay mapping.
        It assumes time reversal of the damped random walk. For time series, please provide illumination so it's consistent!
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
        

def SetupDRW(t_max, delta_t, SF_inf, tau):
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

        



def InsertFlare(Disk, amplitude, theta_inc, decaytime, initialpoint=False, returnmovie=False, returnlightcurve=False, verbose=False):
        '''
        This aims to create a multiplicative field which decays in time in order to model a
        flaring effect as it propagates across the accretion disk. For some initial amplification,
        and at some initial location if given, it will spread out across the 'flat' disk radially
        with time. The amplitude will decrease as 1/r^2, since this is modeling a spherically
        expanding shell. After being magnified, the multiplicative factor will decrease exponentially
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
        flarefield = np.ones([diskxsize, diskysize, 2*diskxsize])  #Final dimension is unitless time, and should be long enough to show
        radiusfield = np.zeros([diskxsize, diskysize])          #the effect even near the edges as time progresses.
        assert(theta_inc < 90)

        flarefieldydim = 1/np.cos(theta_inc*np.pi/180)  #Distance scaling for the pixels which travel in the compressed direction

        if initialpoint == False:
                initialpoint = [diskxsize//2, diskysize//2] #Int division so we can choose a pixel index

        for xx in range(diskxsize):
                for yy in range(diskysize): #Create a field of values to refer to for radii
                        radiusfield[xx, yy] = ((xx - initialpoint[0])**2 + ((yy - initialpoint[1])*flarefieldydim)**2)**0.5
                        
        decaymask = np.zeros([diskxsize, diskysize])  # Define so the zero loop doesn't break
        for ii in range(np.size(flarefield, 2)):
                mask = (radiusfield <= ii) # Create a mask which shows the actively flaring parts
                if ii > 0:
                        decaymask = (radiusfield <= ii-1) # This mask shows the decaying parts post-flare
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

        
        
        






        
                
























                

        

