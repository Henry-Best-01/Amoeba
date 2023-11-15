'''
This script it set up to ray trace geodesics around a central supermassive black
hole to the plane of the accretion disk.
The output will be a fits file which may be used with the AmoebaExamples notebook
'''
import numpy as np
import sys
sys.path.append(path_to_sim5)   
sys.path.append(path_to_Amoeba/QuasarModelFunctions)
import QuasarModelFunctions as QMF                  
import sim5                                         # Ray tracing code
from astropy.io import fits                         # Used for storing files

# Define inputs for ray tracing

resolution = 500   # number of pixels for the output
numGRs = 250        # number of Gravitational Radii we zoom in on, resolved to ~(resolution/2)/numGRs 
inc_ang = 45        # inclination angle, in degrees
spin = 0.7            # dimensionless spin paramter, ranging from -0.99 to 0.99. 0 is the Scwarzschild case.
m_exp = 8.0
redshift = 2.0
coronaheight = 10
eddingtons = 0.15
savefile = "RayTrace.fits"

# Prepare file
_, _, _, _, _, _, out_temp, out_vel, out_g, out_r = QMF.CreateMaps(m_exp, redshift, numGRs, inc_ang, resolution, spin=spin, eddingtons=eddingtons)


# Code for manually calculating the positions where geodesics cross the midplane
#
#for xx in range(resolution):
#    for yy in range(resolution):
#        alpha = ((xx + 0.5) / resolution - 0.5) * 2.0 * numGRs
#        beta = ((yy + 0.5) / resolution - 0.5) * 2.0 * numGRs
#        gd = sim5.geodesic()
#        error = sim5.intp()
#        sim5.geodesic_init_inf(inc_ang * np.pi / 180, abs(spin), alpha, beta, gd, error)
#        if error.value(): continue
#        P = sim5.geodesic_find_midplane_crossing(gd, 0)     # 0 corresponds to Primary image
#        if np.isnan(P): continue
#        r = sim5.geodesic_position_rad(gd, P)
#        pol = sim5.geodesic_position_pol(gd, P)
#        if r >= SpinToISCO(spin):
#            azi = sim5.geodesic_position_azm(gd, r, pol, P)
#            out_radii[yy, xx] = r
#            out_azimuths[yy, xx] = azi
#            out_energy_shifts[yy, xx] = sim5.gfactorK(r, abs(spin), gd.l)


# Store data

HDU = fits.PrimaryHDU(out_temp)            # Only can have one Primary, rest must be image HDUs
HDU1 = fits.ImageHDU(out_vel)
HDU2 = fits.ImageHDU(out_g)
HDU3 = fits.ImageHDU(out_r)

HDU.header["Res"] = resolution      # Store information about the ray tracing for future reference
HDU.header["numGRs"] = numGRs
HDU.header["inc_ang"] = inc_ang
HDU.header["Spin"] = spin
HDU.header["mass"] = m_exp
HDU.header["zq"] = redshift
HDU.header["c_height"] = coronaheight
HDU.header["beta"] = 0

HDUL = fits.HDUList([HDU, HDU1, HDU2, HDU3])
HDUL.writeto(savefile, overwrite=True)


'''
# Here is a short blurb of code for opening the data into usable arrays
import numpy as np
from astropy.io import fits

# Open data

with fits.open("path_to_file") as f:
    header = f[0].header
    output_temps = f[0].data
    output_vels = f[1].data
    output_g = f[2].data
    output_r = f[3].data

# These outputs loaded in memory are now numpy arrays again
# The header may be used to access information from the ray tracing as such (case insensitive)


'''






























