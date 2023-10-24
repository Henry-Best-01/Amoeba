'''
This code takes in a dictionary of parameters from a .json file and outputs an
accretion disk movie.
Relies on some precompiled ray traces which are read depending on input .json
'''
import QuasarModelFunctions as QMF
import Amoeba
import numpy as np
import json
import astropy.units as u
import astropy.constants as const
from astropy.io import fits
from skimage.transform import rescale
import time
ts = time.time()


# input directory containing ray-traces
ray_trace_dir = "../../DiskRayTraces/"
fname = "json_inputs.json"
output_fname = "MyMovie.fits"

# Load inputs from json file
with open(fname) as file:
    inputs = json.load(file) 

mass_exp = inputs['mass']                           # int / float
redshift_q = inputs['zq']                           # int / float
inclination_angle = inputs['theta']                 # int / float, rounds to nearest 10 deg
spin = inputs['spin']                               # int / float, rounds to -0.99, -0.49, 0, 0.49, 0.99
lamppost_height = inputs['lamp_height']             # int / float, in R_g
asymp_slope = inputs['slope']                       # int / float, 0.75 is thin-disk
lamp_strength = inputs['eta_x']                     # int / float
edd_ratio = inputs['eddingtons']                    # float ideally significantly less than 1
desired_res = inputs['resolution']                  # int, number of pixels per edge on disk image
movie_total_length = inputs['total_time']           # int / float, years
variability_weight = inputs['signal_weighting']     # int / float
snapshots = inputs['snapshots']                     # int / list, if int will treat as number of evenly spaced snapshots.
                                                    #           if list, will treat as observation timestamps in hours
passband = inputs['passband']                       # array (2, n) of type (float, float). Column 1 is wavelength (nm), column 2 is transparency (0-1)  
signal = inputs['signal']                           # None or list (hourly timesteps). If None, a broken power law signal will be constructed with the following parameters
mean_signal_mag = inputs['mean_signal_mag']         # float, Mean signal magnitude
std_signal_mag = inputs['std_signal_mag']           # float, Standard deviation of signal
log_nu_b = inputs['log_nu_b']                       # float
alpha_L = inputs['alpha_L']                         # int / float
alpha_H_minus_L = inputs['alpha_H_minus_L']         # int / float
seed = inputs['seed']                               # int, random seed



# Determine which ray-trace file to read by rounding to known values
if spin < 0:
    spin_label = 'm'
    spin = abs(float(spin))
else:
    spin_label = 'p'
spin *= 2
spin += 0.5
spin = spin//1
spin /= 2
if abs(spin) > 0:
    spin -= 0.01
spin_label += str(abs(spin))[:4]
inclination_angle += 5
inclination_angle = float(inclination_angle // 10)
if inclination_angle >= 9: inclination_angle = 8
inclination_angle *= 10
inc_label = str(inclination_angle)[:3]

ray_trace_fname = ray_trace_dir+spin_label+"Spin"+inc_label+"Inc.fits"
print(ray_trace_fname)


# Load ray traced map
with fits.open(ray_trace_fname) as f:
    header = f[0].header
    d1 = f[0].data.T
    d2 = f[1].data.T
    d3 = f[2].data.T
    d4 = f[3].data.T

# Define some inportant internal values
mass = 10**mass_exp * const.M_sun.to(u.kg) 
gravrad = QMF.CalcRg(mass)
R_in = QMF.SpinToISCO(spin)
if type(snapshots) == int:
    snaps = []
    for jj in range(snapshots):
        snaps.append(int(jj * movie_total_length * 365 * 24 / snapshots))
    snapshots = snaps
initial_wavelength = passband[0][0]
orig_res = np.size(d1, 0)
scale_ratio = desired_res / orig_res
if signal is None:
    if seed is not None: np.random.seed(seed)
    signal = QMF.MakeSignalFromPSD((movie_total_length*365*24*2), 1/24, mean_signal_mag,
                        std_signal_mag, log_nu_b, alpha_L, alpha_H_minus_L)

# Update resolutions  (d1 is replaced with temp map)
d2 = rescale(d2, scale_ratio)
d3 = rescale(d3, scale_ratio)
d4 = rescale(d4, scale_ratio)

# Make new temp profile based on inputs
temp_map = QMF.AccDiskTemp(d4*gravrad, R_in*gravrad, mass, 2984, beta=asymp_slope,
                           coronaheight=lamppost_height, albedo=0, eta=lamp_strength,
                           genericbeta=True, eddingtons=edd_ratio, a=spin, visc_prof='SS')

# Make Disk object
Acc_disk = Amoeba.FlatDisk(mass_exp, redshift_q, header['numGRs'], header['inc_ang'],
                           lamppost_height, temp_map, d2, d3, d4, spin=header['spin'])

# Process passband values
static_brightness = Acc_disk.MakeSurfaceIntensityMap(initial_wavelength) * passband[1][0]
var_brightness = np.nan_to_num(Acc_disk.MakeDBDTMap(initial_wavelength) * passband[1][0] * Acc_disk.MakeDTDLxMap(initial_wavelength))
for jj in range(np.size(passband[1]) - 1):
    static_brightness += Acc_disk.MakeSurfaceIntensityMap(passband[0][jj+1]) * passband[1][jj+1]
    var_brightness += np.nan_to_num(Acc_disk.MakeDBDTMap(passband[0][jj+1]) * passband[1][jj+1] * Acc_disk.MakeDTDLxMap(passband[0][jj+1]))
time_lags = Acc_disk.MakeTimeDelayMap(jitters=False)
max_lag = np.max(time_lags)

# Normalize contributing maps
static_brightness /= np.sum(static_brightness)          # Total of each brightness is now "1"
var_brightness /= np.sum(var_brightness)                # Helps set variability scale



# Make Movie
movie_output = QMF.MakeSnapshots(static_brightness, var_brightness, time_lags.astype(int), snapshots,
                                 SignalWeighting = variability_weight, Signal = signal)

# Derive disk light curve
intrinsic_disk_LC = np.sum(movie_output, axis=(1, 2))


# Save data
HDU1 = fits.PrimaryHDU(movie_output)
HDU1.header['mexp'] = mass_exp
HDU1.header['zq'] = redshift_q
HDU1.header['inc_ang'] = inclination_angle
HDU1.header['spin'] = spin
HDU1.header['cheight'] = lamppost_height
HDU1.header['slope'] = asymp_slope
HDU1.header['eta_x'] = lamp_strength
HDU1.header['edds'] = edd_ratio
HDU1.header['tot_len'] = movie_total_length
HDU1.header['max_lag'] = max_lag
HDU2 = fits.ImageHDU(signal)
HDU3 = fits.ImageHDU(intrinsic_disk_LC)
HDU4 = fits.ImageHDU(snapshots)
HDUL = fits.HDUList([HDU1, HDU2, HDU3, HDU4])                           

HDUL.writeto(output_fname, overwrite=True)

print(time.time() - ts)



                                                                                                     











