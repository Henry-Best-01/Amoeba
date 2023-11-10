"""
Feed me a .json with three sections in order to produce a time
varying signal! You may apply a seed at the top if you like.

Step_1: Choose your signal, or feed one directly in!
    -'DRW' is a damped random walk, and requires two parameters:
        tau the characteristic timescale
        SF_inf the asymptotic structure function
    -'BPL' is a broken (or bending) power law, and requires:
        mean_magnitude the mean magnitude
        standard_dev the standard deviation
        log_nu_b the breakpoint frequency
        alpha_L the low frequency power slope of the PSD
        alpha_H the high frequency power slope of the PSD
    -'user' is a user defined signal. This requires:
        signal, a list of amplitudes sampled at daily cadence
        and will be interpolated between

Step_2: Choose your disk reprocessing model, or feed one directly in!
    *Note* Reprocessing depends on temperature profile, so we define T(r) here
    -'SS' is the basic Shakura-Sunyaev accretion disk in
    the lamppost geometry. This requires:
        mass_exponent the value log_10(M_bh / M_sun) (typically 7-10)
        num_GRs the number of gravitational radii to use
        inc_ang the inclination in degrees
        resolution the resolution of the disk's image
        eddington_ratio the Eddington ratio of the disk
    -'NT' is the Novikov-Thorne disk which takes all above params plus:
        spin the dimensionless spin of the black hole (positive is prograde)
    -'SS-plus' allows for additional control over the profile with params:
        eta_x the irradiating strength of the corona such that L_x = eta_x M_acc c^2
        beta the wind parameter making the profile scale like T \propto r^(-(3-beta)/4)
    *Note* SS-plus converges to SS for eta_x = beta = 0
    -'user' is a user defined profile:
        profile, a 1 dimensional temperature profile given as a
        list at every gravitational radius. Values not provided will default to T=0.

Step_3: Choose what you want calculated
    -'wavelength' is a single wavelength and requires:
        lam the wavelength, in nanometers
    -'wavelengths' is multiple wavelengths
        lams a list of wavelengths in nanometers
    -'user-band' is a wave filter requiring:
        lams a list of wavelengths in nanometers
        throughput a list of weighting factors for each wavelength

Step_4: Choose your output, will be output as array (wavelengths, timestamps, xvals, yvals)
    -'LC-multi' will return a 1 dimensional light curve for each wavelength provided
    -'LC-sum' will return a 1 dimensional light curve for the sum of wavelengths provided
    -'snapshot-multi' will return a 3 dimensional set of images where each pixel
        is a spatially resolved light curve
    -'snapshot-sum' will return a 3 dimensional image where each pixel is a
        spatially resolved light curve

Other global params:
    -'redshift' the redshift of the system
    -'time' the total time (in years) of the resulting light curve (curve will be units days)
    -'time_steps' if int: number of evenly spaced time stamps to create
        if list: time stamps (in days) when observations are simulated at
    -'corona_height' the height of the irradiating corona
    -'ray_trace_fits' a path to a fits file containing t_map, v_map, g_map, and r_map
        (such files obtainable at https://drive.google.com/drive/folders/1vx8HUBXw6SaDq5uS4jQCyWdg13XfCRCv?usp=sharing)
    -'omg0' Omega_0 for flat lambda CDM
    -'omgL' Omega_lambda for flat lambda CDM
    -'H0' H_0 for flat lambda CDM
    -'save_file' is a string to save the output to (as a fits file)
"""
import json
import sys
sys.path.append("../Functions/")
sys.path.append("../Classes/")
import numpy as np
import scipy
import Amoeba
import QuasarModelFunctions as QMF
from os import path
import skimage
from astropy.io import fits
from astropy import constants as const
from astropy import units as u


input_file = "../SampleJsons/Run_Variability_Json.json"

# Step 0 open json with params, make sure all exist

if path.is_file(input_file) != True:
    print("No valid json at target location:", inputs)

with open(input_file) as f:
    inputs = json.load(input_file)
type_signal = inputs['Step_1']
type_reverb = inputs['Step_2']
type_wavelengths = inputs['Step_3']
type_output = inputs['Step_4']

if inputs.get('seed') == True:
    np.random.seed(inputs['seed'])

if type_signal == 'DRW':
    drw_tau = inputs['tau']
    drw_sf_inf = inputs['SF_inf']
elif type_signal == 'BPL':
    mean_magnitude = inputs['mean_magnitude']
    standard_dev = inputs['standard_dev']
    log_nu_b = inputs['log_nu_b']
    alpha_L = inputs['alpha_L']
    alpha_H = inputs['alpha_H']
    alpha_H_minus_L = alpha_H - alpha_L
elif type_signal == 'user':
    signal = inputs['signal']

m_exp = inputs['mass_exponent']
num_grs = inputs['num_GRs']
inc_ang = inputs['inc_ang']
resolution = inputs['resolution']
c_height = inputs['corona_height']
full_time = inputs['time'] * 365
time_steps = inputs['time_steps']
omg0 = inputs['omg0']
omgL = inputs['omgL']
H0 = inputs['H0']

if type(time) == list:
    timestamps = time_steps
    if timestamps[-1] > 2 * full_time:
        full_time = 2 * timestamps[-1]
else:
    total_time = full_time
    timestamps = np.linspace(0, total_time, time_steps)
    full_time *= 2

if type_reverb != 'user':
    eddington_ratio = inputs['eddington_ratio']
if type_reverb == 'NT':
    spin = inputs['spin']
if type_reverb == 'SS-plus':
    eta_x = inputs['eta_x']
    beta = inputs['beta']
if num_grs > 20 * resolution:
    print("Warning, under-resolved accretion disk (1 px < 10 Rg)")
if num_grs < 0.2 * resolution and m_exp < 8:
    print("Warning, potentially over-resolved accretion disk (1 Rg > 10 px)")
if inc_ang == 0: inc_ang += 0.01
if type(resolution) != int:
    resolution = int(resolution)

if type_wavelengths == 'wavelength':
    obs_wavelengths = [inputs['lam']]
if type_wavelengths == 'wavelengths':
    obs_wavelengths = inputs['lams']
if type_wavelength == 'user-band':
    obs_wavelengths = inputs['lams']
    throughputs = inputs['throughput']

redshift = inputs['redshift']

assert redshift >= 0
assert inc_ang < 90
assert c_height >= 0


# Step 1 prepare signal (in hourly units)
output_length = inputs['time'] * 365
if type_signal == 'DRW':
    SIGNAL = QMF.MakeDRW(total_time/365, 1, drw_sf_inf, drw_tau)
if type_signal == 'BPL':
    SIGNAL = QMF.MakeSignalFromPSD(total_time, 100, mean_magnitude, standard_dev, log_nu_b, alpha_L, alpha_H_minus_L)
if type_signal == 'user':
    SIGNAL = signal
    n_loops = 0
    while len(SIGNAL) < 2 * full_time:
        SIGNAL.append(SIGNAL)
        n_loops += 1
    if n_loops > 0:
        print("Input signal wasn't long enough, wrapped", n_loops, "time(s)")
        print("Signal should be at least 2 times the total length of light curve desired")
    

# Step 2 prepare reprocessing map
if inputs.get('ray_trace_fits') is not None:
    with fits.open(inputs['ray_trace_fits']) as f:
        t_map = f[0].data.T
        v_map = f[1].data.T
        g_map = f[2].data.T
        r_map = f[3].data.T
    if np.size(r_map, 0) != resolution:
        print("Warning, resolution should be:", np.size(r_map, 0))
        print("Rescaling ray-traced maps to match...")
        ratio = np.size(r_map, 0)) / resolution
        dummy_map = skimage.transform.rescale(t_map, 1/ratio)
        t_map = dummy_map
        dummy_map = skimage.transform.rescale(v_map, 1/ratio)
        v_map = dummy_map
        dummy_map = skimage.transform.rescale(g_map, 1/ratio)
        g_map = dummy_map
        dummy_map = skimage.transform.rescale(r_map, 1/ratio)
        r_map = dummy_map
else:
    x_values = np.linspace(-num_grs, num_grs, resolution)
    y_values = x_values.copy() / np.cos(inc_ang * np.pi / 180)
    X, Y = np.meshgrid(x_values, y_values, indexing='ij')
    r_map = (X**2 + Y**2)**0.5
    v_map = np.zeros(np.shape(r_map))
    g_map = np.ones(np.shape(r_map))
    
mass = 10**m_exp * const.M_sun.to(u.kg)
gravitational_radius = QMF.CalcRg(mass)
    
if type_reverb == 'user':
    r_values = np.linspace(1, len(inputs['profile']), len(inputs['profile']))
    temp_values = inputs['profile']
if type_reverb == 'SS':
    r_values = np.linspace(1, num_grs, 10*num_grs)
    temp_values = QMF.AccDiskTemp(r_values * gravitational_radius, 6 * gravitational_radius,
                                  mass, 0, eddingtons = eddington_ratio, visc_prof='SS')
if type_reverb == 'NT':
    r_values = np.linspace(1, num_grs, 10*num_grs)
    R_in = QMF.SpinToISCO(spin)
    temp_values = QMF.AccDiskTemp(r_values * gravitational_radius, R_in * gravitational_radius,
                                  mass, 0, eddingtons = eddington_ratio, a=spin, visc_prof='NT')
if type_reverb == 'SS-plus':
    r_values = np.linspace(1, num_grs, 10*num_grs)
    R_in = QMF.SpinToISCO(spin)
    temp_values = QMF.AccDiskTemp(r_values * gravitational_radius, R_in * gravitational_radius,
                                  mass, 0, coronaheight=c_height, eta=eta_x, beta=beta,
                                  eddingtons = eddington_ratio, a=spin, visc_prof='SS')    
prof_interpolation = scipy.interpolate.interp1d(r_values, temp_values,
                                bounds_error=False, fill_value='extrapolate')
t_map = prof_interpolation(r_map)

Acc_Disk = Amoeba.FlatDisk(m_exp, redshift, num_grs, inc_ang, c_height, t_map, v_map, g_map,
                           r_map, spin=spin, omg0=omg0, omgl=omgL, H0=H0)

# Step 2.5 initialize output, in case the file will be too large we should know before we make it
# ESPECIALLY WITH SNAPSHOT-MULTI MODE OUTPUT which is a 4dim array, of shape (nwavelengths, ntimestamps, imgx, imgy)

if type_output == 'LC-multi':
    output = np.zeros((len(wavelengths), len(timestamps)))
    static_out = np.zeros(len(wavelengths))
if type_output == 'LC-sum':
    output = np.zeros(len(timestamps))
    static_out = 0
if type_output == 'snapshot-multi':
    output = np.zeros((len(wavelengths), len(timestamps), np.size(r_map, 0), np.size(r_map, 1)))
    static_out = np.zeros((len(wavelengths), np.size(r_map, 0), np.size(r_map, 1)))
if type_output == 'snapshot-sum':
    output = np.zeros((len(timestamps), np.size(r_map, 0), np.size(r_map, 1)))
    static_out = np.zeros((np.size(r_map, 0), np.size(r_map, 1)))
    

# Step 3 start calculating responses 

if len(obs_wavelengths) > 20:
    print("More than 20 wavelengths detected, may take a while...")


if type_output == 'snapshots-multi':
    td_map = AccDisk.MakeTimeDelayMap()
elif type_output == 'snapshots-sum':
    td_map = AccDisk.MakeTimeDelayMap()
for ii in range(len(obs_wavelengths)):
    


    if type_output == 'LC-multi' or 'LC-sum':
        TF_current = Acc_Disk.ConstructDiskTransferFunction(obs_wavelengths[ii], scaleratio=10)
        signal_contribution = QMF.Convolve_TF_With_Signal(SIGNAL, TF_current, timestamps)
        static_emission = np.sum(Acc_Disk.MakeSurfaceIntensityMap(obs_wavelengths[ii]))
    elif type_output == 'snapshots-multi' or 'snapshots-sum':
        dBdT = Acc_Disk.MakeDBDTMap(obs_wavelengths[ii])
        dTdLx = Acc_Disk.MakeDTDLxMap(obs_wavelengths[ii])
        static_emission = Acc_Disk.MakeSurfaceIntensityMap(obs_wavelengths[ii]) 
        signal_contribution = QMF.MakeSnapshots(np.zeros(np.shape(static_emission)), dBdT * dTdLx, td_map,
                                     Signal=SIGNAL)
    if type_wavelengths == 'user-band':
        signal_contribution *= throughputs[ii]
        static_emission *= throughputs[ii]
        
    if type_output == 'LC-multi':
        output[ii, :] += signal_contribution
        static_out[ii, :] += static_emission
    elif type_output == 'LC-sum':
        output += signal_contribution
        static_out += static_emission
    elif type_output == 'snapshots-multi':
        output[ii, :, :, :] += signal_contribution
        static_out[ii, :, :] += static_emission
    elif type_output == 'snapshots_sum':
        output += signal_contribution
        static_out += static_emission

# Step 4: return output!


HDU = fits.PrimaryHDU(output)
staticHDU = fits.ImageHDU(static_out)
HDUL = fits.HDUList([HDU, staticHDU])
HDUL.writeto(inputs['save_file'])



















