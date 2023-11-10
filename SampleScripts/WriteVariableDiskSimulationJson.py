"""
This writes a few sample .json inputs for 'VariableDiskSimulation.py'.

File 1: DRW driving signal with standard SS disk outputting 1 wavelength
File 2: DRW driving signal with NT disk outputting 3 wavelengths
File 3: BPL driving signal with user defined temp profile outputting 1 wavelength
File 4: user defined signal with SS-plus disk outputting in a defined filter

Global (common) parameters to all models are defined above Generate File 1
"""

import json
import sys
import numpy as np
import scipy

# save_file names
fname1 = "../SampleJsons/Variability_model_1.json"
fname2 = "../SampleJsons/Variability_model_2.json"
fname3 = "../SampleJsons/Variability_model_3.json"
fname4 = "../SampleJsons/Variability_model_4.json"


mass_exponent = 8.0
redshift = 3.0
time = 4.0
time_steps = 300
corona_height = 10
omg0 = 0.0
omgL = 0.7
H0 = 70



################ Generate File 1 #################
Step_1 = 'DRW'
tau = 30
SF_inf = 100

Step_2 = 'SS'
num_GRs = 1000
inc_ang = 30
resolution = 500
eddington_ratio = 0.15

Step_3 = 'wavelength'
lam = 700

Step_4 = 'LC-multi'

json_dict = {'Step_1' : Step_1,
             'Step_2' : Step_2,
             'Step_3' : Step_3,
             'Step_4' : Step_4,
             'mass_exponent' : mass_exponent,
             'redshift' : redshift,
             'time' : time,
             'time_steps' : time_steps,
             'corona_height' : corona_height,
             'omg0' : omg0,
             'omgL' : omgL,
             'H0' : H0,
             'tau' : tau,
             'SF_inf' : SF_inf,
             'num_GRs' : num_GRs,
             'inc_ang' : inc_ang,
             'resolution' : resolution,
             'eddington_ratio' : eddington_ratio,
             'lam' : lam
             }
with open(fname1, 'w') as file:
    json.dump(json_dict, file)

################### End File 1 ###################

################ Generate File 2 #################
Step_1 = 'DRW'
tau = 50
SF_inf = 10

Step_2 = 'NT'
num_GRs = 1000
inc_ang = 60
resolution = 500
eddington_ratio = 0.15
spin = 0.5

Step_3 = 'wavelengths'
lams = [600, 700, 800]

Step_4 = 'LC-multi'

json_dict = {'Step_1' : Step_1,
             'Step_2' : Step_2,
             'Step_3' : Step_3,
             'Step_4' : Step_4,
             'mass_exponent' : mass_exponent,
             'redshift' : redshift,
             'time' : time,
             'time_steps' : time_steps,
             'corona_height' : corona_height,
             'omg0' : omg0,
             'omgL' : omgL,
             'H0' : H0,
             'tau' : tau,
             'SF_inf' : SF_inf,
             'num_GRs' : num_GRs,
             'inc_ang' : inc_ang,
             'resolution' : resolution,
             'eddington_ratio' : eddington_ratio,
             'spin' : spin,
             'lams' : lams
             }

with open(fname2, 'w') as file:
    json.dump(json_dict, file)


################### End File 2 ###################

################ Generate File 3 #################
Step_1 = 'BPL'
mean_magnitude = 20
standard_dev = 1
log_nu_b = 0
alpha_L = -1
alpha_H = -3

Step_2 = 'user'
num_GRs = 1000
inc_ang = 0
resolution = 500
radial_vals = np.linspace(1, 1000, 1000)
FWHM = 70  # units : R_g!
profile = (5e4 * np.exp(-radial_vals**2 / (FWHM / 2.35)**2)) #Gaussian
profile *= (profile > 1e-5)
profile = profile.tolist()

Step_3 = 'wavelength'
lam = 1300

Step_4 = 'LC-multi'

json_dict = {'Step_1' : Step_1,
             'Step_2' : Step_2,
             'Step_3' : Step_3,
             'Step_4' : Step_4,
             'mass_exponent' : mass_exponent,
             'redshift' : redshift,
             'time' : time,
             'time_steps' : time_steps,
             'corona_height' : corona_height,
             'omg0' : omg0,
             'omgL' : omgL,
             'H0' : H0,
             'mean_magnitude' : mean_magnitude,
             'standard_dev' : standard_dev,
             'log_nu_b' : log_nu_b,
             'alpha_L' : alpha_L,
             'alpha_H' : alpha_H,
             'num_GRs' : num_GRs,
             'inc_ang' : inc_ang,
             'resolution' : resolution,
             'eddington_ratio' : eddington_ratio,
             'profile' : profile,
             'lam' : lam
             }
with open(fname3, 'w') as file:
    json.dump(json_dict, file)

################### End File 3 ###################


################ Generate File 4 #################
Step_1 = 'user'
time_vals = np.linspace(0, 1e4, 10000) # make a long signal!
signal = (np.cos(time_vals * np.pi / 180)).tolist()

Step_2 = 'SS-plus'
num_GRs = 1000
inc_ang = 30
resolution = 500
eddington_ratio = 0.15
eta_x = 0.05
beta = 0.5

Step_3 = 'user-band'
lams = [700, 720, 740, 760, 780, 800]
throughput = [0.7, 0.8, 0.9, 0.95, 0.8, 0.7]

Step_4 = 'LC-multi'

json_dict = {'Step_1' : Step_1,
             'Step_2' : Step_2,
             'Step_3' : Step_3,
             'Step_4' : Step_4,
             'mass_exponent' : mass_exponent,
             'redshift' : redshift,
             'time' : time,
             'time_steps' : time_steps,
             'corona_height' : corona_height,
             'omg0' : omg0,
             'omgL' : omgL,
             'H0' : H0,
             'signal' : signal,
             'num_GRs' : num_GRs,
             'inc_ang' : inc_ang,
             'resolution' : resolution,
             'eddington_ratio' : eddington_ratio,
             'eta_x' : eta_x,
             'beta' : beta,
             'lams' : lams,
             'throughput' : throughput
             }
with open(fname4, 'w') as file:
    json.dump(json_dict, file)









































