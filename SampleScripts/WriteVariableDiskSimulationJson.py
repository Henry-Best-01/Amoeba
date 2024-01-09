"""
This writes a few sample .json inputs for 'VariableDiskSimulation.py'.

File 1: DRW driving signal with standard SS disk outputting 1 wavelength snapshots
File 2: DRW driving signal with a ray traced NT disk outputting 3 wavelengths
File 3: BPL driving signal with power-law temp profile outputting 1 wavelength
File 4: user defined signal with SS-plus disk outputting in a defined filter

Global (common) parameters to all models are defined above Generate File 1

Also writes a sample throughput and signal file
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

fname_signal = "../SampleJsons/Sample_signal.json"
fname_throughput = "../SampleJsons/Sample_throughput.json"

path_to_ray_trace = "../SampleDisks/RayTrace.fits"


mass_exponent = 8.0
redshift = 1.0
time = 2.0
time_steps = 2000
corona_height = 10
omg0 = 0.3
omgL = 0.7
H0 = 70



################ Generate File 1 #################
Step_1 = 'user'
tau = 30
SF_inf = 100

Step_2 = 'SS'
num_GRs = 200
inc_ang = 30
resolution = 200
eddington_ratio = 0.15

Step_3 = 'wavelength'
lam = 700

Step_4 = 'snapshots-multi'

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
inc_ang = 45
resolution = 500
eddington_ratio = 0.15
spin = 0.7

time_steps = 10000

Step_3 = 'wavelengths'
lams = [400, 600, 1000]

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
             'lams' : lams,
             'ray_trace_fits' : path_to_ray_trace,
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

Step_2 = 'power-law'
num_GRs = 1000
inc_ang = 10
resolution = 1000
power = 2.0    # Thin disk has power = 3/4

time_steps = 2000

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
             'mean_magnitude' : mean_magnitude,
             'standard_dev' : standard_dev,
             'log_nu_b' : log_nu_b,
             'alpha_L' : alpha_L,
             'alpha_H' : alpha_H,
             'num_GRs' : num_GRs,
             'inc_ang' : inc_ang,
             'resolution' : resolution,
             'eddington_ratio' : eddington_ratio,
             'power' : power,
             'lam' : lam
             }
with open(fname3, 'w') as file:
    json.dump(json_dict, file)

################### End File 3 ###################


################ Generate File 4 #################
Step_1 = 'user'
time_vals = np.linspace(0, 1e4, 10000) # make a long signal!
signal = (np.cos(time_vals * np.pi / 180)).tolist()

signal_dict = {'signal' : signal}

with open(fname_signal, 'w') as file:
    json.dump(signal_dict, file)

Step_2 = 'SS-plus'
num_GRs = 1000
inc_ang = 30
resolution = 500
eddington_ratio = 0.15
eta_x = 0.05
beta = 0.5

Step_3 = 'throughput'
lams = [700, 720, 740, 760, 780, 800]
throughput = [0.7, 0.8, 0.9, 0.95, 0.8, 0.7]

throughput_dict = {'lams' : lams,
                   'throughput' : throughput
                   }

with open(fname_throughput, 'w') as file:
    json.dump(throughput_dict, file)
    

Step_4 = 'LC-sum'

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
             'num_GRs' : num_GRs,
             'inc_ang' : inc_ang,
             'resolution' : resolution,
             'eddington_ratio' : eddington_ratio,
             'eta_x' : eta_x,
             'beta' : beta,
             }
with open(fname4, 'w') as file:
    json.dump(json_dict, file)









































