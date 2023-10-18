'''
This writes a .json file to use with "CreateMovie.py"
'''
import numpy as np
import json
import QuasarModelFunctions as QMF

fname = "json_inputs.json"

mass_exp = 8.0
redshift_q = 2.0
inc_ang = 47
spin = -0.7
lamp_height = 10
asympt_slope = 0.75
lamp_strength = 0.1
eddington_ratio = 0.1
movie_length = 10 # years
mean_signal_mag = 1
std_signal_mag = 0.1
log_nu_b = -2
alpha_L = 1
alpha_H_minus_L = 1
var_weighting = 1.0 # compared to static brightness
snapshots = 200
output_res = 400 # pixels per side
signal = None  #  Input predefined signal if desired
seed = 5


passband = [[500, 510, 520, 530, 540, 550], [0.4, 0.5, 0.8, 0.9, 0.5, 0.4]]

output_vals = {'mass' : mass_exp,
               'zq' : redshift_q,
               'theta' : inc_ang,
               'spin' : spin,
               'lamp_height' : lamp_height,
               'slope' : asympt_slope,
               'eta_x' : lamp_strength,
               'eddingtons' : eddington_ratio,
               'total_time' : movie_length,
               'signal_weighting' : var_weighting,
               'snapshots' : snapshots,
               'passband' : passband,
               'resolution' : output_res,
               'mean_signal_mag' : mean_signal_mag,
               'std_signal_mag' : std_signal_mag,
               'log_nu_b' : log_nu_b,
               'alpha_L' : alpha_L,
               'alpha_H_minus_L' : alpha_H_minus_L,
               'signal' : signal,
               'seed' : seed
               }

with open(fname, 'w') as file:
    json.dump(output_vals, file)





