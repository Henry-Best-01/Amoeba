'''
This script creates a few plots to show the .fits file produced with
"VariablediskSimulation.py".
It prepares a plot of:
    - The total variability of each snapshot summed over all pixels
    - A contour plot of the first snapshot if snapshots are saved
'''
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import sys
sys.path.append("../Functions")
import QuasarModelFunctions as QMF
import glob

if len(sys.argv) > 1:
    file_name = sys.argv[1]
else:
    potential_fnames = glob.glob("*.fits")
    file_name = potential_fnames[0]

with fits.open(file_name) as f:
    static_out = f[1].data
    variable_out = f[0].data
    timestamps = f[2].data
    wavelengths = f[3].data
    header = f[0].header



fig, ax = plt.subplots()
if header['output_type'][:9] == 'snapshots':
    light_curve = np.sum(variable_out, axis=(-1,-2))
else:
    light_curve = variable_out
if header['output_type'][-3:] == 'sum':
    ax.plot(timestamps, light_curve)
    ax.set_title("sum of wavelengths:" + str(wavelengths) + " nm")
else:
    for wavelength in range(np.size(light_curve, 0)):
        ax.plot(timestamps, light_curve[wavelength], label=str(wavelengths[wavelength])+" nm")
ax.set_xlabel("Time [days]")
ax.set_ylabel("Flux [arb.]")
if header['output_type'] == 'LC-multi':
    ax.legend()
plt.show()

if header['output_type'][:9] == 'snapshots':
    animation = QMF.animate_snapshots(variable_out, interval=10)


