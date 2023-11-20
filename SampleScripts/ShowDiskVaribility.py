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

if len(sys.argv) > 1:
    file_name = sys.argv[1]
else:
    file_name = "VariableDisk2.fits"

with fits.open(file_name) as f:
    static_out = f[1].data
    variable_out = f[0].data
    timestamps = f[2].data
    header = f[0].header



fig, ax = plt.subplots()
if header['output_type'][:9] == 'snapshots':
    light_curve = np.sum(variable_out, axis=(-1,-2))
else:
    light_curve = variable_out
if header['output_type'][-3:] == 'sum':
    ax.plot(timestamps, light_curve)
else:
    for wavelength in range(np.size(light_curve, 0)):
        ax.plot(timestamps, light_curve[wavelength])
ax.set_xlabel("Time [days]")
ax.set_ylabel("Flux [arb.]")
plt.show()

if header['output_type'][:9] == 'snapshots':
    animation = QMF.animate_snapshots(variable_out, interval=10)


