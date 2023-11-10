'''
This simply writes a .json file to be read for InterpolateMovie.py
Note that movies were HOURLY positions
'''

import numpy as np
import json
from astropy import units as u

fname = "../SampleJsons/New_timestamps.json"

# Create set of timestamps
new_timestamps = np.linspace(0, 365*10, 400)
new_timestamps = [188, 238, 1087, 237, 365, 700, 4, 3200]
timestamp_units = u.d

# Convert input units to hourly units
dummy_times = np.asarray(new_timestamps, dtype=float)   # Make sure we can do array manipulations
dummy_times -= int(np.min(dummy_times))                 # Avoid issue of imputing MJDs
new_timestamps = dummy_times 
new_timestamps.sort()                                   # Arrange in ascending order
factor = timestamp_units.to(u.h)                        # Conversion factor through astropy.units
new_timestamps *= factor
prepped_timestamps = new_timestamps.tolist()

# Prepare output
outputs = {'New_times' : prepped_timestamps}

# Dump to .json
with open(fname, 'w') as file:
    json.dump(outputs, file)














