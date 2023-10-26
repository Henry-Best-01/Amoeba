'''
This script aims to take a series of snapshots and resample them at different
time intervals to facilitate not recreating the movies every time.
'''
import numpy as np
from astropy.io import fits
import scipy
import json
import matplotlib.pyplot as plt

input1 = "MyMovie.fits"                 # Movie stored as fits file
input2 = "New_timestamps.json"          # New sample times to interpolate to

verbose = True
plot = True


# original movie file was assumed to be f[0].data = movie, f[1].data = disk_LC (movie collapsed on spacial axes),
# f[2].data = intrinsic X-ray signal which generated the movie, and f[3].data = snapshot timestamps

with fits.open(input1) as f:
    movie = f[0].data               # movie should have time on axis 0, while axis 1 and 2 are spacial
    signal_points = f[-1].data       # 1d array or list


with open(input2) as f:             # assumes json_input_file has ['New_times'] as the list of new (hourly) timestamps
    json_input_file = json.load(f)
new_timestamps = json_input_file['New_times']
if verbose:
    print("Data will be interpolated to",len(new_timestamps), "points")


initial_shape = np.shape(movie)
if verbose: print("The initial movie shape was",initial_shape)
npix = initial_shape[1] * initial_shape[2]
if verbose: print("This had",npix,"pixels")

space_positions = np.linspace(1, npix, npix)                        # Define a 2d pixel-time space
time_positions = signal_points
T, S = np.meshgrid(time_positions, space_positions, indexing='ij')

intermediate_movie = np.reshape(movie, (initial_shape[0], npix))    # reshape image to a line

interpolation = scipy.interpolate.RegularGridInterpolator((time_positions, space_positions),
                                        intermediate_movie, bounds_error=False, fill_value=None)

new_timelength = np.size(new_timestamps)
new_points_t, new_points_s = np.meshgrid(new_timestamps, space_positions, indexing='ij')
movie_resampled = interpolation((new_points_t, new_points_s))
if verbose: print("The new interpolated movie is now shape",np.shape(movie_resampled),"in time and spacial coordinates")


# unpack to movie shape (time axis is and was first)
resampled_movie = np.reshape(movie_resampled, (new_timelength, initial_shape[1], initial_shape[2])) 
if verbose: print("The reconstructed movie is now shape",np.shape(resampled_movie))


if plot:
    fig, ax = plt.subplots()

    ax.plot(np.asarray(time_positions)/24, np.sum(movie, axis=(1,2)), linewidth=2)
    ax.plot(np.asarray(new_timestamps)/24, np.sum(resampled_movie, axis=(1,2)), '-o',alpha=0.5)

    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Flux [arb]")
    
    fig2, ax2 = plt.subplots(1,2)

    conts1 = ax2[0].contourf(movie[1])
    conts2 = ax2[1].contourf(resampled_movie[1])

    ax2[0].set_xlabel("x [px]")
    ax2[1].set_xlabel("x [px]")
    ax2[0].set_ylabel("y [px]")
    ax2[1].set_ylabel("y [px]")
    ax2[0].set_title("Orig. movie, frame 2")
    ax2[1].set_title("Interp. movie, frame 2")

    plt.colorbar(conts1, ax=ax2[0], label="Flux [arb.]")
    plt.colorbar(conts2, ax=ax2[1], label="Flux [arb.]")


    for axis in ax2:
        axis.set_xlim(4*initial_shape[1]/9, 5*initial_shape[1]/9)
        axis.set_ylim(4*initial_shape[2]/9, 5*initial_shape[2]/9)
        axis.set_aspect(1)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3)
    
    plt.show()
    


















