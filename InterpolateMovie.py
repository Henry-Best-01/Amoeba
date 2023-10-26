import numpy as np
from astropy.io import fits
import json
import matplotlib.pyplot as plt
import os


Movie_file = "MyMovie.fits"
New_timestamps_file = "New_timestamps.json"




def interpolate_movie(Movie, Orig_timestamps, New_timestamps, verbose=False, plot=False):

    '''
    This function aims to take a series of snapshots and resample them at different time intervals.

    Movie is expected to be an array of (time, space_x, space_y),
    and Orig_timestamps is an array of timestamps representing the time values of each snapshot.
    New_timestamps is a list or array of new time stamps using the same units as Orig_timestamps.

    verbose allows some information to be printed to standard output.
    plot creates two plots, one with the original light curve plotted with the new interpolated light curve,
    and one where the second frame of each image is plotted side by side.

    returns an array representing the new resampled movie at timestamps New_timestamps, of shape
    (len(New_timestamps, np.size(Movie, 1), np.size(Movie, 2))
    '''
    import numpy as np
    import scipy
    

    initial_shape = np.shape(Movie)
    if verbose: print("The initial movie shape was",initial_shape)
    npix = initial_shape[1] * initial_shape[2]
    if verbose: print("This had",npix,"pixels")
    if verbose: print("Data will be interpolated to",len(New_timestamps), "frames")

    space_positions = np.linspace(1, npix, npix)                        # Define linear positions of pixels

    intermediate_movie = np.reshape(Movie, (initial_shape[0], npix))    # reshape image to a line

    interpolation = scipy.interpolate.RegularGridInterpolator((Orig_timestamps, space_positions),
                                            intermediate_movie, bounds_error=False, fill_value=None)

    new_timelength = np.size(New_timestamps)
    new_points_t, new_points_s = np.meshgrid(New_timestamps, space_positions, indexing='ij')
    movie_resampled = interpolation((new_points_t, new_points_s))
    if verbose: print("The new interpolated movie is now shape",np.shape(movie_resampled),"in time and spatial coordinates")

    # unpack to movie shape (time axis is and was first)
    resampled_movie = np.reshape(movie_resampled, (new_timelength, initial_shape[1], initial_shape[2])) 
    if verbose: print("The reconstructed movie is now shape",np.shape(resampled_movie))

    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

        ax.plot(np.asarray(Orig_timestamps), np.sum(Movie, axis=(1,2)), linewidth=2)
        ax.plot(np.asarray(New_timestamps), np.sum(resampled_movie, axis=(1,2)), '-o',alpha=0.5)

        ax.set_xlabel("Time [days]")
        ax.set_ylabel("Flux [arb]")
        
        fig2, ax2 = plt.subplots(1,2)

        conts1 = ax2[0].contourf(Movie[1])
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
            axis.set_xlim(0, initial_shape[1])
            axis.set_ylim(0, initial_shape[2])
            axis.set_aspect(1)
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3)
        
        plt.show()
    return resampled_movie        


if os.path.isfile(Movie_file) and os.path.isfile(New_timestamps_file):
    with fits.open(Movie_file) as f:
        Movie = f[0].data               # movie should have time on axis 0, while axis 1 and 2 are spatial
        Orig_timestamps = f[-1].data       # 1d array or list



    with open(New_timestamps_file) as f:
        json_input_file = json.load(f)
    New_timestamps = json_input_file['New_times']

    interpolate_movie(Movie, Orig_timestamps, New_timestamps, verbose=True, plot=True)
    interpolate_movie(Movie, Orig_timestamps, [10,20,30,40,50,5000,5010,5020,5300,10000,20000], verbose=True, plot=True)
    interpolate_movie(Movie, Orig_timestamps, np.linspace(0, 24*3000, 400), verbose=True, plot=True)

pixels = np.zeros((20, 2, 3))
for tt in range(np.size(pixels, 0)):

    pixels[tt, 0, 0] = np.sin(tt)
    pixels[tt, 0, 1] = -np.sin(tt)
    pixels[tt, 0, 2] = 0
    pixels[tt, 1, 0] = tt - 10
    pixels[tt, 1, 1] = 10 - tt
    pixels[tt, 1, 2] = np.exp(-tt)

new_movie = interpolate_movie(pixels, np.linspace(0, 20, 20), [0, 2, 5, 5.5, 7, 18.2], verbose=True, plot=False)


fig, ax = plt.subplots()
ax.plot(np.linspace(0, 20, 20), pixels[:, 0, 0], label="px orig [0,0]")
ax.plot(np.linspace(0, 20, 20), pixels[:, 0, 1], label="px orig [0,1]")
ax.plot(np.linspace(0, 20, 20), pixels[:, 0, 2], label="px orig [0,2]")
ax.plot(np.linspace(0, 20, 20), pixels[:, 1, 0], label="px orig [1,0]")
ax.plot(np.linspace(0, 20, 20), pixels[:, 1, 1], label="px orig [1,1]")
ax.plot(np.linspace(0, 20, 20), pixels[:, 1, 2], label="px orig [1,2]")
plt.gca().set_prop_cycle(None)
ax.plot([0, 2, 5, 5.5, 7, 18.2], new_movie[:, 0, 0], '-o', alpha=0.5, label='interp [0,0]')
ax.plot([0, 2, 5, 5.5, 7, 18.2], new_movie[:, 0, 1], '-o', alpha=0.5, label='interp [0,1]')
ax.plot([0, 2, 5, 5.5, 7, 18.2], new_movie[:, 0, 2], '-o', alpha=0.5, label='interp [0,2]')
ax.plot([0, 2, 5, 5.5, 7, 18.2], new_movie[:, 1, 0], '-o', alpha=0.5, label='interp [1,0]')
ax.plot([0, 2, 5, 5.5, 7, 18.2], new_movie[:, 1, 1], '-o', alpha=0.5, label='interp [1,1]')
ax.plot([0, 2, 5, 5.5, 7, 18.2], new_movie[:, 1, 2], '-o', alpha=0.5, label='interp [1,2]')
ax.legend(ncol=2)

ax.set_xlabel("Timestamp [arb.]")
ax.set_ylabel("Pixel value [arb.]")
plt.show()




 









