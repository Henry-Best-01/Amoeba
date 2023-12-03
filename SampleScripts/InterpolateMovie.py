import numpy as np
from astropy.io import fits
import json
import matplotlib.pyplot as plt
import os
import sys

if len(sys.argv) > 1:
    Movie_file = sys.argv[1]
else:
    print("Please provide a movie to work with")
if len(sys.argv) > 2:
    New_timestamps_file = sys.argv[2]
else:
    print("Please provide new time stamps to interpolate to")



def interpolate_movie(Movie, Orig_timestamps, New_timestamps, index=0, verbose=False, plot=False):

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

    if Movie.ndim == 4:
        Movie = Movie[index, :, :, :]

    initial_shape = np.shape(Movie)
    if verbose: print("The initial movie shape was",initial_shape)
    npix = initial_shape[-1] * initial_shape[-2]
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
    with fits.open(Movie_file) as fmov:
        Movie = fmov[0].data               # movie should have time on axis 0, while axis 1 and 2 are spatial
        Orig_timestamps = fmov[2].data       # 1d array or list
        staticHDU = fits.ImageHDU(fmov[1].data)
        wavelengthsHDU = fits.ImageHDU(fmov[-1].data)
        header = fmov[0].header


    with open(New_timestamps_file) as f:
        json_input_file = json.load(f)
    New_timestamps = json_input_file['New_times']

    if Movie.ndim == 4:
        output = np.zeros((np.size(Movie, 0), np.size(New_timestamps), np.size(Movie, 2), np.size(Movie, 3)))
        for index in range(np.size(Movie, 0)):
            output[index, :, :, :] = interpolate_movie(Movie, Orig_timestamps, New_timestamps, index=index)
    else:
        output = interpolate_movie(Movie, Orig_timestamps, New_timestamps)

    HDU = fits.PrimaryHDU(output)
    HDU.header['output_type'] = header['output_type']
    
    
    timestampsHDU = fits.ImageHDU(New_timestamps)
    HDUL = fits.HDUList([HDU, staticHDU, timestampsHDU, wavelengthsHDU])
    print("Please enter file name to save as (leave blank to cancel):")
    save_file_name = input()
    if save_file_name != '':
        HDUL.writeto(save_file_name+'.fits', overwrite=True)





 









