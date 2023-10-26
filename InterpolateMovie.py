
def InterpolateMovie(Movie_file, New_timestamps_file, verbose=False, plot=False):

    '''
    This function aims to take a series of snapshots and resample them at different time intervals.

    Movie_file is expected to be a FITS file with primary HDU as an array of (time, space_x, space_y),
    and an image HDU of timestamps representing the time values of each snapshot. The image HDU must be
    the final image HDU in the HDUList.
    New_timestamps_file is either a list or a JSON file path containing keyword "New_times" corresponding
    to a list of new time stamps using the same units as the timestamps in the original movie.

    verbose allows some information to be printed to standard output.
    plot creates two plots, one with the original light curve plotted with the new interpolated light curve,
    and one where the second frame of each image is plotted side by side.
    '''


    import numpy as np
    from astropy.io import fits
    import scipy
    import json
    import matplotlib.pyplot as plt


    with fits.open(Movie_file) as f:
        movie = f[0].data               # movie should have time on axis 0, while axis 1 and 2 are spacial
        signal_points = f[-1].data       # 1d array or list

    initial_shape = np.shape(movie)
    if verbose: print("The initial movie shape was",initial_shape)
    npix = initial_shape[1] * initial_shape[2]
    if verbose: print("This had",npix,"pixels")

    # assumes json_input_file has ['New_times'] as the list of new timestamps if New_timestamps_file is a string
    if type(New_timestamps_file) == str:
        with open(New_timestamps_file) as f:
            json_input_file = json.load(f)
        new_timestamps = json_input_file['New_times']
    # otherwise, we assume a list or array of new timestamps are provided
    else: new_timestamps = New_timestamps_file
    if verbose: print("Data will be interpolated to",len(new_timestamps), "frames")

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
    return resampled_movie        

InterpolateMovie("MyMovie.fits", "New_timestamps.json", verbose=True, plot=True)

InterpolateMovie("MyMovie.fits", [10,20,30,40,50,5000,5010,5020,5300,10000,20000], verbose=True, plot=True)

import numpy as np
InterpolateMovie("MyMovie.fits", np.linspace(0, 24*3000, 400), verbose=True, plot=True)















