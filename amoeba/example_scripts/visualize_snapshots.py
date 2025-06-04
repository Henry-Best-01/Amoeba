import numpy as np
import matplotlib.pyplot as plt
from amoeba.Classes.accretion_disk import AccretionDisk
from amoeba.Util.util import create_maps, generate_signal_from_psd
from astropy import units as u
from astropy import constants as const


# define an accretion disk
mexp = 8.6  # solution to log_10(M_smbh/M_sun)
redshift = 0.2  # dimensionless
number_grav_rad = 1000  # Rg
resolution = 1000  # total px
eddington_ratio = 0.05  # dimensionless
inclination = 45  # deg
my_observed_wavelength = 750  # nm
min_radius = 10  # Rg
spin = 0.2

# define observation times
snapshot_timestamps = np.linspace(3500, 8000, 50)

# define the driving signal
total_time = 10000  # days
time_axis = np.linspace(0, total_time - 1, int(total_time))
frequencies = np.linspace(1 / total_time, 1 / 2, int(total_time))
power_spectrum = frequencies ** (-3.0)
random_seed = 17
driving_signal_strength = 1.0
signal_amplitude = 10**23


# make some objects
my_disk_dictionary = create_maps(
    smbh_mass_exp=mexp,
    redshift_source=redshift,
    number_grav_radii=number_grav_rad,
    inclination_angle=inclination,
    resolution=resolution,
    spin=spin,
    eddington_ratio=eddington_ratio,
    visc_temp_prof="NT",  # use Novikov-Thorne disk for fun
)

my_accretion_disk = AccretionDisk(**my_disk_dictionary)

my_static_flux = my_accretion_disk.calculate_surface_intensity_map(
    my_observed_wavelength
)

my_driving_signal_times, my_driving_signal = generate_signal_from_psd(
    total_time, power_spectrum, frequencies, random_seed
)

my_driving_signal = signal_amplitude * my_driving_signal

my_snapshots = my_accretion_disk.generate_snapshots(
    my_observed_wavelength,
    snapshot_timestamps,
    my_driving_signal,
    driving_signal_strength,
)

my_normalization = np.sum(my_static_flux.flux_array) / np.sum(
    my_snapshots[0].flux_array
)

fig, ax = plt.subplots(3, 4, sharex="all", sharey="all")

for jj in range(min(len(my_snapshots), 10)):

    current_snapshot = (
        my_snapshots[jj].flux_array * my_normalization - my_static_flux.flux_array
    )

    X, Y = my_snapshots[jj].get_plotting_axes()
    contours = ax[jj // 4, jj % 4].contourf(X, Y, (current_snapshot), 20)
    cbar = plt.colorbar(contours, ax=ax[jj // 4, jj % 4])
    ax[jj // 4, jj % 4].set_aspect(1)
    ax[jj // 4, jj % 4].set_label(
        "snapshot of day:" + str(round(snapshot_timestamps[jj], 0))
    )

X, Y = my_static_flux.get_plotting_axes()
contours = ax[-1, -1].contourf(X, Y, (abs(my_static_flux.flux_array)))
cbar = plt.colorbar(contours, ax=ax[-1, -1])
ax[-1, -1].set_title("static case")
ax[-1, -1].set_aspect(1)

fig2, ax2 = plt.subplots(2, sharex="all")
ax2[0].plot(my_driving_signal_times, my_driving_signal, label="driving signal")
for jj in range(len(my_snapshots)):
    ax2[1].scatter(
        [snapshot_timestamps[jj]],
        [np.nansum(my_snapshots[jj].flux_array)],
        color="black",
        marker=".",
    )
ax2[0].legend()
ax2[0].set_title("driving signal and snapshot times")
ax2[1].set_title("sum of each snapshot")
ax2[1].set_xlabel("time [days]")


plt.show()
