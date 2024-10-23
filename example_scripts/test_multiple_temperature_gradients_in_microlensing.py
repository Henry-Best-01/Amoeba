import numpy as np
import numpy.testing as npt
from amoeba.Classes.accretion_disk import AccretionDisk
from amoeba.Classes.magnification_map import MagnificationMap
from amoeba.Util.util import (
    create_maps,
)
from astropy import units as u
import matplotlib.pyplot as plt
from astropy.io import fits

plt.style.use("/Users/henrybest/PythonStuff/Code/plot_style.txt")

magmapfile = "/Users/henrybest/PythonStuff/LensingMaps/SampleMagMaps/map_1/map.fits"
with fits.open(magmapfile) as f:
    map2d = f[0].data


convergence = 0.3
shear = 0.4


smbh_mass_exp = 8.0
beta1 = 0.75
beta2 = 0.55


redshift_source = 2.0
redshift_lens = 1.0
inclination_angle = 0.0
corona_height = 10
number_grav_radii = 1000
resolution = 2000
spin = 0.0
wavelength_1 = 400
wavelength_2 = 900

transverse_vel = 800  # km/s
duration = 20  # years
x_start = 3000
y_start = 3000

magnification_map_initial = MagnificationMap(
    redshift_source,
    redshift_lens,
    map2d,
    convergence,
    shear,
)


accretion_disk_data_1 = create_maps(
    smbh_mass_exp,
    redshift_source,
    number_grav_radii,
    inclination_angle,
    resolution,
    spin=spin,
    temp_beta=beta1,
    corona_height=corona_height,
    generic_beta=True,
)
Disk = AccretionDisk(**accretion_disk_data_1)


disk_emission_1 = Disk.calculate_surface_intensity_map(wavelength_1)
disk_emission_2 = Disk.calculate_surface_intensity_map(wavelength_2)


accretion_disk_data_2 = create_maps(
    smbh_mass_exp,
    redshift_source,
    number_grav_radii,
    inclination_angle,
    resolution,
    spin=spin,
    temp_beta=beta2,
    corona_height=corona_height,
    generic_beta=True,
)
Disk2 = AccretionDisk(**accretion_disk_data_2)


disk_emission_1_2 = Disk2.calculate_surface_intensity_map(wavelength_1)
disk_emission_2_2 = Disk2.calculate_surface_intensity_map(wavelength_2)


convolution_1 = magnification_map_initial.convolve_with_flux_projection(
    disk_emission_1,
)

convolution_2 = magnification_map_initial.convolve_with_flux_projection(
    disk_emission_2,
)

convolution_1_2 = magnification_map_initial.convolve_with_flux_projection(
    disk_emission_1_2,
)

convolution_2_2 = magnification_map_initial.convolve_with_flux_projection(
    disk_emission_2_2,
)


random_seed = 3

lc_1 = convolution_1.pull_light_curve(
    transverse_vel,
    duration,
    random_seed=random_seed,
    x_start_position=x_start,
    y_start_position=y_start,
)
lc_2 = convolution_2.pull_light_curve(
    transverse_vel,
    duration,
    random_seed=random_seed,
    x_start_position=x_start,
    y_start_position=y_start,
)
lc_3 = magnification_map_initial.pull_light_curve(
    transverse_vel,
    duration,
    random_seed=random_seed,
    x_start_position=x_start,
    y_start_position=y_start,
)


lc_1_2 = convolution_1_2.pull_light_curve(
    transverse_vel,
    duration,
    random_seed=random_seed,
    x_start_position=x_start,
    y_start_position=y_start,
)
lc_2_2 = convolution_2_2.pull_light_curve(
    transverse_vel,
    duration,
    random_seed=random_seed,
    x_start_position=x_start,
    y_start_position=y_start,
)


color_disk_1 = np.log10(np.asarray(lc_1) / np.asarray(lc_2))
color_disk_2 = np.log10(np.asarray(lc_1_2) / np.asarray(lc_2_2))


tax = np.linspace(0, duration, np.size(lc_1))

fig, ax = plt.subplots(2, sharex=True)
ax[0].plot(tax, lc_1 / np.max(lc_1), label="thin disk, wavelength 1")
ax[0].plot(tax, lc_2 / np.max(lc_2), label="thin disk, wavelength 2")
ax[0].plot(tax, lc_1_2 / np.max(lc_1_2), label="slim disk, wavelength 1")
ax[0].plot(tax, lc_2_2 / np.max(lc_2_2), label="slim disk, wavelength 2")

ax[0].plot(tax, lc_3 / np.max(lc_3), color="black", linewidth=0.5, label="point source")

ax[1].plot(tax, color_disk_1, label="thin disk color 400 / 900 nm")
ax[1].plot(tax, color_disk_2, label="slim disk color 400 / 900 nm")


ax[0].set_ylabel("flux [arb.]")
for axis in ax:
    axis.set_xlabel("time [years]")
    axis.legend()

plt.show()
