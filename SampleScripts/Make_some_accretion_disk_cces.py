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


smbh_mass_exp = 7.3
smbh_mass_exp_2 = 8.5
redshift_source = 2.0
redshift_lens = 1.0
inclination_angle = 0.0
corona_height = 10
number_grav_radii = 1000
resolution = 2000
spin = 0.0
wavelength_1 = 400
wavelength_2 = 900
x_init = 5000
y_init = 5000


transverse_vel = 400  # km/s
duration = 20  # years


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
    corona_height=corona_height,
)
Disk = AccretionDisk(**accretion_disk_data_1)


disk_emission_1 = Disk.calculate_surface_intensity_map(wavelength_1)
disk_emission_2 = Disk.calculate_surface_intensity_map(wavelength_2)


accretion_disk_data_2 = create_maps(
    smbh_mass_exp_2,
    redshift_source,
    number_grav_radii,
    inclination_angle,
    resolution,
    spin=spin,
    corona_height=corona_height,
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
    x_start_position=x_init,
    y_start_position=y_init,
)
lc_2 = convolution_2.pull_light_curve(
    transverse_vel,
    duration,
    random_seed=random_seed,
    x_start_position=x_init,
    y_start_position=y_init,
)
lc_3 = magnification_map_initial.pull_light_curve(
    transverse_vel,
    duration,
    random_seed=random_seed,
    x_start_position=x_init,
    y_start_position=y_init,
)


lc_1_2 = convolution_1_2.pull_light_curve(
    transverse_vel,
    duration,
    random_seed=random_seed,
    x_start_position=x_init,
    y_start_position=y_init,
)
lc_2_2 = convolution_2_2.pull_light_curve(
    transverse_vel,
    duration,
    random_seed=random_seed,
    x_start_position=x_init,
    y_start_position=y_init,
)

color_1 = np.log10(np.asarray(lc_1)) - np.log10(np.asarray(lc_2))
# color_1 /= np.max(color_1)

color_2 = np.log10(np.asarray(lc_1_2)) - np.log10(np.asarray(lc_2_2))

# color_2 /= np.max(color_2)

fig, ax = plt.subplots()
tax = np.linspace(0, duration, len(lc_1))


ax.set_xlabel("time [years]")
ax.set_ylabel("color 400 nm - 700 nm [arb.]")


ax.plot(
    tax, color_1, label=r"$M_{\rm{bh}} = 10^{" + str(smbh_mass_exp) + r"} M_{\odot}$"
)

ax.plot(
    tax, color_2, label=r"$M_{\rm{bh}} = 10^{" + str(smbh_mass_exp_2) + r"} M_{\odot}$"
)

ax.legend()


plt.show()


fig, ax = plt.subplots()
tax = np.linspace(0, duration, len(lc_1))


ax.set_xlabel("time [years]")
ax.set_ylabel("color 400 nm - 700 nm [arb.]")


ax.plot(
    tax,
    np.asarray(lc_1) / np.max(lc_1),
)

ax.plot(
    tax,
    np.asarray(lc_1_2) / np.max(lc_1_2),
)

ax.plot(
    tax,
    np.asarray(lc_3) / np.max(lc_3),
)


plt.show()
"""
fig, ax = plt.subplots(2, 2)


xax = np.linspace(
    0,
    np.size(convolution_1.magnification_array, 0) - 1,
    np.size(convolution_1.magnification_array, 0)
)
convo_1_x, convo_1_y = np.meshgrid(xax, xax)

xax = np.linspace(
    0,
    np.size(convolution_1_2.magnification_array, 0) - 1,
    np.size(convolution_1_2.magnification_array, 0)
)
convo_2_x, convo_2_y = np.meshgrid(xax, xax)


ax[0, 0].contourf(convo_1_x, convo_1_y, convolution_1.magnification_array)
ax[0, 1].contourf(convo_1_x, convo_1_y, convolution_2.magnification_array)
ax[1, 0].contourf(convo_2_x, convo_2_y, convolution_1_2.magnification_array)
ax[1, 1].contourf(convo_2_x, convo_2_y, convolution_2_2.magnification_array)


        

plt.show()

"""

"""

xax = np.linspace(
    -number_grav_radii, number_grav_radii, np.size(disk_emission_1.flux_array, 0)
)

X, Y = np.meshgrid(xax, xax)


fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)

conts = ax[0].contourf(X, Y, disk_emission_1.flux_array, cmap="plasma")
cbar = plt.colorbar(conts, ax=ax[0])
ax[0].set_title(
    r"$\lambda _{\rm{obs}}$ = "
    + str(disk_emission_1.observer_frame_wavelength_in_nm)
    + "\n"
    + r"$\lambda _{\rm{rest}}$ = "
    + str(disk_emission_1.rest_frame_wavelength_in_nm)
)

conts = ax[1].contourf(X, Y, disk_emission_2.flux_array, cmap="plasma")
cbar = plt.colorbar(conts, ax=ax[1])
ax[1].set_title(
    r"$\lambda _{\rm{obs}}$ = "
    + str(disk_emission_2.observer_frame_wavelength_in_nm)
    + "\n"
    + r"$\lambda _{\rm{rest}}$ = "
    + str(disk_emission_2.rest_frame_wavelength_in_nm)
)


for axis in ax:
    axis.scatter([0], [0], c="black", s=40)
    axis.set_aspect(1)
    axis.set_xlabel(r"X [$r_{\rm{g}}$]")
    axis.set_ylabel(r"Y [$r_{\rm{g}}$]")


ax[0].set_xlim(-1, 100)
ax[0].set_ylim(-50, 50)


plt.show()
"""
