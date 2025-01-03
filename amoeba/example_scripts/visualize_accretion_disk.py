import numpy as np
import numpy.testing as npt
from amoeba.Classes.accretion_disk import AccretionDisk
from amoeba.Util.util import (
    create_maps,
)
from astropy import units as u
import matplotlib.pyplot as plt


smbh_mass_exp = 8.0
redshift_source = 2.0
inclination_angle = 80.0
corona_height = 10
number_grav_radii = 500
resolution = 1000
spin = 0.0
wavelength_1 = 400
wavelength_2 = 700


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
