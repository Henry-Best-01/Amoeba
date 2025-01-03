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
inclination_angle = 30.0
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


transfer_function_1 = Disk.construct_accretion_disk_transfer_function(wavelength_1)
transfer_function_2 = Disk.construct_accretion_disk_transfer_function(wavelength_2)


taus = np.linspace(0, len(transfer_function_1) - 1, len(transfer_function_1))

mean_1 = round(np.sum(taus * transfer_function_1), 1)
mean_2 = round(np.sum(taus * transfer_function_2), 1)

max_val = np.max([np.max(transfer_function_1), np.max(transfer_function_2)])


fig, ax = plt.subplots()
ax.plot(
    transfer_function_1, label=r"$\lambda _{\rm{obs}}$ = " + str(wavelength_1) + " nm"
)
ax.plot(
    transfer_function_2, label=r"$\lambda _{\rm{obs}}$ = " + str(wavelength_2) + " nm"
)

ax.set_prop_cycle(None)
ax.plot([mean_1, mean_1], [0, max_val], "--", label=str(mean_1) + r"$r_{\rm{g}}$")
ax.plot([mean_2, mean_2], [0, max_val], "--", label=str(mean_2) + r"$r_{\rm{g}}$")


ax.set_xlabel(r"$ \bar{\tau}_{\rm{x}} [r_{\rm{g}}]$")
ax.set_ylabel(r"$ \Psi_{\lambda}$ [arb.]")

ax.legend()


plt.show()
