import numpy as np
import numpy.testing as npt
from amoeba.Classes.magnification_map import MagnificationMap, ConvolvedMap
from amoeba.Classes.flux_projection import FluxProjection
from amoeba.Classes.accretion_disk import AccretionDisk
from amoeba.Util.util import create_maps
import matplotlib.pyplot as plt


smbh_mass_exp = 8.0
redshift_source = 2.0
inclination_angle = 40.0
corona_height = 10
number_grav_radii = 1000
resolution = 1000
spin = 0.0
wavelength_1 = 1000

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
dt_dlx_array = Disk.calculate_dt_dlx_array()
db_dt_array = Disk.calculate_db_dt_array(wavelength_1)
response_map_manual = dt_dlx_array * db_dt_array


response_map_auto, time_lags = Disk.construct_accretion_disk_transfer_function(
    wavelength_1, return_response_array_and_lags=True
)


redshift_source = 2.0
redshift_lens = 1.0
convergence = 0.3
shear = 0.1
name = "silly test array"
total_microlens_einstein_radii = 1


big_magnification_ones = np.ones((1000, 1000))

identity_magnification_array = MagnificationMap(
    redshift_source,
    redshift_lens,
    big_magnification_ones,
    convergence,
    shear,
    total_microlens_einstein_radii=1,
    name="identity",
)


id_microlensed_response, descaled_timelags = (
    identity_magnification_array.calculate_microlensed_transfer_function(
        Disk, wavelength_1, return_descaled_response_array_and_lags=True
    )
)

contour_levels = np.linspace(100, 1000, 5)


x_ax_numbers = np.linspace(
    -number_grav_radii, number_grav_radii, np.size(response_map_manual, 0)
)
X, Y = np.meshgrid(x_ax_numbers, x_ax_numbers)


fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)

ax[0].set_title("manual")
conts1 = ax[0].contourf(
    X, Y, response_map_manual / np.max(response_map_manual), cmap="plasma"
)
cbar1 = plt.colorbar(conts1, ax=ax[0], label="relative db/dL_{x}")

ax[1].set_title("auto microlensed")
conts2 = ax[1].contourf(
    X, Y, id_microlensed_response / np.max(id_microlensed_response), cmap="plasma"
)
cbar2 = plt.colorbar(conts2, ax=ax[1], label="relative db/dL_{x}")

ax[2].set_title("automatic")
conts1 = ax[2].contourf(
    X, Y, response_map_auto / np.max(response_map_auto), cmap="plasma"
)
cbar1 = plt.colorbar(conts1, ax=ax[2], label="relative db/dL_{x}")

for axis in ax:

    tcontours = axis.contour(
        X, Y, time_lags, colors="white", alpha=0.7, levels=contour_levels
    )
    plt.clabel(tcontours, inline=1, fontsize=16)

    axis.set_aspect(1)
    axis.set_xlabel(r"X, $R_{\rm{g}}$")
    axis.set_ylabel(r"Y, $R_{\rm{g}}$")
fig.set_figwidth(14)

plt.show()
