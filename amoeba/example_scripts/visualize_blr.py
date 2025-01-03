import numpy as np
import matplotlib.pyplot as plt
from amoeba.Classes.blr import BroadLineRegion
from amoeba.Classes.blr_streamline import Streamline
from amoeba.Classes.accretion_disk import AccretionDisk
from amoeba.Util.util import (
    create_maps,
    determine_emission_line_velocities,
)
from astropy import units as u
from astropy import constants as const


# agn params

smbh_mass_exp = 8.0
redshift_source = 0.0
inclination_angle = 70  # deg
smbh_spin = 0.6
name = "test agn"
max_disk_radius = 1000  # rg
accretion_disk_resolution = 1000  # pixels / radial length

# blr params

inner_launch_radius = 100  # rg
inner_launch_angle = 70  # deg
outer_launch_radius = 200  # rg
outer_launch_angle = 70  # deg

maximum_blr_height = 1000  # rg
characteristic_distance = 500  # in rg, for outflowing velocity model
asymptotic_poloidal_velocity = 0.5  # units v / c
height_step = 10  # rg, defines thickness of each blr slab
radial_step = 10  # rg, defines resolution of each slab

blr_rest_frame_emission = 500  # nm


# projection params
filter_1 = [300, 450]
filter_2 = [400, 550]
filter_3 = [500, 700]


# create accretion disk and broad line region

agn_params_dict = create_maps(
    smbh_mass_exp,
    redshift_source,
    max_disk_radius,
    inclination_angle,
    accretion_disk_resolution,
    spin=smbh_spin,
    name=name,
)

accretion_disk = AccretionDisk(**agn_params_dict)


inner_streamline = Streamline(
    inner_launch_radius,
    inner_launch_angle,
    maximum_blr_height,
    characteristic_distance,
    asymptotic_poloidal_velocity,
    height_step=height_step,
)

outer_streamline = Streamline(
    outer_launch_radius,
    outer_launch_angle,
    maximum_blr_height,
    characteristic_distance,
    asymptotic_poloidal_velocity,
    height_step=height_step,
)


broad_line_region = BroadLineRegion(
    smbh_mass_exp,
    maximum_blr_height,
    blr_rest_frame_emission,
    redshift_source,
    radial_step=radial_step,
    height_step=height_step,
)

broad_line_region.add_streamline_bounded_region(
    inner_streamline,
    outer_streamline,
)


# project into the source plane

eff_wavelength_1 = np.mean(filter_1)
eff_wavelength_2 = np.mean(filter_2)
eff_wavelength_3 = np.mean(filter_3)


disk_projection_1 = accretion_disk.calculate_surface_intensity_map(eff_wavelength_1)
disk_projection_2 = accretion_disk.calculate_surface_intensity_map(eff_wavelength_2)
disk_projection_3 = accretion_disk.calculate_surface_intensity_map(eff_wavelength_3)


v_range_1 = determine_emission_line_velocities(
    blr_rest_frame_emission, filter_1[0], filter_1[1], redshift_source
)
v_range_2 = determine_emission_line_velocities(
    blr_rest_frame_emission, filter_2[0], filter_2[1], redshift_source
)
v_range_3 = determine_emission_line_velocities(
    blr_rest_frame_emission, filter_3[0], filter_3[1], redshift_source
)


blr_projection_1 = broad_line_region.project_blr_intensity_over_velocity_range(
    inclination_angle,
    v_range_1,
)

blr_projection_2 = broad_line_region.project_blr_intensity_over_velocity_range(
    inclination_angle,
    v_range_2,
)

blr_projection_3 = broad_line_region.project_blr_intensity_over_velocity_range(
    inclination_angle,
    v_range_3,
)


# plot

xax_disk = np.linspace(
    -disk_projection_1.r_out_in_gravitational_radii,
    disk_projection_1.r_out_in_gravitational_radii,
    np.size(disk_projection_1.flux_array, 0),
)

X, Y = np.meshgrid(xax_disk, xax_disk)

xax_blr = np.linspace(
    -blr_projection_1.r_out_in_gravitational_radii,
    blr_projection_1.r_out_in_gravitational_radii,
    np.size(blr_projection_1.flux_array, 0),
)

Xb, Yb = np.meshgrid(xax_blr, xax_blr)


fig, ax = plt.subplots(3, 2, sharex=True, sharey=True)


conts1 = ax[0, 0].contourf(X, Y, disk_projection_1.flux_array)
conts2 = ax[1, 0].contourf(X, Y, disk_projection_2.flux_array)
conts3 = ax[2, 0].contourf(X, Y, disk_projection_3.flux_array)

conts4 = ax[0, 1].contourf(Xb, Yb, blr_projection_1.flux_array)
conts5 = ax[1, 1].contourf(Xb, Yb, blr_projection_2.flux_array)
conts6 = ax[2, 1].contourf(Xb, Yb, blr_projection_3.flux_array)

cbar1 = plt.colorbar(conts1, ax=ax[0, 0])
cbar2 = plt.colorbar(conts2, ax=ax[1, 0])
cbar3 = plt.colorbar(conts3, ax=ax[2, 0])
cbar4 = plt.colorbar(conts4, ax=ax[0, 1])
cbar5 = plt.colorbar(conts5, ax=ax[1, 1])
cbar6 = plt.colorbar(conts6, ax=ax[2, 1])

ax[0, 0].set_xlim(-500, 500)
ax[0, 0].set_ylim(-300, 300)

ax[0, 0].set_title("$\lambda$ = " + str(filter_1[0]) + " - " + str(filter_1[1]) + " nm")
ax[1, 0].set_title("$\lambda$ = " + str(filter_2[0]) + " - " + str(filter_2[1]) + " nm")
ax[2, 0].set_title("$\lambda$ = " + str(filter_3[0]) + " - " + str(filter_3[1]) + " nm")
ax[0, 1].set_title("$\lambda$ = " + str(filter_1[0]) + " - " + str(filter_1[1]) + " nm")
ax[1, 1].set_title("$\lambda$ = " + str(filter_2[0]) + " - " + str(filter_2[1]) + " nm")
ax[2, 1].set_title("$\lambda$ = " + str(filter_3[0]) + " - " + str(filter_3[1]) + " nm")


for axis_row in ax:
    for axis in axis_row:
        axis.set_aspect(1)
        axis.set_xlabel(r"X [$r_{\rm{g}}$]")
        axis.set_ylabel(r"Y [$r_{\rm{g}}$]")

fig.set_figheight(10)
fig.set_figwidth(8)

plt.show()
