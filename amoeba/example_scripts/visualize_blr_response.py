import numpy as np
from amoeba.Classes.blr import BroadLineRegion
from amoeba.Classes.blr_streamline import Streamline
import matplotlib.pyplot as plt


inc_ang = 45  # deg
redshift = 1
zmax = 1000  # R_g
mexp = 8.0
sl_1_rlaunch = 100  # R_g
sl_2_rlaunch = 400
sl_1_theta = 10  # deg
sl_2_theta = 30
sl_1_char_dist = 200  # R_g
sl_2_char_dist = 500
sl_1_asy_vel = 0.10  # v/c
sl_2_asy_vel = 0.05
zres = 5
rres = 5
filter1min = 950
filter1max = 1080
filter1mp = 1020
filter2min = 818
filter2max = 922
filter2mp = 900
emit_line = 486


# set up streamlines

inner_streamline = Streamline(
    sl_1_rlaunch,
    sl_1_theta,
    zmax,
    sl_1_char_dist,
    sl_1_asy_vel,
)

outer_streamline = Streamline(
    sl_2_rlaunch,
    sl_2_theta,
    zmax,
    sl_2_char_dist,
    sl_2_asy_vel,
)

# set up blr
blr = BroadLineRegion(
    mexp,
    zmax,
    emit_line,
    redshift,
)

blr.add_streamline_bounded_region(
    inner_streamline,
    outer_streamline,
)


# project blr

filter_1_projection = blr.project_blr_intensity_over_velocity_range(
    inc_ang, observed_wavelength_range_in_nm=[filter1min, filter1max]
)

filter_2_projection = blr.project_blr_intensity_over_velocity_range(
    inc_ang, observed_wavelength_range_in_nm=[filter2min, filter2max]
)

# get blr transfer function

filter_1_tf = blr.calculate_blr_emission_line_transfer_function(
    inc_ang, observed_wavelength_range_in_nm=[filter1min, filter1max]
)

filter_2_tf = blr.calculate_blr_emission_line_transfer_function(
    inc_ang, observed_wavelength_range_in_nm=[filter2min, filter2max]
)

tau_ax_1 = np.linspace(0, len(filter_1_tf) - 1, len(filter_1_tf))
tau_ax_2 = np.linspace(0, len(filter_2_tf) - 1, len(filter_2_tf))

mean_blr_1 = np.sum(filter_1_tf * tau_ax_1) / np.sum(filter_1_tf)
mean_blr_2 = np.sum(filter_2_tf * tau_ax_2) / np.sum(filter_2_tf)


# plot
xax = np.linspace(
    -filter_1_projection.r_out_in_gravitational_radii,
    filter_1_projection.r_out_in_gravitational_radii,
    np.size(filter_1_projection.flux_array, 0),
)
X, Y = np.meshgrid(xax, xax)


fig, ax = plt.subplots(2, 2)

conts1 = ax[0, 0].contourf(X, Y, filter_1_projection.flux_array, cmap="plasma")
ax[0, 0].set_title(r"$y$ projection")
ax[0, 0].scatter([0], [0], marker="o", color="white")
ax[0, 0].set_xlabel(r"X [$r_{\rm{g}}$]")
ax[0, 0].set_ylabel(r"Y [$r_{\rm{g}}$]")
ax[0, 0].set_aspect(1)
cbar1 = plt.colorbar(conts1, ax=ax[0, 0], label="response [arb.]")

conts2 = ax[1, 0].contourf(X, Y, filter_2_projection.flux_array, cmap="plasma")
ax[1, 0].set_title(r"$z$ projection")
ax[1, 0].scatter([0], [0], marker="o", color="white")
ax[1, 0].set_xlabel(r"X [$r_{\rm{g}}$]")
ax[1, 0].set_ylabel(r"Y [$r_{\rm{g}}$]")
ax[1, 0].set_aspect(1)
cbar2 = plt.colorbar(conts2, ax=ax[1, 0], label="response [arb.]")

ax[0, 1].plot(filter_1_tf / np.max(filter_1_tf))
ax[0, 1].set_title(r"response in $y$ filter")
ax[0, 1].set_xlabel(r"$\tau [r_{\rm{g}}]$")
ax[0, 1].set_ylabel(r"$\Psi_{\rm{BLR}}$ [arb.]")

ax[0, 1].set_prop_cycle(None)
ax[0, 1].plot([mean_blr_1, mean_blr_1], [0, 1], "--")


ax[1, 1].plot(filter_2_tf / np.max(filter_2_tf))
ax[1, 1].set_title(r"response in $z$ filter")
ax[1, 1].set_xlabel(r"$\tau [r_{\rm{g}}]$")
ax[1, 1].set_ylabel(r"$\Psi_{\rm{BLR}}$ [arb.]")
ax[1, 1].set_prop_cycle(None)
ax[1, 1].plot([mean_blr_2, mean_blr_2], [0, 1], "--")

plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.show()
