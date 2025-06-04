import numpy as np
import matplotlib.pyplot as plt
from amoeba.Util.util import calculate_geometric_disk_factor, convert_cartesian_to_polar

res = 1000
max_r = 100

r_offset = 00
phi_offset = 70


radii = np.linspace(-max_r, max_r, res)
corona_height = 50

line_index = int((2 * res) / 5)


X, Y = np.meshgrid(radii, radii)
R, Phi = convert_cartesian_to_polar(X, Y)

# make some non-intuitive geometry with phi dependence.
# note that phi dependence should be suppressed but not removed


# test one, very complex wavy disk.
R_mask_1 = R < 50
R_mask_2 = R >= 50

heights = (
    np.cos(R * R_mask_1 * np.pi / (10))
    + 10 * R / max_r
    + np.sin(R * R_mask_2 * np.pi / (4))
    + Phi % (2 * np.pi / 3) * R_mask_1
    - Phi % (2 * np.pi / 5) * R_mask_2
)

# test two, height = radius from black hole
# heights = R.copy()

# test three, flat disk
# heights = np.zeros(np.shape(R))

# temp and smbh_mass_exp arguments are not important here
test_val = calculate_geometric_disk_factor(
    R,
    R,
    Phi,
    1,
    corona_height,
    height_array=heights,
    axis_offset_in_gravitational_radii=r_offset,
    angle_offset_in_degrees=phi_offset,
)

fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots(2, sharex="all")

hconts = ax2.contourf(X, Y, heights, 51)
testconts = ax3.contourf(X, Y, test_val, 51)


ax4[0].plot(radii, heights[line_index], label="disk surface")
ax4[1].plot(
    radii,
    test_val[line_index],  # / np.max(test_val[line_index]),
    label=r"f$_{\rm{geo}}$ [norm.]",
)


ax4[0].scatter(
    [r_offset * np.cos(phi_offset)],
    [corona_height],
    marker="*",
    label="corona height (+ "
    + str(round(radii[line_index] - r_offset * np.sin(phi_offset), 0))
    + str(r" r$_{\rm{g}}$)"),
)

ax2.plot(
    [-max_r, max_r],
    [radii[line_index], radii[line_index]],
    "-.",
    color="white",
    alpha=0.7,
)
ax3.plot(
    [-max_r, max_r],
    [radii[line_index], radii[line_index]],
    "-.",
    color="white",
    alpha=0.7,
)

plt.colorbar(hconts, ax=ax2, label=r"Height [r$_{\rm{g}}$]")
plt.colorbar(testconts, ax=ax3, label=r"f$_{\rm{geo}}$")


ax2.set_aspect(1)
ax3.set_aspect(1)
ax4[0].set_aspect(1)
fig4.set_figheight(3)

ax2.set_xlabel(r"x [r$_{\rm{g}}$]")
ax2.set_ylabel(r"y [r$_{\rm{g}}$]")
ax3.set_xlabel(r"x [r$_{\rm{g}}$]")
ax3.set_ylabel(r"y [r$_{\rm{g}}$]")
ax4[-1].set_xlabel(r"r [r$_{\rm{g}}$]")
ax4[0].set_ylabel(r"z [r$_{\rm{g}}$]")
ax4[1].set_ylabel(r"f$_{\rm{geo}}$ [arb.]")

ax4[0].legend(loc=2, framealpha=0.3)


plt.show()
