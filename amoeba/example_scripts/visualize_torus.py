import pytest
import numpy as np
from amoeba.Classes.blr_streamline import Streamline
from amoeba.Classes.torus import Torus
from amoeba.Classes.flux_projection import FluxProjection
import astropy.units as u
import numpy.testing as npt
import matplotlib.pyplot as plt




smbh_mass_exp = 7.2
launch_radius = 600  # Rg
launch_theta = 50  # degrees
max_height = 1000
inclination = 40
characteristic_distance = 100
asymptotic_poloidal_velocity = 0
poloidal_launch_velocity = 0
height_step = 10
redshift_source = 1.1
power_law_density_dependence = 0

test_torus_streamline = Streamline(
    launch_radius,
    launch_theta,
    max_height,
    characteristic_distance,
    asymptotic_poloidal_velocity,
    poloidal_launch_velocity=poloidal_launch_velocity,
    height_step=height_step,
)

my_torus = Torus(
    smbh_mass_exp,
    max_height,
    redshift_source,
    height_step=height_step,
    power_law_density_dependence=power_law_density_dependence,
)

my_torus.add_streamline_bounded_region(test_torus_streamline)

projection = my_torus.project_density_to_source_plane(
    inclination
)

xax = np.linspace(-my_torus.max_radius, my_torus.max_radius, np.size(projection, 0))

X, Y = np.meshgrid(xax, xax)

fig, ax = plt.subplots()
contours = ax.contourf(X, Y, projection, 50, cmap='plasma')
cbar = plt.colorbar(contours, ax=ax, label='column density [arb.]')
ax.set_xlabel("X [Rg]")
ax.set_ylabel("Y [Rg]")
ax.set_aspect(1)
plt.show()









        
