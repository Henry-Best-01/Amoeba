import numpy as np
import matplotlib.pyplot as plt

plt.style.use("/Users/henrybest/PythonStuff/Code/plot_style_rainbow_6.txt")
from amoeba.Classes.accretion_disk import AccretionDisk
from amoeba.Util.util import (
    planck_law,
    calculate_gravitational_radius,
    accretion_disk_temperature,
)
import time

t_start = time.time()


def calculate_half_light_radius(
    rest_wavelength, radial_temp_dependence, mass_exp=8.0, r_min=6, r_max=100000
):

    mass = 10**mass_exp
    grav_rad = calculate_gravitational_radius(mass)
    radii_rg = np.linspace(r_min, r_max, r_max - r_min + 1)
    radii_meters = radii_rg * grav_rad
    efficiency = 1 - (1 - 2 / (3 * r_min)) ** 0.5

    temps = accretion_disk_temperature(
        radii_meters,
        r_min * grav_rad,
        mass,
        0.1,
        beta=radial_temp_dependence,
        efficiency=efficiency,
        generic_beta=True,
    )

    flux_profile = planck_law(temps, rest_wavelength)

    weighted_fluxes = flux_profile * radii_rg

    cumulative_flux = np.cumsum(weighted_fluxes)

    half_flux = cumulative_flux[-1] / 2

    half_light_radius_arg = np.argmin((weighted_fluxes - half_flux) ** 2)
    if half_light_radius_arg <= r_min + 1:
        print("too small! half light radius = r_min")
    if half_light_radius_arg >= r_max - 1:
        print("half light radius = r_max")

    return half_light_radius_arg, temps


# np.log10(half_light_radius_arg * grav_rad)

wavelengths = [400, 500, 600, 700, 800, 1000]
r_min = 6
r_max = 100000
asympt_power = 0.75
mass_exp = 8.0
grav_rad = calculate_gravitational_radius(mass_exp)
radii = np.linspace(r_min, r_max, r_max - r_min + 1)
fig, ax = plt.subplots()
half_lights = []

for jj, lam in enumerate(wavelengths):
    half_light, temps = calculate_half_light_radius(
        lam, asympt_power, mass_exp=mass_exp, r_max=r_max
    )
    half_lights.append(half_light)
    print(half_light)
    ax.plot([half_lights[jj], half_lights[jj]], [0, temps[int(half_lights[jj])]], "--")
ax.loglog(radii, temps, color="black")
ax.set_aspect(1)
print(time.time() - t_start)
plt.show()
