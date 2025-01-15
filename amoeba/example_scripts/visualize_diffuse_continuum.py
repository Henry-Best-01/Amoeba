import numpy as np
from amoeba.Classes.diffuse_continuum import DiffuseContinuum
from amoeba.Util.util import convert_cartesian_to_polar
import matplotlib.pyplot as plt
import time

tstart = time.time()


inclination_angle = 0

x_vals = np.linspace(-2000, 2000, 1000)
X, Y = np.meshgrid(x_vals, x_vals / np.cos(inclination_angle * np.pi / 180))
R, Phi = convert_cartesian_to_polar(X, Y)

smbh_mass_exp = 8
redshift_source = 0
radii_array = R
phi_array = Phi
cloud_density_radial_dependence = 0.0
cloud_density_array = None
Om0 = 0.3
H0 = 70
r_in_in_gravitational_radii = 500
r_out_in_gravitational_radii = 2000
name = "my diffuse continuum"

diffuse_continuum_spectra_wavelengths = [100, 140, 150, 160, 360, 370, 800, 810, 1000]
diffuse_continuum_spectra_emissivities = [
    0.05,
    0.07,
    0.19,
    0.08,
    0.45,
    0.15,
    0.37,
    0.2,
    0.25,
]
constant_a = 0.5

my_kwargs = {
    "smbh_mass_exp": smbh_mass_exp,
    "redshift_source": redshift_source,
    "inclination_angle": inclination_angle,
    "radii_array": radii_array,
    "phi_array": phi_array,
    "cloud_density_radial_dependence": cloud_density_radial_dependence,
    "cloud_density_array": cloud_density_array,
    "Om0": Om0,
    "H0": H0,
    "r_in_in_gravitational_radii": r_in_in_gravitational_radii,
    "r_out_in_gravitational_radii": r_out_in_gravitational_radii,
    "emissivity_etas": diffuse_continuum_spectra_emissivities,
    "rest_frame_wavelengths": diffuse_continuum_spectra_wavelengths,
    "responsivity_constant": constant_a,
    "name": name,
}


my_continuum = DiffuseContinuum(**my_kwargs)


test_wavelengths = np.linspace(100, 1000, 500)

dc_spectrum = []
dc_flux_contribution = []
total_increased_lags = []

for jj in test_wavelengths:
    dc_spectrum.append(my_continuum.interpolate_spectrum_to_wavelength(jj))
    total_increased_lags.append(my_continuum.get_diffuse_continuum_lag_contribution(jj))

dc_spectrum = np.asarray(dc_spectrum)
total_increased_lags = np.asarray(total_increased_lags)

mean_lag_dc = my_continuum.get_diffuse_continuum_mean_lag(200)

print("diffuse continuum mean lag:", mean_lag_dc, r"$R_{\rm{g}}$")


dc_projection = my_continuum.get_diffuse_continuum_emission(200)

X, Y = dc_projection.get_plotting_axes()


plot_kwargs = {
    "height_ratios": [4, 4, 1, 10],
}

fig, ax = plt.subplots(4, gridspec_kw=plot_kwargs)
ax[0].plot(test_wavelengths, dc_spectrum)
ax[1].plot(test_wavelengths, total_increased_lags)
ax[1].sharex(ax[0])
ax[1].set_xlabel(r"$\lambda _{\rm{rest}}$ [nm]")
ax[0].set_ylabel(r"$\eta$")
ax[1].set_ylabel(r"$ <\bar{\tau}_{\rm{DC - inci}}> [R_{\rm{g}}]$")

ax[3].plot(
    dc_spectrum,
    total_increased_lags / mean_lag_dc,
    label=r"$\alpha$ = " + str(constant_a),
)


const_a = 0.1
my_continuum.set_responsivity_constant(const_a)
dc_spectrum = []
dc_flux_contribution = []
total_increased_lags = []

for jj in test_wavelengths:
    dc_spectrum.append(my_continuum.interpolate_spectrum_to_wavelength(jj))
    total_increased_lags.append(my_continuum.get_diffuse_continuum_lag_contribution(jj))

dc_spectrum = np.asarray(dc_spectrum)
total_increased_lags = np.asarray(total_increased_lags)

mean_lag_dc = my_continuum.get_diffuse_continuum_mean_lag(200)

ax[3].plot(
    dc_spectrum, total_increased_lags / mean_lag_dc, label=r"$\alpha$ = " + str(const_a)
)

const_a = 0.3
my_continuum.set_responsivity_constant(const_a)
dc_spectrum = []
dc_flux_contribution = []
total_increased_lags = []

for jj in test_wavelengths:
    dc_spectrum.append(my_continuum.interpolate_spectrum_to_wavelength(jj))
    total_increased_lags.append(my_continuum.get_diffuse_continuum_lag_contribution(jj))

dc_spectrum = np.asarray(dc_spectrum)
total_increased_lags = np.asarray(total_increased_lags)

mean_lag_dc = my_continuum.get_diffuse_continuum_mean_lag(200)

ax[3].plot(
    dc_spectrum, total_increased_lags / mean_lag_dc, label=r"$\alpha$ = " + str(const_a)
)

const_a = 0.7
my_continuum.set_responsivity_constant(const_a)
dc_spectrum = []
dc_flux_contribution = []
total_increased_lags = []

for jj in test_wavelengths:
    dc_spectrum.append(my_continuum.interpolate_spectrum_to_wavelength(jj))
    total_increased_lags.append(my_continuum.get_diffuse_continuum_lag_contribution(jj))

dc_spectrum = np.asarray(dc_spectrum)
total_increased_lags = np.asarray(total_increased_lags)

mean_lag_dc = my_continuum.get_diffuse_continuum_mean_lag(200)

ax[3].plot(
    dc_spectrum, total_increased_lags / mean_lag_dc, label=r"$\alpha$ = " + str(const_a)
)


ax[3].plot([0, 1], [0, 1], "--", linewidth=0.5, color="black")

ax[3].set_xlabel("frac. flux")
ax[3].set_ylabel("frac lag increase")
ax[3].set_aspect(1)
ax[3].legend()

ax[2].set_axis_off()
plt.subplots_adjust(hspace=0.2)
fig.set_figheight(8)


fig2, ax2 = plt.subplots()
conts = ax2.contourf(X, Y, dc_projection.flux_array, 20, cmap="plasma")
cbar = plt.colorbar(conts, ax=ax2)
ax2.set_xlabel(r"$X [R_{\rm{g}}]$")
ax2.set_ylabel(r"$Y [R_{\rm{g}}]$")

ax2.set_title("total fraction of continuum: " + str(round(dc_projection.total_flux, 3)))


print("total elapsed time:", round(time.time() - tstart, 1), "s")


plt.show()
