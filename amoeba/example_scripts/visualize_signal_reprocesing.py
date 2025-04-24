import numpy as np
from amoeba.Classes.accretion_disk import AccretionDisk
from amoeba.Util.util import (
    calculate_gravitational_radius,
    generate_drw_signal,
    generate_signal_from_psd,
    create_maps,
    convolve_signal_with_transfer_function,
)
import matplotlib.pyplot as plt
from scipy.signal import welch


# define params of disk

wavelength = 400
mass_exponent = 9.663
redshift_source = 0.11
number_gravitational_radii = 1000
inclination_angle = 33
resolution = number_gravitational_radii
spin = 0.44
eddington_ratio = 0.07
temp_beta = 0.01
corona_height = 13
visc_temp_prof = "NT"
# "NT" is Novikov-Thorne disk. Use "SS" for traditional thin disc


disk_dictionary = create_maps(
    mass_exponent,
    redshift_source=redshift_source,
    number_grav_radii=number_gravitational_radii,
    inclination_angle=inclination_angle,
    resolution=resolution,
    spin=spin,
    eddington_ratio=eddington_ratio,
    temp_beta=temp_beta,
    corona_height=corona_height,
    visc_temp_prof=visc_temp_prof,
    name="sample accretion disk",
)


# define params of driving signal

time_scale_breakpoint = 200  # days
length_light_curve = 10000  # days

sf_infinity = 100
tau_drw = time_scale_breakpoint

random_seed = 9001

bpl_low_freq_dependence = 1
bpl_high_freq_dependence = 3


# generate power spectra and signals

signal_drw = generate_drw_signal(
    length_light_curve, 1, sf_infinity, tau_drw, random_seed=random_seed
)


frequency_axis = np.linspace(1 / (2 * length_light_curve), 1 / 2, length_light_curve)
## see Edelson + Nandra 99 (https://ui.adsabs.harvard.edu/abs/1999ApJ...514..682E/abstract)
my_bpl_psd = frequency_axis ** (-bpl_low_freq_dependence) * (
    1
    + (frequency_axis * time_scale_breakpoint)
    ** (bpl_high_freq_dependence - bpl_low_freq_dependence)
) ** (-1)

# the DRW is a specific instance of BPL, with low dependence = 0 and high dependence = 2
my_drw_psd = frequency_axis ** (0) * (
    1 + (frequency_axis * time_scale_breakpoint) ** (2)
) ** (-1)


fig, ax = plt.subplots()
ax.loglog(frequency_axis, my_bpl_psd, label="bending power law")
ax.loglog(frequency_axis, my_drw_psd, label="damped random walk")


ax.plot(
    [1 / time_scale_breakpoint, 1 / time_scale_breakpoint],
    [min(my_bpl_psd), max(my_bpl_psd)],
    "--",
    color="black",
    alpha=0.4,
    label="breakpoint frequency",
)
ax.set_xlabel(r"$\nu$ [day$^{-1}$]")
ax.set_ylabel("Power [arb.]")
fig.set_figheight(3)

# regenerate the power spectra of signals produced
sample_frequencies, regenerated_drw_psd = welch(
    signal_drw, nperseg=min(10 * time_scale_breakpoint, int(length_light_curve / 10))
)
ax.plot(sample_frequencies, regenerated_drw_psd, label="regenerated drw")


time_bpl, bpl_signal = generate_signal_from_psd(
    length_light_curve, my_bpl_psd, frequency_axis, random_seed=random_seed
)

sample_frequencies, regenerated_bpl_psd = welch(
    bpl_signal, nperseg=min(10 * time_scale_breakpoint, int(length_light_curve / 10))
)
ax.plot(sample_frequencies, regenerated_bpl_psd, label="regenerated bpl")

ax.legend(loc=3)
ax.set_xlim(10**-3, 0.5)

time_drw, drw_signal = generate_signal_from_psd(
    length_light_curve, my_drw_psd, frequency_axis, random_seed=random_seed
)


fig2, ax2 = plt.subplots()
time_axis = np.linspace(0, (length_light_curve) - 1, (length_light_curve))
ax2.plot(time_bpl, bpl_signal, alpha=0.7, label="broken power law")
ax2.plot(time_drw, drw_signal, alpha=0.7, label="damped random walk")
ax2.plot(time_axis, signal_drw[: len(time_axis)], alpha=0.7, label="convenience drw")

ax2.legend()

ax2.set_xlabel("time in source frame [days]")
ax2.set_ylabel("flux [arb.]")
fig2.set_figheight(3)


# make disk and transfer function

Disk = AccretionDisk(**disk_dictionary)

my_tf = Disk.construct_accretion_disk_transfer_function(wavelength)

fig3, ax3 = plt.subplots()
ax3.plot(my_tf)
ax3.set_xlabel(r"$\tau [r_{\rm{g}}]$")
ax3.set_ylabel(r"$\Psi$ [arb.]")

fig3.set_figheight(3)

# convolve transfer function with signals to get reprocessed signals

time_ax, reprocessed_signal = convolve_signal_with_transfer_function(
    smbh_mass_exp=mass_exponent,
    driving_signal=bpl_signal,
    transfer_function=my_tf,
    redshift_source=redshift_source,
    desired_cadence_in_days=0.1,
)


reprocessed_signal /= np.std(reprocessed_signal)

fig4, ax4 = plt.subplots()
ax4.plot(time_ax, reprocessed_signal, alpha=0.7, label="reprocessed light curve")
ax4.plot(
    time_axis * (1 + redshift_source),
    bpl_signal,
    alpha=0.7,
    color="black",
    linewidth=0.2,
    label="driving signal",
)

ax4.legend()
fig4.set_figheight(3)

ax4.set_xlabel("time in observer frame [day]")
ax4.set_ylabel("flux [arb.]")

plt.show()
