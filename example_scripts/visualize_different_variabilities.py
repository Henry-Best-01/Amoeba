import numpy as np
from amoeba.Util.util import generate_signal_from_psd, create_maps
from amoeba.Classes.accretion_disk import AccretionDisk
import matplotlib.pyplot as plt
from scipy.signal import convolve
from astropy import units as u
from astropy import constants as const
from scipy.interpolate import interp1d

plt.style.use("/Users/henrybest/PythonStuff/Code/plot_style.txt")


# signal
signal_length = 20000
maxtau = 1000
tax = np.linspace(0, signal_length - 1, signal_length)
frequencies = np.linspace(1 / (2 * signal_length), 1 / 2, signal_length)
power_spectrum = frequencies ** (-3.0)
random_seed = None

driving_signal = generate_signal_from_psd(
    signal_length,
    power_spectrum,
    frequencies,
    random_seed,
)

# disk params

mexp1 = 8.0
mexp2 = 9.0
mexp3 = 10.0
redshift = 0.0
number_grav_rad = 1000
inclination = 0
resolution = 2000
disk_beta = 1.0

wavelength = 500

disk1kwargs = create_maps(
    mexp1,
    redshift,
    number_grav_rad,
    inclination,
    resolution,
    temp_beta=disk_beta,
    generic_beta=True,
)
disk2kwargs = create_maps(
    mexp2,
    redshift,
    number_grav_rad,
    inclination,
    resolution,
    temp_beta=disk_beta,
    generic_beta=True,
)
disk3kwargs = create_maps(
    mexp3,
    redshift,
    number_grav_rad,
    inclination,
    resolution,
    temp_beta=disk_beta,
    generic_beta=True,
)

Disk1 = AccretionDisk(**disk1kwargs)
Disk2 = AccretionDisk(**disk2kwargs)
Disk3 = AccretionDisk(**disk3kwargs)

disk_1_tf = Disk1.construct_accretion_disk_transfer_function(
    wavelength,
)
disk_2_tf = Disk2.construct_accretion_disk_transfer_function(
    wavelength,
)
disk_3_tf = Disk3.construct_accretion_disk_transfer_function(
    wavelength,
)


disk_1_rg_to_days = (Disk1.rg / const.c.to(u.m / u.day)).value
disk_2_rg_to_days = (Disk2.rg / const.c.to(u.m / u.day)).value
disk_3_rg_to_days = (Disk3.rg / const.c.to(u.m / u.day)).value


lag_axis_disk_1 = np.linspace(0, len(disk_1_tf) * disk_1_rg_to_days, len(disk_1_tf))

lag_axis_disk_2 = np.linspace(0, len(disk_2_tf) * disk_2_rg_to_days, len(disk_2_tf))

lag_axis_disk_3 = np.linspace(0, len(disk_3_tf) * disk_3_rg_to_days, len(disk_3_tf))


disk_1_interp = interp1d(lag_axis_disk_1, disk_1_tf)
disk_2_interp = interp1d(lag_axis_disk_2, disk_2_tf)
disk_3_interp = interp1d(lag_axis_disk_3, disk_3_tf)

tau_ax_1 = np.linspace(
    0, ((max(lag_axis_disk_1)) - 1) / (1 + redshift), int(max(lag_axis_disk_1))
)
tau_ax_2 = np.linspace(
    0, ((max(lag_axis_disk_2)) - 1) / (1 + redshift), int(max(lag_axis_disk_2))
)
tau_ax_3 = np.linspace(
    0, ((max(lag_axis_disk_3)) - 1) / (1 + redshift), int(max(lag_axis_disk_3))
)


daily_lags_1 = disk_1_interp(tau_ax_1)
daily_lags_2 = disk_2_interp(tau_ax_2)
daily_lags_3 = disk_3_interp(tau_ax_3)

daily_lags_1 /= np.sum(daily_lags_1)
daily_lags_2 /= np.sum(daily_lags_2)
daily_lags_3 /= np.sum(daily_lags_3)


conv_signal_disk_1 = convolve(
    driving_signal,
    daily_lags_1,
)
conv_signal_disk_2 = convolve(
    driving_signal,
    daily_lags_2,
)
conv_signal_disk_3 = convolve(
    driving_signal,
    daily_lags_3,
)

conv_signal_disk_1 /= np.std(conv_signal_disk_1)
conv_signal_disk_2 /= np.std(conv_signal_disk_2)
conv_signal_disk_3 /= np.std(conv_signal_disk_3)

SF1 = []
SF2 = []
SF3 = []

SF1_dev = []
SF2_dev = []
SF3_dev = []


for tau in range(maxtau):
    sig_1_contribution = []
    sig_2_contribution = []
    sig_3_contribution = []

    for jj in range(signal_length // 2 - tau):
        sig_1_contribution.append(
            conv_signal_disk_1[signal_length // 2 + jj]
            - conv_signal_disk_1[signal_length // 2 + jj - tau]
        )
        sig_2_contribution.append(
            conv_signal_disk_2[signal_length // 2 + jj]
            - conv_signal_disk_2[signal_length // 2 + jj - tau]
        )
        sig_3_contribution.append(
            conv_signal_disk_3[signal_length // 2 + jj]
            - conv_signal_disk_3[signal_length // 2 + jj - tau]
        )

    SF1.append(np.mean(sig_1_contribution) / (signal_length // 2 - tau))
    SF2.append(np.mean(sig_2_contribution) / (signal_length // 2 - tau))
    SF3.append(np.mean(sig_3_contribution) / (signal_length // 2 - tau))

    SF1_dev.append(np.std(sig_1_contribution))
    SF2_dev.append(np.std(sig_2_contribution))
    SF3_dev.append(np.std(sig_3_contribution))


fig, ax = plt.subplots(3, gridspec_kw={"height_ratios": [3, 1, 1]})

ax[0].plot(tax, driving_signal / np.std(driving_signal))
ax[0].plot(
    tax,
    conv_signal_disk_1[: len(tax)] / np.std(conv_signal_disk_1[: len(tax)]),
    label=r"mass = $10^{8} M_{\odot}$",
)
ax[0].plot(
    tax,
    conv_signal_disk_2[: len(tax)] / np.std(conv_signal_disk_2[: len(tax)]),
    label=r"mass = $10^{9} M_{\odot}$",
)
ax[0].plot(
    tax,
    conv_signal_disk_3[: len(tax)] / np.std(conv_signal_disk_3[: len(tax)]),
    label=r"mass = $10^{10} M_{\odot}$",
)
ax[0].legend(loc=1)
ax[0].set_xlabel("time [days]")
ax[0].set_ylabel(r"mag. [$\sigma$]")

tauax = np.linspace(1, maxtau, maxtau)
ax[1].plot(tauax, SF1, label=r"mass = $10^{8} M_{\odot}$")
ax[1].plot(tauax, SF2, label=r"mass = $10^{9} M_{\odot}$")
ax[1].plot(tauax, SF3, label=r"mass = $10^{10} M_{\odot}$")
ax[1].set_xlabel(r"$\tau$ [days]")
ax[1].set_ylabel(r"mean(SF) [$\sigma$]")
ax[1].set_ylim(-0.1, 0.1)

ax[2].plot(tauax, SF1_dev)
ax[2].plot(tauax, SF2_dev)
ax[2].plot(tauax, SF3_dev)
ax[2].set_xlabel(r"$\tau$ [days]")
ax[2].set_ylabel(r"std(SF) [$\sigma$]")


plt.subplots_adjust(hspace=0.45)
plt.show()
