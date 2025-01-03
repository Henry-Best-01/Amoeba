import numpy as np
import matplotlib.pyplot as plt
from amoeba.Classes.accretion_disk import AccretionDisk
from amoeba.Util.util import (
    planck_law,
    calculate_gravitational_radius,
    accretion_disk_temperature,
    create_maps,
    convert_spin_to_isco_radius,
)
import time

t_start = time.time()


wavelength = 700
r_max = 2000
res = 1000
mass_exp = 8.5
spin = 0

inclinations = [5, 25, 45, 65]
tfs = []
mean_tfs = []

for inc in inclinations:
    disk_kwargs = create_maps(
        mass_exp,
        0,
        r_max,
        inc,
        int(res),
        spin,
    )
    Disk = AccretionDisk(**disk_kwargs)

    cur_tf = Disk.construct_accretion_disk_transfer_function(wavelength)

    tfs.append(cur_tf)

    tau_ax = np.linspace(0, len(cur_tf) - 1, len(cur_tf))
    cur_mean = np.sum(tau_ax * cur_tf) / np.sum(cur_tf)
    mean_tfs.append(cur_mean)

fig, ax = plt.subplots()
for jj, inc in enumerate(inclinations):
    tau_ax = np.linspace(0, len(tfs[jj]) - 1, len(tfs[jj]))
    ax.plot(tau_ax, tfs[jj], label=r"$i$ = " + str(inc) + r"$^{o}$")

ax.set_prop_cycle(None)

for jj, inc in enumerate(inclinations):
    ax.plot(
        [mean_tfs[jj], mean_tfs[jj]],
        [0, tfs[jj][int(mean_tfs[jj])]],
        "--",
    )

ax.set_xlabel(r"$\tau [r_{\rm{g}}]$")
ax.set_ylabel(r"$\Psi$ (" + str(wavelength) + " nm $| i)$")

ax.set_xlim(-10, 1000)
ax.set_ylim(-0.0001, 0.006)

ax.legend(loc=1)

fig.set_figheight(3)

print("total time :", str(round(time.time() - t_start, 1)), "s")

plt.show()
