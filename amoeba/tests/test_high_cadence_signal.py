from amoeba.Util.util import convolve_signal_with_transfer_function
import numpy as np
import matplotlib.pyplot as plt
import numpy.testing as npt


def test_convolve_signal_with_transfer_function():
    tax = np.linspace(0, 1000, 1001)
    sample_tf = np.linspace(0, 1000, 1001) * np.linspace(1000, 0, 1001)
    sample_tf /= np.sum(sample_tf)

    smbh_mass_exp = 9.61

    driving_signal = np.sin(tax * np.pi / 25) * np.sin(tax * np.pi / 103)

    cadence = 0.1
    redshift_source = 1

    new_tax, new_signal = convolve_signal_with_transfer_function(
        smbh_mass_exp=smbh_mass_exp,
        driving_signal=driving_signal,
        transfer_function=sample_tf,
        initial_time_axis=tax,
        redshift_source=redshift_source,
        desired_cadence_in_days=cadence,
    )

    assert len(new_tax) == len(new_signal)
    assert len(new_tax) > len(tax)
    assert max(new_tax) == max(tax * (1 + redshift_source))

    # test that really tiny transfer functions are handled
    mass_exp = 0.1

    cadence = 1
    redshift_source = 0

    new_tax, new_signal = convolve_signal_with_transfer_function(
        smbh_mass_exp=smbh_mass_exp,
        driving_signal=driving_signal,
        transfer_function=sample_tf,
        initial_time_axis=tax,
        redshift_source=redshift_source,
        desired_cadence_in_days=cadence,
    )
