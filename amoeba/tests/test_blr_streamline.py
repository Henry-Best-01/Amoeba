import pytest
import numpy as np
import numpy.testing as npt
from amoeba.Classes.blr_streamline import Streamline


def test_init():

    launch_radius = 500  # Rg
    launch_theta = 0  # degrees
    max_height = 1000  # Rg
    characteristic_distance = max_height // 5
    asymptotic_poloidal_velocity = 0.2
    poloidal_launch_velocity = 10**-5
    too_fast = 1.2
    too_close = 0
    too_wide = 90
    bad_radial_vector = np.linspace(0, 100, 100)
    bad_velocity_vector = np.linspace(0, 0.4, 20)

    with npt.assert_raises(AssertionError):
        Streamline(
            launch_radius,
            launch_theta,
            max_height,
            characteristic_distance,
            too_fast,
        )

    with npt.assert_raises(AssertionError):
        Streamline(
            too_close,
            launch_theta,
            max_height,
            characteristic_distance,
            asymptotic_poloidal_velocity,
        )

    with npt.assert_raises(AssertionError):
        Streamline(
            launch_radius,
            too_wide,
            max_height,
            characteristic_distance,
            asymptotic_poloidal_velocity,
        )

    with npt.assert_raises(AssertionError):
        Streamline(
            launch_radius,
            launch_theta,
            max_height,
            characteristic_distance,
            asymptotic_poloidal_velocity,
            poloidal_launch_velocity=too_fast,
        )

    with npt.assert_raises(AssertionError):
        Streamline(
            launch_radius,
            launch_theta,
            max_height,
            characteristic_distance,
            asymptotic_poloidal_velocity,
            poloidal_launch_velocity=too_fast,
        )

    with npt.assert_raises(AssertionError):
        Streamline(
            launch_radius,
            launch_theta,
            max_height,
            characteristic_distance,
            asymptotic_poloidal_velocity,
            velocity_vector=bad_velocity_vector,
            radial_vector=bad_radial_vector,
        )

    test_blr_streamline = Streamline(
        launch_radius,
        launch_theta,
        max_height,
        characteristic_distance,
        asymptotic_poloidal_velocity,
        poloidal_launch_velocity=poloidal_launch_velocity,
    )

    manual_radii = np.linspace(20, 100, 101)
    manual_velocities = 0.1 + 0.05 * np.sin(np.linspace(0, 10, 101) / np.pi)

    strange_but_okay_streamline = Streamline(
        launch_radius,
        launch_theta,
        max_height,
        characteristic_distance,
        asymptotic_poloidal_velocity,
        velocity_vector=manual_velocities,
        radial_vector=manual_radii,
    )

    with npt.assert_raises(AssertionError):
        Streamline(
            launch_radius,
            launch_theta,
            max_height,
            characteristic_distance,
            asymptotic_poloidal_velocity,
            velocity_vector=manual_velocities,
            radial_vector=bad_radial_vector,
        )

    strange_but_differently_overridden_streamline = Streamline(
        launch_radius,
        launch_theta,
        max_height,
        characteristic_distance,
        asymptotic_poloidal_velocity,
        radial_vector=manual_radii,
    )

    automatic_radii_streamline = Streamline(
        launch_radius,
        launch_theta,
        max_height,
        characteristic_distance,
        asymptotic_poloidal_velocity,
        velocity_vector=manual_velocities,
    )

    assert test_blr_streamline.launch_radius == launch_radius
    assert test_blr_streamline.launch_theta == launch_theta * np.pi / 180
    assert test_blr_streamline.launch_height > 0
    assert test_blr_streamline.poloidal_launch_velocity == poloidal_launch_velocity
    assert test_blr_streamline.height_step == 10  # default value
    assert test_blr_streamline.poloidal_launch_velocity == poloidal_launch_velocity
    assert (
        test_blr_streamline.radial_launch_velocity
        == poloidal_launch_velocity * np.sin(test_blr_streamline.launch_theta)
    )
    assert np.size(test_blr_streamline.height_values) == np.size(
        test_blr_streamline.poloidal_velocity
    )

    # velocities should be increasing!
    # radii should be constant!
    for jj in range(len(test_blr_streamline.poloidal_velocity) - 1):
        assert (
            test_blr_streamline.poloidal_velocity[jj + 1]
            > test_blr_streamline.poloidal_velocity[jj]
        )
        assert (
            test_blr_streamline.radii_values[jj + 1]
            == test_blr_streamline.radii_values[jj]
        )

    # choose a nice angle to relate max radius to max height
    launch_theta_angled = 45
    test_blr_streamline_angled = Streamline(
        launch_radius,
        launch_theta_angled,
        max_height,
        characteristic_distance,
        asymptotic_poloidal_velocity,
        poloidal_launch_velocity=poloidal_launch_velocity,
    )

    for jj in range(len(test_blr_streamline_angled.poloidal_velocity) - 1):
        assert (
            test_blr_streamline_angled.poloidal_velocity[jj + 1]
            > test_blr_streamline_angled.poloidal_velocity[jj]
        )
        assert (
            test_blr_streamline_angled.radii_values[jj + 1]
            > test_blr_streamline_angled.radii_values[jj]
        )
        print(test_blr_streamline_angled.poloidal_velocity)
        print(test_blr_streamline_angled.dpol_vel_dz_on_vel)
        assert test_blr_streamline_angled.dpol_vel_dz_on_vel[jj + 1] > 0

    assert (
        test_blr_streamline_angled.radii_values[-1] - test_blr_streamline.launch_radius
        == test_blr_streamline_angled.height_values[-1]
    )
