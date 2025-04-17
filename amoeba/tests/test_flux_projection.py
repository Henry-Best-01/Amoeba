import pytest
from amoeba.Classes.flux_projection import FluxProjection
from amoeba.Classes.accretion_disk import AccretionDisk
from amoeba.Classes.blr import BroadLineRegion
from amoeba.Classes.blr_streamline import Streamline
from amoeba.Util.util import create_maps
import astropy.units as u
import numpy as np
import numpy.testing as npt


def test_initialization():

    # start with completely generic maps
    # take a gaussian
    x_vals = np.linspace(-100, 100, 201)
    X, Y = np.meshgrid(x_vals, x_vals)
    R2 = X**2 + Y**2
    flux_map = np.exp(-(R2 / 20**2))

    observer_frame_wavelength_in_nm = 500
    smbh_mass_exp = 7.5
    redshift_source = 1.2
    r_out_in_gravitational_radii = 100
    inclination_angle = 0
    OmM = 0.3
    H0 = 70

    my_gaussian_projection = FluxProjection(
        flux_map,
        observer_frame_wavelength_in_nm,
        smbh_mass_exp,
        redshift_source,
        r_out_in_gravitational_radii,
        inclination_angle,
        OmM=0.3,
        H0=70,
    )

    # check the sum naturally
    natural_sum = np.sum(flux_map * my_gaussian_projection.pixel_size**2)
    assert natural_sum == my_gaussian_projection.total_flux
    assert (
        observer_frame_wavelength_in_nm
        == my_gaussian_projection.observer_frame_wavelength_in_nm
    )
    assert smbh_mass_exp == my_gaussian_projection.smbh_mass_exp
    assert redshift_source == my_gaussian_projection.redshift_source

    # test natural initialization through the AccretionDisk object

    agn_dict = create_maps(
        smbh_mass_exp,
        redshift_source,
        r_out_in_gravitational_radii,
        inclination_angle,
        100,
    )
    agn_disk = AccretionDisk(**agn_dict)

    projected_flux_object = agn_disk.calculate_surface_intensity_map(
        observer_frame_wavelength_in_nm,
    )

    assert (
        projected_flux_object.observer_frame_wavelength_in_nm
        == my_gaussian_projection.observer_frame_wavelength_in_nm
    )
    assert projected_flux_object.total_flux > 0
    assert (
        projected_flux_object.inclination_angle
        == my_gaussian_projection.inclination_angle
    )
    assert projected_flux_object.pixel_size > 0
    assert projected_flux_object.rg > 0
    assert projected_flux_object.lum_dist > 0
    assert projected_flux_object.ang_diam_dist > 0
    assert projected_flux_object.redshift_source == redshift_source

    # test construction via BLR object
    smbh_mass_exp = 7.28384
    launch_radius = 10  # Rg
    launch_theta = 0  # degrees
    max_height = 100  # Rg
    rest_frame_wavelength_in_nm = 600
    characteristic_distance = max_height // 5
    asymptotic_poloidal_velocity = 0.2
    poloidal_launch_velocity = 10**-5

    test_blr_streamline = Streamline(
        launch_radius,
        launch_theta,
        max_height,
        characteristic_distance,
        asymptotic_poloidal_velocity,
        poloidal_launch_velocity=poloidal_launch_velocity,
    )

    launch_theta_angled = 45
    test_blr_streamline_angled = Streamline(
        launch_radius,
        launch_theta_angled,
        max_height,
        characteristic_distance,
        asymptotic_poloidal_velocity,
        poloidal_launch_velocity=poloidal_launch_velocity,
    )

    blr = BroadLineRegion(
        smbh_mass_exp, max_height, rest_frame_wavelength_in_nm, redshift_source
    )

    blr.add_streamline_bounded_region(test_blr_streamline, test_blr_streamline_angled)

    inclination_angle = 50

    projection = blr.project_blr_total_intensity(inclination_angle)

    assert projection.flux_array.ndim == 2
    assert projection.total_flux > 0
    assert isinstance(
        projection.observer_frame_wavelength_in_nm,
        (u.Quantity, float, list, np.ndarray),
    )
    assert projection.smbh_mass_exp == smbh_mass_exp
    assert projection.r_out_in_gravitational_radii > max_height + launch_radius
    assert projection.inclination_angle == inclination_angle


def test_add_flux_projection():

    # generate two flux projections and test that we can raise errors
    x_vals = np.linspace(-100, 100, 200)
    X, Y = np.meshgrid(x_vals, x_vals)
    R2 = X**2 + Y**2
    flux_map = 200 * np.exp(-(R2 / 20**2))

    observer_frame_wavelength_in_nm = 500
    smbh_mass_exp = 7.5
    redshift_source = 1.2
    r_out_in_gravitational_radii = 100
    inclination_angle = 0
    OmM = 0.3
    H0 = 70

    my_gaussian_projection_1 = FluxProjection(
        flux_map,
        observer_frame_wavelength_in_nm,
        smbh_mass_exp,
        redshift_source,
        r_out_in_gravitational_radii,
        inclination_angle,
        OmM=0.3,
        H0=70,
    )

    wavelength_range = [23, 440]

    flux_map_2 = 500 * np.exp(-(abs(R2 - 50) / 20**2))
    my_gaussian_projection_2 = FluxProjection(
        flux_map_2,
        wavelength_range,
        smbh_mass_exp,
        redshift_source + 1,
        r_out_in_gravitational_radii,
        inclination_angle,
        OmM=0.3,
        H0=70,
    )

    assert np.sum((flux_map - flux_map_2) ** 2) > 0

    with npt.assert_raises(AssertionError):
        my_gaussian_projection_1.add_flux_projection(my_gaussian_projection_2)

    my_gaussian_projection_2.redshift_source = redshift_source
    my_gaussian_projection_2.smbh_mass_exp += 1

    with npt.assert_raises(AssertionError):
        my_gaussian_projection_1.add_flux_projection(my_gaussian_projection_2)

    my_gaussian_projection_2.smbh_mass_exp = smbh_mass_exp
    my_gaussian_projection_2.inclination_angle += 1

    with npt.assert_raises(AssertionError):
        my_gaussian_projection_1.add_flux_projection(my_gaussian_projection_2)

    my_gaussian_projection_2.inclination_angle = inclination_angle

    previous_total_flux = my_gaussian_projection_1.total_flux
    assert my_gaussian_projection_1.add_flux_projection(my_gaussian_projection_2)

    assert my_gaussian_projection_1.total_flux > previous_total_flux
    assert np.shape(my_gaussian_projection_1.flux_array) == np.shape(flux_map)

    assert np.sum((my_gaussian_projection_1.flux_array - flux_map) ** 2) > 0
    assert np.sum((my_gaussian_projection_1.flux_array - flux_map_2) ** 2) > 0

    assert (
        my_gaussian_projection_1.r_out_in_gravitational_radii
        == r_out_in_gravitational_radii
    )

    x_vals_big = np.linspace(-200, 200, 100)
    X_big, Y_big = np.meshgrid(x_vals_big, x_vals_big)
    R2_big = X_big**2 + Y_big**2
    flux_map_big = np.exp(-(R2_big / 50**2))

    observer_frame_wavelength_in_nm_2 = 800
    big_r_out_in_gravitational_radii = 200

    my_big_gaussian_projection = FluxProjection(
        flux_map_big,
        observer_frame_wavelength_in_nm_2,
        smbh_mass_exp,
        redshift_source,
        big_r_out_in_gravitational_radii,
        inclination_angle,
        OmM=0.3,
        H0=70,
    )

    low_res_rg_per_pix = (
        my_big_gaussian_projection.pixel_size / my_big_gaussian_projection.rg
    )
    high_res_rg_per_pix = (
        my_gaussian_projection_1.pixel_size / my_gaussian_projection_1.rg
    )
    small_flux_distribution_shape = np.shape(flux_map)
    big_flux_distribution_shape = np.shape(flux_map_big)

    assert low_res_rg_per_pix != high_res_rg_per_pix
    assert small_flux_distribution_shape != big_flux_distribution_shape

    assert my_gaussian_projection_1.add_flux_projection(my_big_gaussian_projection)

    output_res_rg_per_pix = (
        my_gaussian_projection_1.pixel_size / my_gaussian_projection_1.rg
    )
    output_flux_distribution_shape = np.shape(my_gaussian_projection_1.flux_array)

    assert output_res_rg_per_pix == low_res_rg_per_pix
    assert output_flux_distribution_shape == big_flux_distribution_shape

    # assure total flux is the expected value after adding projections
    assert (
        np.sum(
            my_gaussian_projection_1.flux_array * my_gaussian_projection_1.pixel_size**2
        )
        == my_gaussian_projection_1.total_flux
    )
    # assure total flux is actually the sum of all fluxes (at least to 0.01%)
    npt.assert_approx_equal(
        my_gaussian_projection_1.total_flux,
        np.sum(
            [
                previous_total_flux,
                my_gaussian_projection_2.total_flux,
                my_big_gaussian_projection.total_flux,
            ]
        ),
        4,
    )
    # make sure wavelength data is transfered as expected
    assert isinstance(my_gaussian_projection_1.observer_frame_wavelength_in_nm, list)
    min_lam = np.min(
        np.concatenate(
            (
                [observer_frame_wavelength_in_nm],
                wavelength_range,
                [observer_frame_wavelength_in_nm_2],
            )
        )
    )
    max_lam = np.max(
        np.concatenate(
            (
                [observer_frame_wavelength_in_nm],
                wavelength_range,
                [observer_frame_wavelength_in_nm_2],
            )
        )
    )
    assert min_lam == np.min(my_gaussian_projection_1.observer_frame_wavelength_in_nm)
    assert max_lam == np.max(my_gaussian_projection_1.observer_frame_wavelength_in_nm)

    assert my_big_gaussian_projection.add_flux_projection(my_gaussian_projection_1)


def test_get_plotting_axes():
    x_vals = np.linspace(-100, 100, 201)
    X, Y = np.meshgrid(x_vals, x_vals)
    R2 = X**2 + Y**2
    flux_map = np.exp(-(R2 / 20**2))

    observer_frame_wavelength_in_nm = 500
    smbh_mass_exp = 7.5
    redshift_source = 1.2
    r_out_in_gravitational_radii = 100
    inclination_angle = 0
    OmM = 0.3
    H0 = 70

    my_gaussian_projection = FluxProjection(
        flux_map,
        observer_frame_wavelength_in_nm,
        smbh_mass_exp,
        redshift_source,
        r_out_in_gravitational_radii,
        inclination_angle,
        OmM=0.3,
        H0=70,
    )

    generated_X, generated_Y = my_gaussian_projection.get_plotting_axes()

    assert np.shape(generated_X) == np.shape(generated_Y)
    assert np.shape(generated_X) == np.shape(X)
    assert np.sum(X - generated_X) == 0
    assert np.sum(Y - generated_Y) == 0
