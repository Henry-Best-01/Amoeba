from amoeba.Util.util import (
    create_maps,
    calculate_keplerian_velocity,
    convert_spin_to_isco_radius,
    convert_eddington_ratio_to_accreted_mass,
    accretion_disk_temperature,
    calculate_gravitational_radius,
    planck_law,
    planck_law_derivative,
    calculate_angular_diameter_distance,
    calculate_angular_diameter_distance_difference,
    calculate_luminosity_distance,
    calculate_angular_einstein_radius,
    calculate_einstein_radius_in_meters,
    pull_value_from_grid,
    convert_1d_array_to_2d_array,
    convert_cartesian_to_polar,
    convert_polar_to_cartesian,
    perform_microlensing_convolution,
    extract_light_curve,
    calculate_time_lag_array,
    calculate_geometric_disk_factor,
    calculate_dt_dlx,
    construct_accretion_disk_transfer_function,
    calculate_microlensed_transfer_function,
    generate_drw_signal,
    generate_signal_from_psd,
    generate_snapshots_of_radiation_pattern,
    project_blr_to_source_plane,
    calculate_blr_transfer_function,
    determine_emission_line_velocities,
    convolve_signal_with_transfer_function,
    convert_speclite_filter_to_wavelength_range,
)
import pytest
import numpy as np
import numpy.testing as npt
from astropy import units as u
from astropy import constants as const
from astropy.cosmology import FlatLambdaCDM
from speclite.filters import FilterResponse, load_filters


def test_create_maps():
    mass_exp = 7.0
    redshift = 1.0
    number_grav_radii = 100
    inc_ang = 45
    resolution = 100

    neg_redshift = -1.0
    bad_spin = 1.5
    bad_beta = -42
    bad_inc_ang = 175
    with pytest.raises(AssertionError):
        create_maps(mass_exp, neg_redshift, number_grav_radii, inc_ang, resolution)
    with pytest.raises(AssertionError):
        create_maps(
            mass_exp, redshift, number_grav_radii, inc_ang, resolution, spin=bad_spin
        )
    with pytest.raises(AssertionError):
        create_maps(mass_exp, redshift, number_grav_radii, bad_inc_ang, resolution)
    with pytest.raises(AssertionError):
        create_maps(
            mass_exp,
            neg_redshift,
            number_grav_radii,
            inc_ang,
            resolution,
            temp_beta=bad_beta,
        )
    map1 = create_maps(mass_exp, redshift, number_grav_radii, inc_ang, resolution)
    map2 = create_maps(
        mass_exp, redshift, number_grav_radii, inc_ang, 2 * resolution, albedo=0.4
    )
    # There should be 10 objects
    assert type(map1) == dict
    # The 4 array objects should be of resolution of the image
    assert np.size(map2["radii_array"]) == np.size(map1["radii_array"]) * 4

    edge_on_inc = 90
    albedo_array = np.zeros((resolution, resolution))
    map_testing = create_maps(
        mass_exp,
        redshift,
        number_grav_radii,
        edge_on_inc,
        resolution,
        albedo=albedo_array,
    )


def test_calculate_keplerian_velocity():
    radius_1 = 100
    radius_2 = 10000
    mass_1 = 10
    mass_2 = 20

    kep_vel_1_1 = calculate_keplerian_velocity(radius_1, mass_1)
    kep_vel_1_2 = calculate_keplerian_velocity(radius_1, mass_2)
    kep_vel_2_1 = calculate_keplerian_velocity(radius_2, mass_1)

    # check scaling with params
    assert kep_vel_1_1 == 10 * kep_vel_2_1
    assert kep_vel_1_1 == kep_vel_1_2 / np.sqrt(2)

    # check astropy units work
    mass_astropy_1 = 10 * const.M_sun.to(u.kg)
    radius_astropy_1 = 100 * u.m
    kep_vel_astropy = calculate_keplerian_velocity(radius_1, mass_astropy_1)
    assert kep_vel_1_1 == kep_vel_astropy

    # check that we are returning a numerical value (not an astropy unit).
    assert isinstance(kep_vel_1_1, float)


def test_convert_spin_to_isco_radius():
    spin_positive = 1
    spin_zero = 0
    spin_negative = -1

    # test known limits
    assert convert_spin_to_isco_radius(spin_positive) == 1
    assert convert_spin_to_isco_radius(spin_zero) == 6
    assert convert_spin_to_isco_radius(spin_negative) == 9

    # test out of bounds
    with pytest.raises(ValueError):
        convert_spin_to_isco_radius(4)
    with pytest.raises(ValueError):
        convert_spin_to_isco_radius(-20)


def test_convert_eddington_ratio_to_accreted_mass():
    mass = 10**7
    mass_quantity = 10**7 * const.M_sun.to(u.kg)
    eddington_ratio_1 = 0.1
    efficiency_1 = 0.1
    eddington_ratio_2 = 0.2
    efficiency_2 = 0.2

    accreted_mass_1_1 = convert_eddington_ratio_to_accreted_mass(
        mass, eddington_ratio_1, efficiency=efficiency_1
    )
    accreted_mass_1_2 = convert_eddington_ratio_to_accreted_mass(
        mass, eddington_ratio_1, efficiency=efficiency_2
    )
    accreted_mass_2_1 = convert_eddington_ratio_to_accreted_mass(
        mass, eddington_ratio_2, efficiency=efficiency_1
    )
    accreted_mass_quantity = convert_eddington_ratio_to_accreted_mass(
        mass_quantity, eddington_ratio_1, efficiency=efficiency_1
    )

    # test that this function has the proper scaling
    # e.g. twice as efficient means half the accreted mass
    # and twice the eddington ratio means twice the accreted mass
    assert (accreted_mass_1_1 / 2) == accreted_mass_1_2
    assert (accreted_mass_1_1 * 2) == accreted_mass_2_1
    # Check astropy quantity works as an input
    assert accreted_mass_1_1 == accreted_mass_quantity


def test_accretion_disk_temperature():
    mass_in_m_sun = 10**8
    grav_rad = calculate_gravitational_radius(mass_in_m_sun)
    radii_in_meters = np.linspace(10, 100, 90) * grav_rad
    min_radius_in_meters = 6 * grav_rad
    eddington_ratio = 0.1

    temp_profile_SS = accretion_disk_temperature(
        radii_in_meters,
        min_radius_in_meters,
        mass_in_m_sun,
        eddington_ratio,
        beta=0,
        corona_height=6,
        albedo=1,
        eta_x_rays=0.1,
        generic_beta=False,
        disk_acc=None,
        efficiency=0.1,
        spin=0,
        visc_temp_prof="SS",
    )

    print("next: NT")
    temp_profile_NT = accretion_disk_temperature(
        radii_in_meters,
        min_radius_in_meters,
        mass_in_m_sun,
        eddington_ratio,
        beta=0,
        corona_height=6,
        albedo=1,
        eta_x_rays=0.1,
        generic_beta=False,
        disk_acc=None,
        efficiency=0.1,
        spin=0,
        visc_temp_prof="NT",
    )

    print("next: SS + wind")
    temp_profile_SS_wind = accretion_disk_temperature(
        radii_in_meters,
        min_radius_in_meters,
        mass_in_m_sun * const.M_sun.to(u.kg),
        eddington_ratio,
        beta=0.72,
        corona_height=6,
        albedo=0,
        eta_x_rays=0.1,
        generic_beta=True,
        disk_acc=None,
        efficiency=0.1,
        spin=0,
        visc_temp_prof="SS",
    )

    print("next: rejected, so SS + wind")
    temp_profile_rejected_profile = accretion_disk_temperature(
        radii_in_meters * u.m,
        min_radius_in_meters / 1000 * u.km,
        mass_in_m_sun * const.M_sun.to(u.kg),
        eddington_ratio,
        beta=0.72,
        corona_height=6,
        albedo=0,
        eta_x_rays=0.1,
        generic_beta=True,
        disk_acc=None,
        efficiency=0.1,
        spin=0,
        visc_temp_prof="Not_a_profile",
    )
    print("this one below")

    temp_profile_little_accreted_mass = accretion_disk_temperature(
        radii_in_meters * u.m,
        min_radius_in_meters / 1000 * u.km,
        mass_in_m_sun * const.M_sun.to(u.g),
        eddington_ratio,
        disk_acc=0.001 * const.M_sun.to(u.kg) / u.yr,
        efficiency=0.1,
        spin=0,
        visc_temp_prof="SS",
    )

    temp_profile_another_little_accreted_mass = accretion_disk_temperature(
        radii_in_meters * u.m,
        min_radius_in_meters / 1000 * u.km,
        mass_in_m_sun * const.M_sun.to(u.g),
        eddington_ratio,
        disk_acc=0.001 * (const.M_sun.to(u.kg) / u.yr).to(u.kg / u.s).value,
        efficiency=0.1,
        spin=0,
        visc_temp_prof="SS",
    )

    # check that they're all different
    assert abs(sum(temp_profile_SS - temp_profile_NT)) != 0
    assert abs(sum(temp_profile_SS - temp_profile_SS_wind)) != 0

    # check that the rejected profile defaulted to a SS profile
    assert abs(sum(temp_profile_SS_wind - temp_profile_rejected_profile)) == 0

    assert np.sum(temp_profile_little_accreted_mass) < np.sum(temp_profile_SS)
    assert np.sum(temp_profile_little_accreted_mass) != 0

    # check that at large radii, SS and NT profiles converge
    large_radius = 10**5 * grav_rad
    temperature_difference_tolerance = 10  # Kelvins
    temp_large_radius_SS = accretion_disk_temperature(
        large_radius, min_radius_in_meters, mass_in_m_sun, eddington_ratio
    )
    temp_large_radius_NT = accretion_disk_temperature(
        large_radius,
        min_radius_in_meters,
        mass_in_m_sun,
        eddington_ratio,
        visc_temp_prof="NT",
    )
    assert (
        abs(temp_large_radius_SS - temp_large_radius_NT)
        < temperature_difference_tolerance
    )


def test_planck_law():
    temp_1 = 100
    temp_2 = 500
    wavelength_1 = 100
    wavelength_2 = 500

    radiance_1_1 = planck_law(temp_1, wavelength_1)
    radiance_1_2 = planck_law(temp_1, wavelength_2 * u.nm)
    radiance_2_1 = planck_law(temp_2, wavelength_2)

    # increasing wavelength and temperature should increase the radiance
    # for relatively low temps and short wavelengths.
    assert radiance_1_1 < radiance_1_2
    assert radiance_1_1 < radiance_2_1

    # giving a minimal temperature should provide minimal radiance
    temp_mini = 0.0001
    radiance_mini = planck_law(temp_mini, wavelength_1)
    tolerance = radiance_1_1 / 10**5
    assert radiance_mini == 0


def test_planck_law_derivative():
    temp_1 = 100
    temp_2 = 500
    wavelength = 100

    delta_radiance_1 = planck_law_derivative(temp_1, wavelength)
    delta_radiance_2 = planck_law_derivative(temp_2, wavelength)

    # At higher temperature, the radiance should be increasing faster
    # with respect to temp
    assert delta_radiance_2 > delta_radiance_1


def test_calculate_gravitational_radius():
    mass_1 = 10
    mass_2 = 20
    mass_astropy = 10 * const.M_sun
    mass_u_kg = 10 * u.kg

    grav_rad_1 = calculate_gravitational_radius(mass_1)
    grav_rad_2 = calculate_gravitational_radius(mass_2)
    grav_rad_astropy = calculate_gravitational_radius(mass_astropy)
    grav_rad_kg = calculate_gravitational_radius(mass_u_kg)

    assert grav_rad_1 == grav_rad_2 / 2
    assert grav_rad_1 == grav_rad_astropy
    assert grav_rad_1 > grav_rad_kg


def test_calculate_angular_diameter_distance():
    redshift = 0.1
    OmM = 0.3
    H0 = 70

    ang_diam_dist = calculate_angular_diameter_distance(redshift, OmM=OmM, H0=H0)
    astropy_cosmo = FlatLambdaCDM(H0, OmM)
    ang_diam_dist_astropy = (
        astropy_cosmo.angular_diameter_distance(redshift).to(u.m).value
    )

    # set tolerance to 0.5% due to quadrature integration and
    # rounding of constants
    tolerance = ang_diam_dist_astropy / 500

    assert abs(ang_diam_dist - ang_diam_dist_astropy) < tolerance


def test_calculate_angular_diameter_distance_difference():
    redshift_lens = 0.1
    redshift_source = 0.5
    OmM = 0.3
    H0 = 70

    ang_diam_dist_lens = calculate_angular_diameter_distance(
        redshift_lens, OmM=OmM, H0=H0
    )
    ang_diam_dist_source = calculate_angular_diameter_distance(
        redshift_source, OmM=OmM, H0=H0
    )

    ang_diam_dist_diff = calculate_angular_diameter_distance_difference(
        redshift_lens, redshift_source, OmM=OmM, H0=H0
    )

    ang_diam_dist_diff_reversed = calculate_angular_diameter_distance_difference(
        redshift_source, redshift_lens, OmM=OmM, H0=H0
    )

    # Check that redshift misordering doesn't cause issues
    assert ang_diam_dist_diff == ang_diam_dist_diff_reversed

    # Check that ADDD != ADD(source) - ADD(lens)
    assert ang_diam_dist_diff != (ang_diam_dist_source - ang_diam_dist_lens)

    # for extremely small redshifts, the ADDD should converge to ADD(source)

    redshift_lens_tiny = 10 ** (-10)
    redshift_source = 0.5

    ang_diam_dist_diff_tiny = calculate_angular_diameter_distance_difference(
        redshift_lens_tiny, redshift_source, OmM=OmM, H0=H0
    )
    ang_diam_dist_tiny = calculate_angular_diameter_distance(
        redshift_lens_tiny, OmM=OmM, H0=H0
    )
    # define a small tolerance
    tolerance = ang_diam_dist_source / 10**8
    assert (
        abs(ang_diam_dist_source - (ang_diam_dist_diff_tiny + ang_diam_dist_tiny))
        < tolerance
    )


def test_calculate_luminosity_distance():
    redshift = 0.1
    OmM = 0.3
    H0 = 70

    lum_dist = calculate_luminosity_distance(redshift, OmM=OmM, H0=H0)

    astropy_cosmo = FlatLambdaCDM(H0, OmM)
    lum_dist_astropy = astropy_cosmo.luminosity_distance(redshift).to(u.m).value

    # define a small tolerance due to parameter rounding
    tolerance = lum_dist_astropy / 500
    assert abs(lum_dist_astropy - lum_dist) < tolerance


def test_calculate_angular_einstein_radius():
    redshift_lens = 1.0
    redshift_source = 2.0
    avg_microlens_mass = 0.3 * const.M_sun.to(u.kg)
    OmM = 0.3
    H0 = 70
    test_distance_source = 300 * u.Mpc
    test_distance_lens = 100 * u.Mpc

    star_ang_ein_rad = calculate_angular_einstein_radius(
        redshift_lens,
        redshift_source,
        mean_microlens_mass_in_kg=avg_microlens_mass,
        OmM=OmM,
        H0=H0,
    )

    human_mass = 75 * u.kg
    human_ang_ein_rad = calculate_angular_einstein_radius(
        redshift_lens,
        redshift_source,
        mean_microlens_mass_in_kg=human_mass,
        OmM=OmM,
        H0=H0,
    )

    assert star_ang_ein_rad > human_ang_ein_rad
    assert isinstance(star_ang_ein_rad, float)

    # make sure the code switches the distances in case they are input backwards
    nearby_einstein_radius = calculate_angular_einstein_radius(
        mean_microlens_mass_in_kg=human_mass,
        OmM=OmM,
        H0=H0,
        D_lens=test_distance_source,
        D_source=test_distance_lens,
    )
    nearby_einstein_radius_2 = calculate_angular_einstein_radius(
        mean_microlens_mass_in_kg=human_mass,
        OmM=OmM,
        H0=H0,
        D_lens=test_distance_lens,
        D_source=test_distance_source,
    )
    assert nearby_einstein_radius == nearby_einstein_radius_2


def test_calculate_einstein_radius_in_meters():
    redshift_lens = 1.0
    redshift_source = 2.0
    avg_microlens_mass = 0.3 * const.M_sun.to(u.kg)
    OmM = 0.3
    H0 = 70
    test_distance_source = 300 * u.Mpc
    test_distance_lens = 100 * u.Mpc

    star_ein_rad = calculate_einstein_radius_in_meters(
        redshift_lens,
        redshift_source,
        mean_microlens_mass_in_kg=avg_microlens_mass,
        OmM=OmM,
        H0=H0,
    )

    human_mass = 75 * u.kg
    human_ein_rad = calculate_einstein_radius_in_meters(
        redshift_lens,
        redshift_source,
        mean_microlens_mass_in_kg=human_mass,
        OmM=OmM,
        H0=H0,
    )

    assert star_ein_rad > human_ein_rad
    # Test that we removed the astropy units
    assert isinstance(star_ein_rad, (float, int))

    nearby_einstein_radius = calculate_einstein_radius_in_meters(
        mean_microlens_mass_in_kg=human_mass,
        OmM=OmM,
        H0=H0,
        D_lens=test_distance_source,
        D_source=test_distance_lens,
    )
    nearby_einstein_radius_2 = calculate_einstein_radius_in_meters(
        mean_microlens_mass_in_kg=human_mass,
        OmM=OmM,
        H0=H0,
        D_lens=test_distance_lens,
        D_source=test_distance_source,
    )
    assert nearby_einstein_radius == nearby_einstein_radius_2

    backwards_einstein_radius = calculate_einstein_radius_in_meters(
        redshift_lens=redshift_source,
        redshift_source=redshift_lens,
        mean_microlens_mass_in_kg=human_mass,
        OmM=OmM,
        H0=H0,
    )
    assert backwards_einstein_radius == human_ein_rad


def test_pull_value_from_grid():
    x_values = np.linspace(1, 10, 10)
    y_values = np.linspace(2, 20, 10)

    # Note that arrays are created internally as (y, x)
    X, Y = np.meshgrid(x_values, y_values)

    test_grid = Y + X
    assert test_grid[1, 0] == 5
    assert test_grid[0, 1] == 4

    # explicitly pull values off grid
    assert test_grid[0, 0] == 3
    assert pull_value_from_grid(test_grid, 0, 0) == 3

    # test max value
    assert test_grid[-1, -1] == pull_value_from_grid(test_grid, 9, 9)

    # test small changes follow expected interpolation
    assert pull_value_from_grid(test_grid, 0, 0.5) == 4.0
    assert pull_value_from_grid(test_grid, 0.5, 0) == 3.5

    # show dx < dy
    assert pull_value_from_grid(test_grid, 5, 5.2) > pull_value_from_grid(
        test_grid, 5.2, 5
    )

    # test multiple values pulled at once
    pull_values_x = np.linspace(3, 5, 10)
    pull_values_y = np.linspace(5, 2, 10)
    assert len(pull_values_x) == len(pull_values_y)
    output_values = pull_value_from_grid(
        test_grid,
        pull_values_x,
        pull_values_y,
    )

    # show that these are equivalent
    assert len(output_values) == len(pull_values_y)
    for jj in range(len(pull_values_x)):
        current_value = pull_value_from_grid(
            test_grid, pull_values_x[jj], pull_values_y[jj]
        )
        assert output_values[jj] == current_value


def test_convert_1d_array_to_2d_array():
    test_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    repacked_array = convert_1d_array_to_2d_array(test_list)
    assert np.shape(repacked_array) == (3, 3)
    assert repacked_array[0, 0] == 1
    assert repacked_array[-1, -1] == 9
    assert repacked_array[0, 1] == 2
    assert repacked_array[1, 0] == 4


def test_conversions_between_cartesian_and_polar():
    x1 = 1
    y1 = 2

    hyp1 = (x1**2 + y1**2) ** 0.5

    # we want phi = 0 pointing along the -y axis, to the observer if
    # the plane is inclined. Need to test using a different numpy function.

    expected_angle = np.arcsin(y1 / hyp1) + np.pi / 2

    x2 = 2
    y2 = 4

    r1, azi1 = convert_cartesian_to_polar(x1, y1)
    r2, azi2 = convert_cartesian_to_polar(x2, y2)

    np.testing.assert_almost_equal(azi1, expected_angle)

    assert azi2 == azi1
    assert r1 == (x1**2 + y1**2) ** 0.5
    assert r2 == (x2**2 + y2**2) ** 0.5

    # show this is invertible
    x_out_1, y_out_1 = convert_polar_to_cartesian(r1, azi1)
    x_out_2, y_out_2 = convert_polar_to_cartesian(r2, azi2)

    # internal rounding makes these off by ~10^-8
    np.testing.assert_almost_equal(x1, x_out_1)
    np.testing.assert_almost_equal(x2, x_out_2)
    np.testing.assert_almost_equal(y1, y_out_1)
    np.testing.assert_almost_equal(y2, y_out_2)

    _, x_axis = convert_cartesian_to_polar(1, 0)
    _, y_axis = convert_cartesian_to_polar(0, 1)
    _, neg_x_axis = convert_cartesian_to_polar(-1, 0)
    _, neg_y_axis = convert_cartesian_to_polar(0, -1)

    np.testing.assert_almost_equal(x_axis, np.pi / 2)
    np.testing.assert_almost_equal(y_axis, np.pi)
    np.testing.assert_almost_equal(neg_x_axis, 3 * np.pi / 2)
    np.testing.assert_almost_equal(neg_y_axis, 0)


def test_perform_microlensing_convolution():
    magnification_array_identity = np.ones((100, 100))
    x_ax = 5 - abs(5 - np.linspace(1, 10, 10))
    flux_x, flux_y = np.meshgrid(x_ax, x_ax)
    sample_flux_map = (flux_x**2 + flux_y**2) ** 0.5

    redshift_l = 0.5
    redshift_s = 2.0
    relative_orientation_1 = 0
    relative_orientation_1a = 360
    relative_orientation_2 = 180

    convolution_1, px_shift_1 = perform_microlensing_convolution(
        magnification_array_identity,
        sample_flux_map,
        redshift_l,
        redshift_s,
        relative_orientation=relative_orientation_1,
    )
    convolution_1a, px_shift_1a = perform_microlensing_convolution(
        magnification_array_identity,
        sample_flux_map,
        redshift_l,
        redshift_s,
        relative_orientation=relative_orientation_1a,
    )
    convolution_2, px_shift_2 = perform_microlensing_convolution(
        magnification_array_identity,
        sample_flux_map,
        redshift_l,
        redshift_s,
        relative_orientation=relative_orientation_2,
    )
    convolution_2b, px_shift_2b = perform_microlensing_convolution(
        magnification_array_identity,
        sample_flux_map,
        redshift_l,
        redshift_s,
        relative_orientation=None,
    )
    assert px_shift_2 == px_shift_2b

    # test that we conserve flux after accounting for rescaling of pixels
    value = convolution_1[3, 2]
    npt.assert_almost_equal(value, np.sum(sample_flux_map))

    # test the identity convolution is constant
    npt.assert_almost_equal(convolution_1a[2, 5], convolution_1a[-4, -1])

    # test that rotation of 180, 360 deg is identity for symmetric source
    npt.assert_almost_equal(np.sum(convolution_1), np.sum(convolution_1a))
    npt.assert_almost_equal(np.sum(convolution_1), np.sum(convolution_2))

    assert px_shift_1 == px_shift_1a
    assert px_shift_1 == px_shift_2

    # by defining 2 point sources, the convolution should only have
    # one point source, spread by the pixel shift
    mag_array_point = np.zeros(np.shape(magnification_array_identity))
    mag_array_point[75, 25] = 1

    convolution_2_points, px_shift = perform_microlensing_convolution(
        mag_array_point,
        mag_array_point,
        redshift_l,
        redshift_s,
        relative_orientation=relative_orientation_1,
    )

    total_value = np.sum(convolution_2_points)
    npt.assert_almost_equal(total_value, 1)


def test_extract_light_curve():
    ones_array = np.ones((100, 100))
    fake_flux_distribution = [
        [0, 0, 1, 0, 0],
        [0, 1, 2, 1, 0],
        [1, 2, 3, 2, 1],
        [0, 1, 2, 1, 0],
        [0, 0, 1, 0, 0],
    ]
    effective_vel = 10000
    effective_vel_too_fast = 10**20
    light_curve_length = 5
    # these parameters make the light curve 10 pixels long
    pixel_size = calculate_einstein_radius_in_meters(
        redshift_lens=1.0,
        redshift_source=2.0,
        mean_microlens_mass_in_kg=0.3 * const.M_sun.to(u.kg),
        OmM=0.3,
        H0=70,
    )

    convolution_1, px_shift_1 = perform_microlensing_convolution(
        ones_array,
        fake_flux_distribution,
        1.0,
        2.0,
        mean_microlens_mass_in_kg=0.3 * const.M_sun.to(u.kg),
        number_of_microlens_einstein_radii=1,
        number_of_smbh_gravitational_radii=20,
    )

    light_curve = extract_light_curve(
        convolution_1,
        pixel_size,
        effective_vel,
        light_curve_length,
        px_shift_1,
    )

    light_curve2 = extract_light_curve(
        convolution_1,
        pixel_size,
        effective_vel * u.km / u.s,
        light_curve_length * u.yr,
        px_shift_1,
    )

    for tt in range(len(light_curve)):
        npt.assert_almost_equal(light_curve[tt], np.asarray(light_curve).mean())

    original_total_integrated_flux = np.sum(fake_flux_distribution)
    convolved_total_flux = light_curve[0]

    npt.assert_almost_equal(original_total_integrated_flux, convolved_total_flux)

    # check we can pull 1000 light curves from this convolution
    # to show it never extends beyond the edge
    for jj in range(1000):
        extract_light_curve(
            convolution_1,
            pixel_size,
            effective_vel,
            light_curve_length,
            px_shift_1,
        )

    light_curve, x_pos, y_pos = extract_light_curve(
        convolution_1,
        pixel_size,
        effective_vel,
        light_curve_length,
        px_shift_1,
        return_track_coords=True,
    )

    # show the light curve is drawn from multiple positions, which
    # we can almost guarantee because rng will never pick exactly
    # n*90 deg for int(n).
    assert x_pos[0] != x_pos[-1]
    assert y_pos[0] != y_pos[-1]

    # show we can pick a specific light curve
    # note that coordinates on an array are in signature (y, x)
    light_curve, x_pos, y_pos = extract_light_curve(
        convolution_1,
        pixel_size,
        effective_vel,
        light_curve_length,
        px_shift_1,
        x_start_position=30,
        y_start_position=40,
        phi_travel_direction=90,
        return_track_coords=True,
    )

    assert x_pos[0] == x_pos[-1]
    assert y_pos[0] != y_pos[-1]

    light_curve, x_pos, y_pos = extract_light_curve(
        convolution_1,
        pixel_size,
        effective_vel,
        light_curve_length,
        px_shift_1,
        x_start_position=30,
        y_start_position=40,
        phi_travel_direction=180,
        return_track_coords=True,
    )

    assert x_pos[0] != x_pos[-1]
    assert y_pos[0] == y_pos[-1]

    # show we can get the average value for many cases
    # e.g. source too large (high pixel shift)
    light_curve_avg_1 = extract_light_curve(
        convolution_1, pixel_size, effective_vel, light_curve_length, 300
    )
    # e.g. light curve too long
    light_curve_avg_2 = extract_light_curve(
        convolution_1, pixel_size, effective_vel * 10e8, light_curve_length, px_shift_1
    )
    # e.g. direction chosen to extend beyond the convolution's boundaries
    light_curve_avg_3 = extract_light_curve(
        convolution_1,
        pixel_size,
        effective_vel,
        light_curve_length,
        px_shift_1,
        x_start_position=90,
        y_start_position=90,
        phi_travel_direction=0,
    )
    light_curve_avg_4 = extract_light_curve(
        convolution_1,
        pixel_size,
        effective_vel,
        light_curve_length,
        px_shift_1,
        x_start_position=90,
        y_start_position=-90,
        phi_travel_direction=0,
    )
    light_curve_avg_5 = extract_light_curve(
        convolution_1,
        pixel_size,
        effective_vel,
        light_curve_length,
        px_shift_1,
        x_start_position=-90,
        y_start_position=90,
        phi_travel_direction=0,
    )
    light_curve_avg_6 = extract_light_curve(
        convolution_1,
        pixel_size,
        effective_vel,
        light_curve_length,
        px_shift_1,
        x_start_position=99,
        y_start_position=99,
        phi_travel_direction=27,
    )

    assert light_curve_avg_1 == light_curve_avg_2
    assert light_curve_avg_3 == light_curve_avg_2
    assert light_curve_avg_1 == light_curve_avg_4
    assert light_curve_avg_1 == light_curve_avg_5
    assert light_curve_avg_1 == light_curve_avg_6


def test_calculate_time_lag_array():
    # point source tests

    radii_array = 0
    phi_array = 0
    inclination_angle = 0
    corona_height = 10
    axis_offset_in_gravitational_radii = 0
    angle_offset_in_degrees = 0
    height_array = 0

    time_lag_1 = calculate_time_lag_array(
        radii_array,
        phi_array,
        inclination_angle,
        corona_height,
        axis_offset_in_gravitational_radii=axis_offset_in_gravitational_radii,
        angle_offset_in_degrees=angle_offset_in_degrees,
        height_array=height_array,
    )

    assert time_lag_1 == (2 * corona_height)

    corona_height = 0
    time_lag_2 = calculate_time_lag_array(
        radii_array,
        phi_array,
        inclination_angle,
        corona_height,
        axis_offset_in_gravitational_radii=axis_offset_in_gravitational_radii,
        angle_offset_in_degrees=angle_offset_in_degrees,
        height_array=height_array,
    )

    assert time_lag_2 == (2 * corona_height)

    # test manual construction of radii and phi arrays
    root2 = np.sqrt(2)

    radii_array = np.asarray([[root2, 1, root2], [1, 0, 1], [root2, 1, root2]])

    phi_array = np.asarray(
        [
            [5 * np.pi / 4, np.pi, 3 * np.pi / 4],
            [6 * np.pi / 4, 0, 2 * np.pi / 4],
            [7 * np.pi / 4, 0, 1 * np.pi / 4],
        ]
    )

    height_array = None
    corona_height = 10

    time_lag_3 = calculate_time_lag_array(
        radii_array,
        phi_array,
        inclination_angle,
        corona_height,
        axis_offset_in_gravitational_radii=axis_offset_in_gravitational_radii,
        angle_offset_in_degrees=angle_offset_in_degrees,
        height_array=height_array,
    )

    assert time_lag_3[0, 0] == time_lag_3[-1, -1]
    assert time_lag_3[0, 0] == time_lag_3[0, -1]
    assert time_lag_3[1, 0] == time_lag_3[1, -1]
    assert time_lag_3[0, 1] == time_lag_3[-1, 1]
    assert time_lag_3[1, 1] == 2 * corona_height

    # test off axis time lags
    axis_offset_in_gravitational_radii = 1

    time_lag_4 = calculate_time_lag_array(
        radii_array,
        phi_array,
        inclination_angle,
        corona_height,
        axis_offset_in_gravitational_radii=axis_offset_in_gravitational_radii,
        angle_offset_in_degrees=angle_offset_in_degrees,
        height_array=height_array,
    )

    assert np.min(time_lag_4) == 2 * corona_height

    # rotate lamppost to other side of disk
    angle_offset_in_degrees = 180
    time_lag_5 = calculate_time_lag_array(
        radii_array,
        phi_array,
        inclination_angle,
        corona_height,
        axis_offset_in_gravitational_radii=axis_offset_in_gravitational_radii,
        angle_offset_in_degrees=angle_offset_in_degrees,
        height_array=height_array,
    )

    assert np.min(time_lag_5) == 2 * corona_height
    assert time_lag_4[1, -1] < time_lag_5[1, -1]
    assert time_lag_4[0, 1] == time_lag_5[0, 1]
    assert time_lag_4[1, 0] > time_lag_5[1, 0]

    # test with heights included (non-flat accretion disk)
    axis_offset_in_gravitational_radii = 0
    height_array = radii_array
    corona_height = 10

    time_lag_6 = calculate_time_lag_array(
        radii_array,
        phi_array,
        inclination_angle,
        corona_height,
        height_array=height_array,
    )

    # with disk flaring, center time lag should be longest lag.
    # argmax of a 2d array flattens the array, so index 4 is center.
    assert np.argmax(time_lag_6) == 4
    assert time_lag_6[1, 1] == 2 * corona_height

    # compare with other points on isodelay surface
    assert time_lag_6[1, 0] == time_lag_6[1, -1]
    assert time_lag_6[1, 0] == time_lag_6[1, 0]
    assert time_lag_6[1, 0] == time_lag_6[-1, 1]

    # explicitly calculate delay for one point
    assert (
        time_lag_6[1, 1]
        == ((corona_height - height_array[1, 1]) ** 2 + radii_array[1, 1] ** 2) ** 0.5
        + corona_height
    )

    # test inclined disk
    inclination_angle = 35
    height_array = None

    time_lag_7 = calculate_time_lag_array(
        radii_array,
        phi_array,
        inclination_angle,
        corona_height,
    )

    assert time_lag_7[1, 0] == time_lag_7[1, -1]
    assert time_lag_7[-1, 1] < time_lag_7[0, 1]


def test_calculate_geometric_disk_factor():
    temp_array = np.ones((100, 100))
    x_vals = abs(np.linspace(1, 100, 100) - 50)
    X, Y = np.meshgrid(x_vals, x_vals)
    radii_map, phi_map = convert_cartesian_to_polar(X, Y)
    smbh_mass_exponent = 8.0
    corona_height = 10
    height_array = radii_map / 10

    geo_disk_factor_array = calculate_geometric_disk_factor(
        temp_array,
        radii_map,
        phi_map,
        smbh_mass_exponent,
        corona_height,
        height_array=height_array,
    )

    assert geo_disk_factor_array[50, 50] > geo_disk_factor_array[0, 50]

    albedo_array = 0 + 0.1 * radii_map / np.max(radii_map)

    geo_disk_factor_array_albedos = calculate_geometric_disk_factor(
        temp_array,
        radii_map,
        phi_map,
        smbh_mass_exponent,
        corona_height,
        height_array=height_array,
        albedo_array=albedo_array,
    )
    geo_disk_factor_array_const_albedos = calculate_geometric_disk_factor(
        temp_array,
        radii_map,
        phi_map,
        smbh_mass_exponent,
        corona_height,
        height_array=height_array,
        albedo_array=0.75,
    )

    # since higher albedo means less absorption, every position (except [50, 50])
    # will be less than before
    test_x = 35
    test_y = 73
    assert (
        geo_disk_factor_array_albedos[test_x, test_y]
        < geo_disk_factor_array[test_x, test_y]
    )

    # show we can input off-axis source
    axis_offset_in_gravitational_radii = 10
    angle_offset_in_degrees = 47

    calculate_geometric_disk_factor(
        temp_array,
        radii_map,
        phi_map,
        smbh_mass_exponent,
        corona_height,
        axis_offset_in_gravitational_radii=axis_offset_in_gravitational_radii,
        angle_offset_in_degrees=angle_offset_in_degrees,
        height_array=height_array,
        albedo_array=albedo_array,
    )


def test_calculate_dt_dlx():

    temp_array = 1000 * np.ones((100, 100))
    # must handle zeros in temp profile
    temp_array[47:53, 47:53] = 0

    x_vals = abs(np.linspace(1, 100, 100) - 50)
    X, Y = np.meshgrid(x_vals, x_vals)
    radii_array, phi_array = convert_cartesian_to_polar(X, Y)
    smbh_mass_exponent = 8.0
    corona_height = 10
    height_array = radii_array / 10

    axis_offset_in_gravitational_radii = 0
    angle_offset_in_degrees = 0
    albedo_array = None

    dt_dlx = calculate_dt_dlx(
        temp_array,
        radii_array,
        phi_array,
        smbh_mass_exponent,
        corona_height,
        axis_offset_in_gravitational_radii=axis_offset_in_gravitational_radii,
        angle_offset_in_degrees=angle_offset_in_degrees,
        height_array=height_array,
        albedo_array=albedo_array,
    )

    # we shouldn't be injecting a massive amount of energy
    # also show that nans are removed
    assert np.sum(dt_dlx) < 10**3


def test_construct_accretion_disk_transfer_function():

    test_wavelength_1 = 100  # nm
    test_wavelength_2 = 300
    smbh_mass_exponent = 8.0
    corona_height = 10
    inclination_angle = 30
    axis_offset_in_gravitational_radii = 0
    angle_offset_in_degrees = 0
    height_array = None
    albedo_array = None

    # to test reasonable transfer functions, we must make a reasonable disk
    disk_dict = create_maps(smbh_mass_exponent, 0, 500, inclination_angle, 500)

    temp_array = disk_dict["temp_array"]
    radii_array = disk_dict["radii_array"]
    phi_array = disk_dict["phi_array"]
    g_array = disk_dict["g_array"]
    inclination_angle = disk_dict["inclination_angle"]
    smbh_mass_exponent = disk_dict["smbh_mass_exp"]
    corona_height = disk_dict["corona_height"]

    transfer_function_test_1 = construct_accretion_disk_transfer_function(
        test_wavelength_1,
        temp_array,
        radii_array,
        phi_array,
        g_array,
        inclination_angle,
        smbh_mass_exponent,
        corona_height,
        axis_offset_in_gravitational_radii=axis_offset_in_gravitational_radii,
        angle_offset_in_degrees=angle_offset_in_degrees,
        height_array=height_array,
        albedo_array=albedo_array,
    )

    # show it was normalized
    npt.assert_almost_equal(np.sum(transfer_function_test_1), 1)
    # there should be an initial time lag
    assert transfer_function_test_1[0] == 0
    assert transfer_function_test_1[5] == 0
    assert transfer_function_test_1[8] == 0

    # the same disk at a redder wavelength should peak at later time lag
    transfer_function_test_2 = construct_accretion_disk_transfer_function(
        test_wavelength_2,
        temp_array,
        radii_array,
        phi_array,
        g_array,
        inclination_angle,
        smbh_mass_exponent,
        corona_height,
        axis_offset_in_gravitational_radii=axis_offset_in_gravitational_radii,
        angle_offset_in_degrees=angle_offset_in_degrees,
        height_array=height_array,
        albedo_array=albedo_array,
    )

    tau_axis = np.linspace(
        0, len(transfer_function_test_1) - 1, len(transfer_function_test_1)
    )
    mean_lag_1 = np.sum(tau_axis * transfer_function_test_1)
    mean_lag_2 = np.sum(tau_axis * transfer_function_test_2)
    assert mean_lag_1 < mean_lag_2


def test_calculate_microlensed_transfer_function():

    test_wavelength = 300  # nm
    smbh_mass_exponent = 8.0
    corona_height = 10
    inclination_angle = 30
    axis_offset_in_gravitational_radii = 0
    angle_offset_in_degrees = 0
    height_array = None
    albedo_array = None

    # to test reasonable transfer functions, we must make a reasonable disk
    disk_dict = create_maps(smbh_mass_exponent, 0, 500, inclination_angle, 500)

    temp_array = disk_dict["temp_array"]
    radii_array = disk_dict["radii_array"]
    phi_array = disk_dict["phi_array"]
    g_array = disk_dict["g_array"]
    inclination_angle = disk_dict["inclination_angle"]
    smbh_mass_exponent = disk_dict["smbh_mass_exp"]
    corona_height = disk_dict["corona_height"]

    transfer_function_test_no_ml = construct_accretion_disk_transfer_function(
        test_wavelength,
        temp_array,
        radii_array,
        phi_array,
        g_array,
        inclination_angle,
        smbh_mass_exponent,
        corona_height,
        axis_offset_in_gravitational_radii=axis_offset_in_gravitational_radii,
        angle_offset_in_degrees=angle_offset_in_degrees,
        height_array=height_array,
        albedo_array=albedo_array,
    )

    magnification_array_identity = np.ones((1000, 1000))

    redshift_l = 0.5
    redshift_s = 2.0
    relative_orientation_1 = 0
    mean_microlens_mass_in_kg = 0.3 * const.M_sun.to(u.kg)
    number_of_microlens_einstein_radii = 5

    orientation_1 = 0
    orientation_2 = 45

    # show we can reclaim the same transfer function when identity is used
    transfer_function_test_id_ml = calculate_microlensed_transfer_function(
        magnification_array_identity,
        redshift_l,
        redshift_s,
        test_wavelength,
        temp_array,
        radii_array,
        phi_array,
        g_array,
        inclination_angle,
        smbh_mass_exponent,
        corona_height,
        mean_microlens_mass_in_kg=mean_microlens_mass_in_kg,
        number_of_microlens_einstein_radii=number_of_microlens_einstein_radii,
        number_of_smbh_gravitational_radii=1000,
        relative_orientation=orientation_1,
    )

    transfer_function_test_id_ml_rotate = calculate_microlensed_transfer_function(
        magnification_array_identity,
        redshift_l,
        redshift_s,
        test_wavelength,
        temp_array,
        radii_array,
        phi_array,
        g_array,
        inclination_angle,
        smbh_mass_exponent,
        corona_height,
        mean_microlens_mass_in_kg=mean_microlens_mass_in_kg,
        number_of_microlens_einstein_radii=number_of_microlens_einstein_radii,
        number_of_smbh_gravitational_radii=1000,
        relative_orientation=orientation_2,
    )

    data_to_construct_transfer_function = calculate_microlensed_transfer_function(
        magnification_array_identity,
        redshift_l,
        redshift_s,
        test_wavelength,
        temp_array,
        radii_array,
        phi_array,
        g_array,
        inclination_angle,
        smbh_mass_exponent,
        corona_height,
        mean_microlens_mass_in_kg=mean_microlens_mass_in_kg,
        number_of_microlens_einstein_radii=number_of_microlens_einstein_radii,
        number_of_smbh_gravitational_radii=1000,
        relative_orientation=orientation_2,
        return_response_array_and_lags=True,
    )
    descaled_data = calculate_microlensed_transfer_function(
        magnification_array_identity,
        redshift_l,
        redshift_s,
        test_wavelength,
        temp_array,
        radii_array,
        phi_array,
        g_array,
        inclination_angle,
        smbh_mass_exponent,
        corona_height,
        mean_microlens_mass_in_kg=mean_microlens_mass_in_kg,
        number_of_microlens_einstein_radii=number_of_microlens_einstein_radii,
        number_of_smbh_gravitational_radii=1000,
        relative_orientation=orientation_2,
        return_descaled_response_array_and_lags=True,
    )

    tau_axis_ml = np.linspace(
        0, len(transfer_function_test_id_ml) - 1, len(transfer_function_test_id_ml)
    )

    tau_axis_no_ml = np.linspace(
        0, len(transfer_function_test_no_ml) - 1, len(transfer_function_test_no_ml)
    )
    mean_tau_no_ml = np.sum(transfer_function_test_no_ml * tau_axis_no_ml) / np.sum(
        transfer_function_test_no_ml
    )
    mean_tau_id_ml = np.sum(transfer_function_test_id_ml * tau_axis_ml) / np.sum(
        transfer_function_test_id_ml
    )
    mean_tau_id_ml_rotated = np.sum(
        transfer_function_test_id_ml_rotate * tau_axis_ml
    ) / np.sum(transfer_function_test_id_ml_rotate)

    # note there is some rounding when rescale and rotate are used and inverted
    tolerance = mean_tau_no_ml / 20
    assert abs(mean_tau_no_ml - mean_tau_id_ml) <= tolerance
    current_diff = mean_tau_no_ml - mean_tau_id_ml_rotated
    assert abs(current_diff) <= tolerance

    assert len(data_to_construct_transfer_function) == 4
    assert np.size(data_to_construct_transfer_function[0]) == np.size(
        data_to_construct_transfer_function[1]
    )
    assert len(descaled_data) == 4
    assert np.size(descaled_data[0]) == np.size(descaled_data[1])

    # test the magnification crop
    crop_data = calculate_microlensed_transfer_function(
        magnification_array_identity,
        redshift_l,
        redshift_s,
        test_wavelength,
        temp_array,
        radii_array,
        phi_array,
        g_array,
        inclination_angle,
        smbh_mass_exponent,
        corona_height,
        mean_microlens_mass_in_kg=mean_microlens_mass_in_kg,
        number_of_microlens_einstein_radii=number_of_microlens_einstein_radii,
        number_of_smbh_gravitational_radii=1000,
        relative_orientation=orientation_2,
        return_magnification_map_crop=True,
    )


def test_generate_drw_signal():
    maximum_time = 2000
    time_step = 2
    sf_inf_1 = 100
    tau_drw_1 = 100
    random_seed = 17
    sf_inf_2 = 400
    tau_drw_2 = 10

    drw_1 = generate_drw_signal(
        maximum_time, time_step, sf_inf_1, tau_drw_1, random_seed=random_seed
    )

    drw_2 = generate_drw_signal(
        maximum_time, time_step, sf_inf_2, tau_drw_2, random_seed=random_seed
    )

    assert len(drw_1) == len(drw_2)
    npt.assert_almost_equal(np.mean(drw_1), np.mean(drw_2))

    random_point_1 = np.random.randint(500)
    random_point_2 = np.random.randint(500)

    if random_point_1 == random_point_2:  # pragma: no cover
        random_point_2 -= 1

    assert drw_1[random_point_1] != drw_1[random_point_2]
    assert drw_2[random_point_1] != drw_2[random_point_2]


def test_generate_signal_from_psd():
    length_of_light_curve = 1000
    frequencies = np.linspace(
        1 / length_of_light_curve, 1 / 4, length_of_light_curve + 1
    )
    psd_flat_spectrum = np.ones(np.shape(frequencies))
    psd_power_law = frequencies ** (-2)
    random_seed = 33

    time_ax, lc_flat_spectrum = generate_signal_from_psd(
        length_of_light_curve,
        psd_flat_spectrum,
        frequencies,
        random_seed=random_seed,
    )

    time_ax, lc_power_spectrum = generate_signal_from_psd(
        length_of_light_curve,
        psd_power_law,
        frequencies,
        random_seed=random_seed,
    )

    assert len(lc_flat_spectrum) == len(lc_power_spectrum)
    assert lc_flat_spectrum[3] != lc_flat_spectrum[0]


def test_generate_snapshots_of_radiation_pattern():

    test_wavelength = 700  # nm
    smbh_mass_exponent = 8.0
    corona_height = 10
    inclination_angle = 30

    # to test radiation patterns, we must make a reasonable disk
    disk_dict = create_maps(smbh_mass_exponent, 0, 1000, inclination_angle, 1000)

    temp_array = disk_dict["temp_array"]
    radii_array = disk_dict["radii_array"]
    phi_array = disk_dict["phi_array"]
    g_array = disk_dict["g_array"]
    inclination_angle = disk_dict["inclination_angle"]
    smbh_mass_exponent = disk_dict["smbh_mass_exp"]
    corona_height = disk_dict["corona_height"]

    static_disk_radiation = planck_law(temp_array, test_wavelength)

    snapshot_list = [0, 10, 100, 500]
    maximum_time = 1000
    time_step = 1
    sf_inf = 50
    tau_drw = 100
    random_seed = 40

    driving_signal_fractional_strength_half = 0.5
    driving_signal_fractional_strength_zero = 0.0
    driving_signal_fractional_strength_all = 1.0

    drw_signal = generate_drw_signal(
        maximum_time, time_step, sf_inf, tau_drw, random_seed=random_seed
    )

    snapshots_half = generate_snapshots_of_radiation_pattern(
        test_wavelength,
        snapshot_list,
        temp_array,
        radii_array,
        phi_array,
        g_array,
        smbh_mass_exponent,
        drw_signal,
        driving_signal_fractional_strength_half,
        corona_height,
        inclination_angle,
    )

    snapshots_zero = generate_snapshots_of_radiation_pattern(
        test_wavelength,
        snapshot_list,
        temp_array,
        radii_array,
        phi_array,
        g_array,
        smbh_mass_exponent,
        drw_signal,
        driving_signal_fractional_strength_zero,
        corona_height,
        inclination_angle,
    )

    snapshots_all = generate_snapshots_of_radiation_pattern(
        test_wavelength,
        snapshot_list,
        temp_array,
        radii_array,
        phi_array,
        g_array,
        smbh_mass_exponent,
        drw_signal,
        driving_signal_fractional_strength_all,
        corona_height,
        inclination_angle,
    )
    snapshots_short_driving_signal = generate_snapshots_of_radiation_pattern(
        test_wavelength,
        snapshot_list,
        temp_array,
        radii_array,
        phi_array,
        g_array,
        smbh_mass_exponent,
        drw_signal[:5],
        driving_signal_fractional_strength_all,
        corona_height,
        inclination_angle,
    )

    assert np.shape(snapshots_half) == np.shape(snapshots_zero)
    assert np.shape(snapshots_half) == np.shape(snapshots_all)

    for jj in range(len(snapshot_list)):
        # use abs to show each pixel doesn't deviate from static case
        assert np.sum(abs(snapshots_zero[jj] - static_disk_radiation)) == 0
        # like before, but show both variabile disks do deviate
        assert np.sum(abs(snapshots_half[jj] - static_disk_radiation)) > 0
        assert np.sum(abs(snapshots_all[jj] - static_disk_radiation)) > 0

    signal_increasing_power = np.linspace(1, 10000, 10000)

    snapshots_rising_power = generate_snapshots_of_radiation_pattern(
        test_wavelength,
        snapshot_list,
        temp_array,
        radii_array,
        phi_array,
        g_array,
        smbh_mass_exponent,
        signal_increasing_power,
        driving_signal_fractional_strength_all,
        corona_height,
        inclination_angle,
    )

    for jj in range(len(snapshot_list) - 1):
        assert np.sum(snapshots_rising_power[jj + 1] - snapshots_rising_power[jj]) > 0


def test_project_blr_to_source_plane():

    # create some relatively large density region
    # want 1000 in the R direction, 50 in the Z direction so the
    # loop over each slab doesn't take long

    max_radius = 500
    max_height = 500

    r_points = 20
    z_points = 10

    r_step = max_radius // r_points
    z_step = max_height // z_points

    test_density = np.ones((r_points, z_points))

    radial_vel_grid = np.zeros(np.shape(test_density))
    vertical_vel_grid = np.zeros(np.shape(test_density))
    vertical_vel_grid[:, 4:] = 0.3
    vertical_vel_grid[:, 8:] = 0.6

    radii = np.linspace(0, max_radius, r_points)
    heights = np.linspace(0, max_height, z_points)

    R, Z = np.meshgrid(radii, heights, indexing="ij")
    distance_grid = (R**2 + Z**2) ** 0.5
    weighting = np.exp(-((distance_grid - 500) ** 2))

    smbh_exp = 8.0
    inclination = 0

    with pytest.raises(AssertionError):
        project_blr_to_source_plane(
            test_density,
            vertical_vel_grid,
            radial_vel_grid,
            -3,
            smbh_exp,
            velocity_range=[-1, 1],
            weighting_grid=weighting,
            radial_resolution=r_step,
            vertical_resolution=z_step,
        )
    with pytest.raises(AssertionError):
        project_blr_to_source_plane(
            test_density,
            vertical_vel_grid,
            radial_vel_grid,
            100,
            smbh_exp,
            velocity_range=[-1, 1],
            weighting_grid=weighting,
            radial_resolution=r_step,
            vertical_resolution=z_step,
        )
    with pytest.raises(AssertionError):
        project_blr_to_source_plane(
            test_density,
            vertical_vel_grid.T,
            radial_vel_grid,
            inclination,
            smbh_exp,
            velocity_range=[-1, 1],
            weighting_grid=weighting,
            radial_resolution=r_step,
            vertical_resolution=z_step,
        )
    with pytest.raises(AssertionError):
        project_blr_to_source_plane(
            test_density,
            vertical_vel_grid,
            radial_vel_grid.T,
            inclination,
            smbh_exp,
            velocity_range=[-1, 1],
            weighting_grid=weighting,
            radial_resolution=r_step,
            vertical_resolution=z_step,
        )
    with pytest.raises(AssertionError):
        project_blr_to_source_plane(
            test_density,
            vertical_vel_grid,
            radial_vel_grid,
            inclination,
            smbh_exp,
            velocity_range=[-1, 1],
            weighting_grid=weighting.T,
            radial_resolution=r_step,
            vertical_resolution=z_step,
        )

    test_projection_1 = project_blr_to_source_plane(
        test_density,
        vertical_vel_grid,
        radial_vel_grid,
        inclination,
        smbh_exp,
        velocity_range=[-1, 1],
        weighting_grid=weighting,
        radial_resolution=r_step,
        vertical_resolution=z_step,
    )[0]

    # since a cylinder of constant density is being projected, a projected
    # row in the center will have a greater value than an edge row in all cases
    assert np.sum(test_projection_1[0]) < np.sum(test_projection_1[r_points])
    assert np.sum(test_projection_1[-1]) < np.sum(test_projection_1[r_points])
    assert np.sum(test_projection_1.T[0]) < np.sum(test_projection_1[r_points])
    assert np.sum(test_projection_1.T[-1]) < np.sum(test_projection_1[r_points])

    # if inclined, the edge row of values on the far side will be greater than
    # the zeros on the near side
    inclination = 20
    test_projection_2 = project_blr_to_source_plane(
        test_density,
        vertical_vel_grid,
        radial_vel_grid,
        inclination,
        smbh_exp,
        velocity_range=[-1, 1],
        weighting_grid=None,
        radial_resolution=r_step,
        vertical_resolution=z_step,
    )[0]

    # note the projection is enlarged w.r.t. the original source to capture
    # the whole projection
    projection_padding = np.size(test_projection_2, 0) // 2 - r_points

    assert np.sum(test_projection_2[projection_padding + 10]) < np.sum(
        test_projection_2[-(projection_padding + 10)]
    )
    # if transposed, both sides should remain the same
    assert np.sum(test_projection_2.T[projection_padding]) == np.sum(
        test_projection_2.T[-projection_padding]
    )
    assert np.sum(test_projection_2.T[(projection_padding + 1)]) == np.sum(
        test_projection_2.T[-(projection_padding + 2)]
    )

    # test that we can only select certain velocities (receeding in this case)
    inclination = 60
    velocity_range_receeding = [-1, -0.001]
    vertical_vel_grid = np.zeros(np.shape(test_density))

    test_projection_3_receeding = project_blr_to_source_plane(
        test_density,
        vertical_vel_grid,
        radial_vel_grid,
        inclination,
        smbh_exp,
        velocity_range=velocity_range_receeding,
        weighting_grid=weighting,
        radial_resolution=r_step,
        vertical_resolution=z_step,
    )[0]

    velocity_range_approaching = [0.001, 1]

    test_projection_3_approaching = project_blr_to_source_plane(
        test_density,
        vertical_vel_grid,
        radial_vel_grid,
        inclination,
        smbh_exp,
        velocity_range=velocity_range_approaching,
        weighting_grid=weighting,
        radial_resolution=r_step,
        vertical_resolution=z_step,
    )[0]

    projection_padding = np.size(test_projection_3_approaching, 0) // 2 - r_points
    middle = np.size(test_projection_3_approaching, 0) // 2

    # when inclined, the velocity is a mix of each direction.
    # in constructing this there was no radial or vertical velocity added.
    # therefore, the los velocity depends on phi coord.

    # test phi dependence on lhs and rhs of the black hole
    assert np.sum(test_projection_3_approaching[:, middle + 5]) < np.sum(
        test_projection_3_receeding[:, middle + 5]
    )
    assert np.sum(test_projection_3_approaching[:, middle - 5]) > np.sum(
        test_projection_3_receeding[:, middle - 5]
    )

    # test outflow viewed perfectly face on by projecting everything to approaching grid
    inclination = 0
    for jj in range(np.size(vertical_vel_grid, 1)):
        vertical_vel_grid[:, jj] = 0.5 * jj / np.size(vertical_vel_grid, 1)

    test_projection_4_receeding = project_blr_to_source_plane(
        test_density,
        vertical_vel_grid,
        radial_vel_grid,
        inclination,
        smbh_exp,
        velocity_range=velocity_range_receeding,
        weighting_grid=weighting,
        radial_resolution=r_step,
        vertical_resolution=z_step,
    )[0]
    test_projection_4_approaching = project_blr_to_source_plane(
        test_density,
        vertical_vel_grid,
        radial_vel_grid,
        inclination,
        smbh_exp,
        velocity_range=velocity_range_approaching,
        weighting_grid=weighting,
        radial_resolution=r_step,
        vertical_resolution=z_step,
    )[0]

    assert np.sum(test_projection_4_approaching) > 0
    assert np.sum(test_projection_4_receeding) == 0


def test_calculate_blr_transfer_function():

    max_radius = 100
    max_height = 100

    r_points = 20
    z_points = 10

    r_step = max_radius // r_points
    z_step = max_height // z_points

    test_density = np.ones((r_points, z_points))

    radial_vel_grid = np.zeros(np.shape(test_density))
    vertical_vel_grid = np.zeros(np.shape(test_density))
    vertical_vel_grid[:, 4:] = 0.3
    vertical_vel_grid[:, 8:] = 0.6

    inclination_angle = 0
    smbh_mass_exponent = 10

    with pytest.raises(AssertionError):
        calculate_blr_transfer_function(
            test_density,
            vertical_vel_grid,
            radial_vel_grid,
            -2,
            smbh_mass_exponent,
            velocity_range=[-1, 1],
            weighting_grid=None,
            radial_resolution=r_step,
            vertical_resolution=z_step,
        )
    with pytest.raises(AssertionError):
        calculate_blr_transfer_function(
            test_density,
            vertical_vel_grid,
            radial_vel_grid,
            50000,
            smbh_mass_exponent,
            velocity_range=[-1, 1],
            weighting_grid=None,
            radial_resolution=r_step,
            vertical_resolution=z_step,
        )
    with pytest.raises(AssertionError):
        calculate_blr_transfer_function(
            test_density,
            vertical_vel_grid.T,
            radial_vel_grid,
            -2,
            smbh_mass_exponent,
            velocity_range=[-1, 1],
            weighting_grid=None,
            radial_resolution=r_step,
            vertical_resolution=z_step,
        )
    with pytest.raises(AssertionError):
        calculate_blr_transfer_function(
            test_density,
            vertical_vel_grid,
            radial_vel_grid.T,
            -2,
            smbh_mass_exponent,
            velocity_range=[-1, 1],
            weighting_grid=None,
            radial_resolution=r_step,
            vertical_resolution=z_step,
        )
    with pytest.raises(AssertionError):
        calculate_blr_transfer_function(
            test_density,
            vertical_vel_grid,
            radial_vel_grid,
            -2,
            smbh_mass_exponent,
            velocity_range=[-1, 1],
            weighting_grid=test_density.T,
            radial_resolution=r_step,
            vertical_resolution=z_step,
        )

    face_on_blr_tf_1 = calculate_blr_transfer_function(
        test_density,
        vertical_vel_grid,
        radial_vel_grid,
        inclination_angle,
        smbh_mass_exponent,
        velocity_range=[-1, 1],
        weighting_grid=None,
        radial_resolution=r_step,
        vertical_resolution=z_step,
    )

    assert face_on_blr_tf_1.ndim == 1

    tau_ax = np.linspace(0, np.size(face_on_blr_tf_1) - 1, np.size(face_on_blr_tf_1))

    mean_response_1 = np.sum(tau_ax * face_on_blr_tf_1)

    assert mean_response_1 > 0
    npt.assert_almost_equal(np.sum(face_on_blr_tf_1), 1)

    # only select face on components (which will be all of them in this case)
    face_on_blr_tf_2 = calculate_blr_transfer_function(
        test_density,
        vertical_vel_grid,
        radial_vel_grid,
        inclination_angle,
        smbh_mass_exponent,
        velocity_range=[0, 1],
        weighting_grid=None,
        radial_resolution=r_step,
        vertical_resolution=z_step,
    )

    mean_response_2 = np.sum(tau_ax * face_on_blr_tf_2)

    npt.assert_almost_equal(mean_response_2, mean_response_1)

    # test that we can remove all components by selecting receeding components only
    face_on_blr_tf_3 = calculate_blr_transfer_function(
        test_density,
        vertical_vel_grid,
        radial_vel_grid,
        inclination_angle,
        smbh_mass_exponent,
        velocity_range=[-1, -0.0001],
        weighting_grid=None,
        radial_resolution=r_step,
        vertical_resolution=z_step,
    )

    mean_response_3 = np.sum(tau_ax * face_on_blr_tf_3)
    assert mean_response_3 == 0


def test_required_velocity_maximum():

    # test with easy numbers
    rest_frame_emitted_wavelength = 100
    passband_minimum = 200
    # try an absurdly high upper limit
    passband_maximum = 10**8

    redshift_0 = 0
    redshift_1 = 1

    vel_range_0 = determine_emission_line_velocities(
        rest_frame_emitted_wavelength, passband_minimum, passband_maximum, redshift_0
    )

    vel_range_1 = determine_emission_line_velocities(
        rest_frame_emitted_wavelength, passband_minimum, passband_maximum, redshift_1
    )

    assert vel_range_0[0] < vel_range_1[0]
    assert vel_range_1[1] == 0
    assert vel_range_0[1] < 0
    assert vel_range_0[0] < 0
    npt.assert_almost_equal(vel_range_0[0], vel_range_1[0])

    with pytest.raises(AssertionError):
        determine_emission_line_velocities(
            rest_frame_emitted_wavelength, passband_minimum, passband_maximum, -1
        )

    # test with required blueshift

    rest_frame_emitted_wavelength = 1000
    passband_minimum = 10 ** (-8)
    passband_maximum = 200

    vel_range_blueshift = determine_emission_line_velocities(
        rest_frame_emitted_wavelength, passband_minimum, passband_maximum, redshift_0
    )

    assert vel_range_blueshift[0] > 0
    assert vel_range_blueshift[1] > 0

    npt.assert_almost_equal(vel_range_blueshift[1], 1)

    # test with an extremely broad filter (wavelengths 10^(-8) - 10^(8))
    passband_maximum = 10**8

    vel_range_all = determine_emission_line_velocities(
        rest_frame_emitted_wavelength, passband_minimum, passband_maximum, redshift_0
    )

    npt.assert_almost_equal(vel_range_all[1], 1)
    npt.assert_almost_equal(vel_range_all[0], -1)


def test_convolve_signal_with_transfer_function():
    sample_transfer_function = [0, 0, 0, 20, 100, 80, 60, 40, 10, 0]
    smbh_mass_exp = 9.8
    driving_signal = np.sin(np.linspace(1, 1000, 1000))
    times, convolved_signal = convolve_signal_with_transfer_function(
        smbh_mass_exp=smbh_mass_exp,
        driving_signal=driving_signal,
        transfer_function=sample_transfer_function,
    )
    daily_signal = np.interp(np.linspace(1, 1000, 1000), times, convolved_signal)
    assert np.sum(abs(driving_signal - daily_signal)) != 0


def test_convert_speclite_filter_to_wavelength_range():
    # collect FilterResponse objects
    my_filters = load_filters("lsst2023-*")
    # try one FilterResponse
    one_filter = my_filters[0]
    # try one string
    one_string = "lsst2023-r"
    # try a list of strings
    string_list = ["lsst2023-g", "lsst2023-y"]

    output_one = convert_speclite_filter_to_wavelength_range(one_filter)
    output_str = convert_speclite_filter_to_wavelength_range(one_string)
    output_list = convert_speclite_filter_to_wavelength_range(string_list)

    # try an invalid string
    assert not convert_speclite_filter_to_wavelength_range("not_a_filter")
    # try something that's not a string, list, or FilterResponse
    assert not convert_speclite_filter_to_wavelength_range({"not_a_filter": 4})
    string_list_bad_names = ["one_filter", "two_filter", "red_filter", "blue"]
    assert not convert_speclite_filter_to_wavelength_range(string_list_bad_names)

    assert isinstance(output_one, list)
    assert isinstance(output_str, list)
    assert isinstance(output_list, list)
    assert isinstance(output_one[0], list)
    assert isinstance(output_str[0], list)
    assert isinstance(output_list[0], list)

    assert output_one[0][0] < output_one[0][1]
    assert output_str[0][0] < output_str[0][1]

    for item in output_list:
        assert item[0] < item[1]
