import pytest
from amoeba.Classes.flux_projection import FluxProjection
from amoeba.Classes.accretion_disk import AccretionDisk
from amoeba.Classes.blr import BroadLineRegion
from amoeba.Classes.blr_streamline import Streamline
from amoeba.Util.util import create_maps
import astropy.units as u

import numpy as np


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
    Om0 = 0.3
    H0 = 70

    my_gaussian_projection = FluxProjection(
        flux_map,
        observer_frame_wavelength_in_nm,
        smbh_mass_exp,
        redshift_source,
        r_out_in_gravitational_radii,
        inclination_angle,
        Om0=0.3,
        H0=70,
    )

    # check it sum naturally
    assert np.sum(flux_map) == my_gaussian_projection.total_flux
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
    assert isinstance(projection.observer_frame_wavelength_in_nm, (u.Quantity, float, list, np.ndarray))
    assert projection.smbh_mass_exp == smbh_mass_exp
    assert projection.r_out_in_gravitational_radii > max_height + launch_radius
    assert projection.inclination_angle == inclination_angle
