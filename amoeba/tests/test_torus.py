import pytest
import numpy as np
from amoeba.Classes.blr_streamline import Streamline
from amoeba.Classes.torus import Torus
from amoeba.Classes.flux_projection import FluxProjection
import astropy.units as u
import numpy.testing as npt


class TestTorus:

    def setup_method(self):

        smbh_mass_exp = 7.28384
        launch_radius = 500  # Rg
        launch_theta = 0  # degrees
        max_height = 1000  # Rg
        height_step = 200
        rest_frame_wavelength_in_nm = 600
        redshift_source = 1.1
        characteristic_distance = max_height // 5
        asymptotic_poloidal_velocity = 0.2
        poloidal_launch_velocity = 10**-5

        self.test_torus_streamline = Streamline(
            launch_radius,
            launch_theta,
            max_height,
            characteristic_distance,
            asymptotic_poloidal_velocity,
            poloidal_launch_velocity=poloidal_launch_velocity,
            height_step=height_step,
        )

        launch_theta_angled = 45
        self.test_torus_streamline_angled = Streamline(
            launch_radius,
            launch_theta_angled,
            max_height,
            characteristic_distance,
            asymptotic_poloidal_velocity,
            poloidal_launch_velocity=poloidal_launch_velocity,
            height_step=height_step,
        )

        self.torus_rect = Torus(
            smbh_mass_exp,
            max_height,
            redshift_source,
            height_step=height_step,
            power_law_density_dependence=-2,
        )

        self.torus_angled = Torus(
            smbh_mass_exp,
            max_height,
            redshift_source,
            height_step=height_step,
            power_law_density_dependence=-2,
        )

        self.torus_rect.add_streamline_bounded_region(self.test_torus_streamline)

        self.torus_angled.add_streamline_bounded_region(
            self.test_torus_streamline_angled
        )

        wavelengths = [100, 500, 10000]
        extinction_coefficients = [10**-5, 10**-20, 0]
        kwarg_dict = {
            "smbh_mass_exp": smbh_mass_exp,
            "redshift_source": redshift_source,
            "max_height": max_height,
            "rest_frame_wavelengths": wavelengths,
            "extinction_coefficients": extinction_coefficients,
            "height_step": height_step,
        }

        self.preset_torus = Torus(**kwarg_dict)
        assert self.preset_torus.add_streamline_bounded_region(
            self.test_torus_streamline
        )
        assert isinstance(self.preset_torus, Torus)

    def test_define_extinction_coefficients(self):

        wavelengths = [100, 500, 10000]
        extinction_coefficients = [10**-5, 10**-20, 0]

        self.torus_rect.define_extinction_coefficients(
            rest_frame_wavelengths=wavelengths,
            extinction_coefficients=extinction_coefficients,
        )

        assert self.torus_angled.rest_frame_wavelengths is None
        assert self.torus_angled.extinction_coefficients is None

        assert self.torus_rect.rest_frame_wavelengths is wavelengths
        assert self.torus_rect.extinction_coefficients is extinction_coefficients

    def test_interpolate_to_extinction_at_wavelength(self):

        wavelengths = [100, 500, 10000]
        extinction_coefficients = [10**-5, 10**-20, 0]

        self.torus_angled.define_extinction_coefficients(
            rest_frame_wavelengths=wavelengths,
            extinction_coefficients=extinction_coefficients,
        )

        rest_wavelength = 400

        target_wavelength = rest_wavelength * (1 + self.torus_rect.redshift_source)

        assert not self.torus_rect.interpolate_to_extinction_at_wavelength(
            target_wavelength
        )

        my_interpolated_extinction = (
            self.torus_angled.interpolate_to_extinction_at_wavelength(target_wavelength)
        )

        delta_lam = rest_wavelength - wavelengths[0]
        lin_slope = (extinction_coefficients[1] - extinction_coefficients[0]) / (
            wavelengths[1] - wavelengths[0]
        )

        expected_extinction = extinction_coefficients[0] + delta_lam * lin_slope

        npt.assert_almost_equal(expected_extinction, my_interpolated_extinction)

    def test_project_density_to_source_plane(self):

        test_inc_1 = 0
        test_inc_2 = 30

        projection_rect_face_on = self.torus_rect.project_density_to_source_plane(
            test_inc_1
        )

        projection_rect_finclined = self.torus_rect.project_density_to_source_plane(
            test_inc_2
        )

        projection_angled_face_on = self.torus_angled.project_density_to_source_plane(
            test_inc_1
        )

        projection_angled_finclined = self.torus_angled.project_density_to_source_plane(
            test_inc_2
        )

        assert type(projection_rect_face_on) == np.ndarray

    def test_project_extinction_to_source_plane(self):

        unobscured_inclination = 0
        obscured_inclination = 60

        wavelengths = np.asarray([10, 500, 10000])
        extinction_coefficients = np.asarray([10**-12, 10**-12, 10**-10])

        self.torus_angled.define_extinction_coefficients(
            rest_frame_wavelengths=wavelengths,
            extinction_coefficients=extinction_coefficients,
        )

        wavelength_optically_thin = 110
        wavelength_optically_thick = 9000

        thin_no_obscure = self.torus_angled.project_extinction_to_source_plane(
            unobscured_inclination,
            wavelength_optically_thin,
        )

        thin_obscured = self.torus_angled.project_extinction_to_source_plane(
            obscured_inclination,
            wavelength_optically_thin,
        )

        thick_no_obscure = self.torus_angled.project_extinction_to_source_plane(
            unobscured_inclination,
            wavelength_optically_thick,
        )

        thick_obscured = self.torus_angled.project_extinction_to_source_plane(
            obscured_inclination,
            wavelength_optically_thick,
        )

        assert isinstance(thin_no_obscure, FluxProjection)

        mask_no_obscure = thin_no_obscure.flux_array > 0
        mask_obscured = thin_obscured.flux_array > 0

        assert thin_no_obscure.redshift_source == self.torus_rect.redshift_source

        assert np.sum(thin_no_obscure.flux_array) < np.sum(thick_no_obscure.flux_array)
        assert np.sum(thin_obscured.flux_array) < np.sum(thick_obscured.flux_array)

        avg_mag_reduction_opt_thin = np.sum(thin_obscured.flux_array) / np.sum(
            mask_obscured
        )
        avg_mag_reduction_opt_thick = np.sum(thick_obscured.flux_array) / np.sum(
            mask_obscured
        )

        assert avg_mag_reduction_opt_thin < avg_mag_reduction_opt_thick
