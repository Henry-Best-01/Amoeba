import pytest
import numpy as np
from amoeba.Classes.blr_streamline import Streamline
from amoeba.Classes.blr import BroadLineRegion
from amoeba.Classes.flux_projection import FluxProjection
import astropy.units as u
import numpy.testing as npt


class TestBlr:

    def setup_method(self):

        smbh_mass_exp = 7.28384
        launch_radius = 500  # Rg
        launch_theta = 0  # degrees
        max_height = 1000  # Rg
        height_step = 200
        radial_step = 50
        rest_frame_wavelength_in_nm = 600
        redshift_source = 1.1
        characteristic_distance = max_height // 5
        asymptotic_poloidal_velocity = 0.2
        poloidal_launch_velocity = 10**-5

        self.test_blr_streamline = Streamline(
            launch_radius,
            launch_theta,
            max_height,
            characteristic_distance,
            asymptotic_poloidal_velocity,
            poloidal_launch_velocity=poloidal_launch_velocity,
            height_step=height_step,
        )

        launch_theta_angled = 45
        self.test_blr_streamline_angled = Streamline(
            launch_radius,
            launch_theta_angled,
            max_height,
            characteristic_distance,
            asymptotic_poloidal_velocity,
            poloidal_launch_velocity=poloidal_launch_velocity,
            height_step=height_step,
        )

        self.blr = BroadLineRegion(
            smbh_mass_exp,
            max_height,
            rest_frame_wavelength_in_nm,
            redshift_source,
            height_step=height_step,
            radial_step=radial_step,
        )

        self.blr.add_streamline_bounded_region(
            self.test_blr_streamline,
            self.test_blr_streamline_angled,
        )

    def test_add_streamline_bounded_region(self):

        smbh_mass_exp = 7.28384
        launch_radius = 500  # Rg
        launch_theta = 0  # degrees
        max_height = 1000  # Rg
        characteristic_distance = max_height // 5
        asymptotic_poloidal_velocity = 0.2
        poloidal_launch_velocity = 10**-5

        self.bad_streamline_1 = Streamline(
            launch_radius,
            launch_theta,
            max_height - 10,
            characteristic_distance,
            asymptotic_poloidal_velocity,
            poloidal_launch_velocity=poloidal_launch_velocity,
        )
        self.bad_streamline_2 = Streamline(
            launch_radius,
            launch_theta,
            max_height,
            characteristic_distance,
            asymptotic_poloidal_velocity,
            height_step=11,
            poloidal_launch_velocity=poloidal_launch_velocity,
        )
        self.bad_streamline_3 = Streamline(
            launch_radius,
            launch_theta,
            max_height,
            characteristic_distance,
            asymptotic_poloidal_velocity,
            height_step=11,
            poloidal_launch_velocity=poloidal_launch_velocity,
        )

        with pytest.raises(AssertionError):
            self.blr.add_streamline_bounded_region(
                self.test_blr_streamline, self.bad_streamline_1
            )
        with pytest.raises(AssertionError):
            self.blr.add_streamline_bounded_region(
                self.test_blr_streamline, self.bad_streamline_2
            )
        with pytest.raises(AssertionError):
            self.blr.add_streamline_bounded_region(
                self.bad_streamline_3, self.bad_streamline_2
            )
        with pytest.raises(AssertionError):
            self.blr.add_streamline_bounded_region(
                self.test_blr_streamline,
                self.test_blr_streamline_angled,
                density_initial_weighting=0,
            )

    def test_project_blr_density_to_source_plane(self):

        face_on_inclination = 0
        face_on_projection_of_density = self.blr.project_blr_density_to_source_plane(
            face_on_inclination
        )

        assert face_on_projection_of_density.ndim == 2
        assert np.max(face_on_projection_of_density) > 0

        inclination = 40
        inclined_projection_of_density = self.blr.project_blr_density_to_source_plane(
            inclination
        )

        # these projections have different shapes due to the enlargement from inclination
        assert np.shape(face_on_projection_of_density) != np.shape(
            inclined_projection_of_density
        )
        assert np.size(face_on_projection_of_density) < np.size(
            inclined_projection_of_density
        )

    def test_project_blr_total_intensity(self):

        inclination = 0
        efficiency_array = np.ones(self.blr.blr_array_shape)
        twice_the_efficiency = efficiency_array * 2

        flux_projection_1 = self.blr.project_blr_total_intensity(
            inclination,
            emission_efficiency_array=efficiency_array,
        )
        flux_projection_2 = self.blr.project_blr_total_intensity(
            inclination,
            emission_efficiency_array=twice_the_efficiency,
        )

        assert isinstance(flux_projection_1, FluxProjection)

        assert flux_projection_1.total_flux * 2 == flux_projection_2.total_flux
        assert np.shape(flux_projection_1.flux_array) == np.shape(
            flux_projection_2.flux_array
        )
        assert flux_projection_1.observer_frame_wavelength_in_nm[0] == 0
        assert flux_projection_1.observer_frame_wavelength_in_nm[1] == np.inf
        assert flux_projection_2.observer_frame_wavelength_in_nm[0] == 0
        assert flux_projection_2.observer_frame_wavelength_in_nm[1] == np.inf

    def test_project_blr_intensity_over_velocity_range(self):

        inclination = 35
        efficiency_array = np.ones(self.blr.blr_array_shape)
        receeding_velocity_range = [-1, 0]
        approaching_velocity_range = [0, 1]
        total_velocity_range = [-1, 1]
        # velocity ranges are defined as the joint region defined by (v>=min, v<max)
        no_velocity_range = [0, 0]

        receeding_projection = self.blr.project_blr_intensity_over_velocity_range(
            inclination,
            velocity_range=receeding_velocity_range,
            emission_efficiency_array=efficiency_array,
        )

        approaching_projection = self.blr.project_blr_intensity_over_velocity_range(
            inclination,
            velocity_range=approaching_velocity_range,
            emission_efficiency_array=efficiency_array,
        )

        total_projection = self.blr.project_blr_intensity_over_velocity_range(
            inclination,
            velocity_range=total_velocity_range,
            emission_efficiency_array=efficiency_array,
        )

        no_projection = self.blr.project_blr_intensity_over_velocity_range(
            inclination,
            velocity_range=no_velocity_range,
            emission_efficiency_array=efficiency_array,
        )

        assert isinstance(receeding_projection, FluxProjection)
        # more material should be approaching
        assert approaching_projection.total_flux > receeding_projection.total_flux
        # taking the combined flux should equal the total flux
        npt.assert_almost_equal(
            total_projection.total_flux,
            approaching_projection.total_flux + receeding_projection.total_flux,
            3,
        )
        # show we have flux unless we exclude all velocities
        assert no_projection.total_flux == 0
        assert receeding_projection.total_flux > 0

    def test_calculate_blr_scattering_transfer_function(self):
        inclination_face_on = 0
        inclination_inclined = 40

        scattering_tf_face_on = self.blr.calculate_blr_scattering_transfer_function(
            inclination_face_on
        )
        scattering_tf_inclined = self.blr.calculate_blr_scattering_transfer_function(
            inclination_inclined
        )

        tau_ax_face_on = np.linspace(
            0, len(scattering_tf_face_on) - 1, len(scattering_tf_face_on)
        )
        tau_ax_inclined = np.linspace(
            0, len(scattering_tf_inclined) - 1, len(scattering_tf_inclined)
        )

        mean_face_on = np.sum(tau_ax_face_on * scattering_tf_face_on)
        mean_inclined = np.sum(tau_ax_inclined * scattering_tf_inclined)

        # when height is involved, inclined mean lags are longer than face on cases
        assert mean_inclined > mean_face_on
        # mean value should be greater than 100 by construction
        # (inner radius of blr is at 500 Rg and max height is 1000 Rg, so the absolute minimum possible
        # is (500**2 + 1000*2)**0.5 - 1000 ~ 118 Rg in the face on case
        assert mean_face_on > 100

        # use assert_almost_equal due to potential rounding
        npt.assert_almost_equal(np.sum(scattering_tf_face_on), 1)
        npt.assert_almost_equal(np.sum(scattering_tf_inclined), 1)

    def test_calculate_blr_emission_line_transfer_function(self):
        inclination = 70
        velocity_range = [-0.4, 0.1]

        weighting, blr_el_tf = self.blr.calculate_blr_emission_line_transfer_function(
            inclination,
            velocity_range=velocity_range,
            emission_efficiency_array=None,
        )

        tau_ax = np.linspace(0, len(blr_el_tf) - 1, len(blr_el_tf))
        mean_tau = np.sum(tau_ax * blr_el_tf)

        # the mean time lag is more complicated this time, but still should be larger than 100 Rg
        assert mean_tau > 100

        npt.assert_almost_equal(np.sum(blr_el_tf), 1)

        new_velocity_range = [-0.1, 0.4]
        new_weighting, new_blr_el_tf = (
            self.blr.calculate_blr_emission_line_transfer_function(
                inclination,
                velocity_range=new_velocity_range,
                emission_efficiency_array=None,
            )
        )

        new_tau_ax = np.linspace(0, len(new_blr_el_tf) - 1, len(new_blr_el_tf))
        new_mean_tau = np.sum(new_tau_ax * new_blr_el_tf)

        if np.sum(new_blr_el_tf) > 0:
            assert new_mean_tau > 100
            npt.assert_almost_equal(np.sum(new_blr_el_tf), 1)
        assert new_weighting > weighting


if __name__ == "__main__":
    pytest.main()
