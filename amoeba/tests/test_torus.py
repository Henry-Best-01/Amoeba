import pytest
import numpy as np
from amoeba.src.amoeba.Classes.blr_streamline import Streamline
from amoeba.src.amoeba.Classes.torus import Torus
from amoeba.src.amoeba.Classes.flux_projection import FluxProjection
import astropy.units as u
import numpy.testing as npt


class TestBlr:

    def setup(self):

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
