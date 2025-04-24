import numpy as np
import numpy.testing as npt
from speclite.filters import FilterResponse
from amoeba.Classes.accretion_disk import AccretionDisk
from amoeba.Classes.blr import BroadLineRegion
from amoeba.Classes.blr_streamline import Streamline
from amoeba.Classes.agn import Agn
from amoeba.Classes.diffuse_continuum import DiffuseContinuum
from amoeba.Classes.torus import Torus
from amoeba.Classes.flux_projection import FluxProjection
from amoeba.Util.util import (
    create_maps,
    convert_cartesian_to_polar,
)
from astropy import units as u
from astropy import constants as const


class TestAgn:

    def setup_method(self):

        self.init_smbh_mass_exp = 8.0
        self.init_redshift_source = 1.0
        self.init_inclination_angle = 0.0
        self.init_corona_height = 10
        self.init_number_grav_radii = 1000
        self.init_resolution = 1000
        self.init_spin = 0
        self.init_OmM = 0.3
        self.init_H0 = 70

        self.my_accretion_disk_kwargs = create_maps(
            self.init_smbh_mass_exp,
            self.init_redshift_source,
            self.init_number_grav_radii,
            self.init_inclination_angle,
            self.init_resolution,
            spin=self.init_spin,
            corona_height=self.init_corona_height,
        )

        self.accretion_disk = AccretionDisk(**self.my_accretion_disk_kwargs)

        x_vals = np.linspace(-2000, 2000, 100)
        X, Y = np.meshgrid(
            x_vals, x_vals / np.cos(self.init_inclination_angle * np.pi / 180)
        )
        R, Phi = convert_cartesian_to_polar(X, Y)

        self.radii_array = R
        self.phi_array = Phi
        cloud_density_radial_dependence = 0
        cloud_density_array = None
        r_in_in_gravitational_radii = 800
        r_out_in_gravitational_radii = 1000
        line_strength = 0.3
        frequencies = np.linspace(1 / 2000, 1 / 2, 100)
        power_law_density_dependence = -0.1
        name = "my diffuse continuum"

        self.my_dc_kwargs = {
            "smbh_mass_exp": self.init_smbh_mass_exp,
            "redshift_source": self.init_redshift_source,
            "inclination_angle": self.init_inclination_angle,
            "radii_array": self.radii_array,
            "phi_array": self.phi_array,
            "cloud_density_radial_dependence": cloud_density_radial_dependence,
            "cloud_density_array": cloud_density_array,
            "OmM": self.init_OmM,
            "H0": self.init_H0,
            "r_in_in_gravitational_radii": r_in_in_gravitational_radii,
            "r_out_in_gravitational_radii": r_out_in_gravitational_radii,
            "name": name,
            "line_strength": line_strength,
            "frequencies": frequencies,
            "power_law_density_dependence": power_law_density_dependence,
        }

        self.my_continuum = DiffuseContinuum(**self.my_dc_kwargs)

        self.init_launch_radius = 500  # Rg
        self.init_launch_theta = 0  # degrees
        self.init_max_height = 1000  # Rg
        self.init_height_step = 200
        self.init_line_strength = 0.3
        self.init_rest_frame_wavelength_in_nm = 600
        self.init_characteristic_distance = self.init_max_height // 5
        self.init_asymptotic_poloidal_velocity = 0.2
        self.init_poloidal_launch_velocity = 10**-5

        self.test_blr_streamline = Streamline(
            self.init_launch_radius,
            self.init_launch_theta,
            self.init_max_height,
            self.init_characteristic_distance,
            self.init_asymptotic_poloidal_velocity,
            poloidal_launch_velocity=self.init_poloidal_launch_velocity,
            height_step=self.init_height_step,
        )

        self.init_launch_theta_angled = 45
        self.test_blr_streamline_angled = Streamline(
            self.init_launch_radius,
            self.init_launch_theta_angled,
            self.init_max_height,
            self.init_characteristic_distance,
            self.init_asymptotic_poloidal_velocity,
            poloidal_launch_velocity=self.init_poloidal_launch_velocity,
            height_step=self.init_height_step,
        )

        self.test_torus_streamline_angled = Streamline(
            self.init_launch_radius * 10,
            self.init_launch_theta_angled,
            self.init_max_height,
            self.init_characteristic_distance,
            self.init_asymptotic_poloidal_velocity,
            poloidal_launch_velocity=self.init_poloidal_launch_velocity,
            height_step=self.init_height_step,
        )

        self.my_blr_kwargs = {
            "smbh_mass_exp": self.init_smbh_mass_exp,
            "max_height": self.init_max_height,
            "rest_frame_wavelength_in_nm": self.init_rest_frame_wavelength_in_nm,
            "redshift_source": self.init_redshift_source,
            "height_step": self.init_height_step,
            "line_strength": self.init_line_strength,
        }

        self.blr = BroadLineRegion(**self.my_blr_kwargs)

        self.blr.add_streamline_bounded_region(
            self.test_blr_streamline,
            self.test_blr_streamline_angled,
        )

        self.my_torus_kwargs = {
            "smbh_mass_exp": self.init_smbh_mass_exp,
            "max_height": self.init_max_height,
            "redshift_source": self.init_redshift_source,
            "height_step": self.init_height_step,
        }

        self.my_agn = Agn(
            agn_name="Wow, what an AGN.",
            **self.my_accretion_disk_kwargs,
        )

        self.my_blr_streamline_kwargs = {
            "InnerStreamline": self.test_blr_streamline,
            "OuterStreamline": self.test_blr_streamline_angled,
        }

        self.my_accretion_disk_kwargs["max_height"] = self.init_max_height
        self.my_accretion_disk_kwargs["power_spectrum"] = None

        self.my_populated_agn = Agn(
            agn_name="Amazing AGN",
            **self.my_accretion_disk_kwargs,
        )
        self.my_populated_agn.add_default_accretion_disk()
        self.my_populated_agn.add_diffuse_continuum(**self.my_dc_kwargs)
        self.my_populated_agn.add_blr(**self.my_blr_kwargs)
        self.my_populated_agn.add_streamline_bounded_region_to_blr(
            **self.my_blr_streamline_kwargs
        )

        self.my_populated_agn.add_torus(**self.my_torus_kwargs)

        self.extra_agn_kwargs = self.my_accretion_disk_kwargs.copy()
        del self.extra_agn_kwargs["H0"]
        del self.extra_agn_kwargs["OmM"]

        self.what_a_basic_agn = Agn(
            agn_name="Not so amazing AGN", **self.extra_agn_kwargs
        )

        min_kwargs = {
            "smbh_mass_exp": self.init_smbh_mass_exp,
            "redshift_source": self.init_redshift_source,
            "inclination_angle": self.init_inclination_angle,
        }
        self.no_kwarg_agn = Agn(agn_name="It's like a box", **min_kwargs)

    def test_initialization(self):

        assert self.my_agn.smbh_mass_exp == self.init_smbh_mass_exp
        assert self.my_agn.inclination_angle == self.init_inclination_angle
        assert self.my_agn.redshift_source == self.init_redshift_source
        assert self.my_agn.OmM == self.init_OmM
        assert self.my_agn.H0 == self.init_H0
        assert self.my_agn.smbh_mass == 10**self.init_smbh_mass_exp * const.M_sun.to(
            u.kg
        )
        assert self.my_agn.redshift_source == self.init_redshift_source
        assert self.my_agn.name == "Wow, what an AGN."
        assert self.my_agn.disk_is_updatable == True
        assert self.my_agn.blr_indicies == []
        assert self.my_agn.default_accretion_disk_kwargs is not None
        assert self.my_agn.generic_accretion_disk_kwargs is not None
        assert self.my_agn.blr_kwargs is not None
        assert self.my_agn.diffuse_continuum_kwargs is not None
        assert self.my_agn.intrinsic_signal_kwargs is not None
        assert self.my_agn.torus_kwargs is not None

    def test_add_default_accretion_disk(self):

        assert "accretion_disk" not in self.my_agn.components.keys()

        self.my_agn.add_default_accretion_disk()
        self.my_agn.add_default_accretion_disk(visc_temp_prof="NT")

        assert "accretion_disk" in self.my_agn.components.keys()
        assert self.my_agn.disk_is_updatable == True

        assert isinstance(self.my_agn.components["accretion_disk"], AccretionDisk)
        assert (
            self.my_agn.components["accretion_disk"].smbh_mass_exp
            == self.init_smbh_mass_exp
        )
        assert (
            self.my_agn.components["accretion_disk"].rg
            == self.my_agn.gravitational_radius
        )

    def test_add_generic_accretion_disk(self):

        agn_kwargs = self.my_accretion_disk_kwargs.copy()

        agn_kwargs["radii_array"] = self.radii_array
        agn_kwargs["phi_array"] = self.phi_array
        agn_kwargs["temp_array"] = 10 * self.radii_array
        agn_kwargs["g_array"] = np.ones(np.shape(self.radii_array))

        self.my_agn.add_generic_accretion_disk(**agn_kwargs)

        assert self.my_agn.disk_is_updatable == False
        assert "accretion_disk" in self.my_agn.components.keys()

        assert isinstance(self.my_agn.components["accretion_disk"], AccretionDisk)
        assert (
            self.my_agn.components["accretion_disk"].smbh_mass_exp
            == self.init_smbh_mass_exp
        )
        assert (
            self.my_agn.components["accretion_disk"].rg
            == self.my_agn.gravitational_radius
        )

    def test_add_blr(self):

        assert self.my_agn.add_blr(**self.my_blr_kwargs)

        assert "blr_0" in self.my_agn.components.keys()

        assert isinstance(self.my_agn.components["blr_0"], BroadLineRegion)

        self.my_blr_kwargs["line_strength"] = 2

        assert self.my_agn.add_blr(blr_index=1, **self.my_blr_kwargs)

        assert "blr_1" in self.my_agn.components.keys()
        assert 1 in self.my_agn.blr_indicies

        assert isinstance(self.my_agn.components["blr_1"], BroadLineRegion)

        assert (
            self.my_agn.components["blr_0"].line_strength
            < self.my_agn.components["blr_1"].line_strength
        )

    def test_add_streamline_bounded_region_to_blr(self):

        self.my_agn.add_blr(**self.my_blr_kwargs)
        self.my_agn.add_blr(blr_index=1, **self.my_blr_kwargs)

        self.one_blr_streamline = {"InnerStreamline": self.test_blr_streamline_angled}

        self.two_blr_streamlines = {
            "InnerStreamline": self.test_blr_streamline,
            "OuterStreamline": self.test_blr_streamline_angled,
        }

        # make sure that by adding only one streamline, we don't alter the blr
        assert False == self.my_agn.add_streamline_bounded_region_to_blr(
            0, **self.one_blr_streamline
        )

        assert np.size(self.my_agn.components["blr_" + str(0)].density_grid) == np.size(
            self.my_agn.components["blr_" + str(1)].density_grid
        )

        # show adding 2 streamlines physically changes the blr
        assert self.my_agn.add_streamline_bounded_region_to_blr(
            0, **self.two_blr_streamlines
        )

        assert np.size(self.my_agn.components["blr_" + str(0)].density_grid) > np.size(
            self.my_agn.components["blr_" + str(1)].density_grid
        )

        bad_blr_index = 33
        assert not self.my_agn.add_streamline_bounded_region_to_blr(
            blr_index=bad_blr_index, **self.two_blr_streamlines
        )

    def test_get_blr_density_axes(self):
        # we want to reliably get the axes for any type of input
        desired_index = 0
        list_of_axes = self.my_populated_agn.get_blr_density_axes(
            blr_index=desired_index
        )
        expected_shape = np.shape(
            self.my_populated_agn.components[
                "blr_" + str(desired_index)
            ].get_density_axis()
        )
        assert isinstance(list_of_axes, list)
        assert isinstance(list_of_axes[0], list)
        assert np.shape(list_of_axes[0]) == expected_shape
        all_axes = self.my_populated_agn.get_blr_density_axes()
        assert len(all_axes) == len(list_of_axes)
        assert isinstance(all_axes, list)
        assert isinstance(all_axes[0], list)
        assert np.shape(all_axes[0]) == expected_shape
        desired_index = "0"
        list_of_axes = self.my_populated_agn.get_blr_density_axes(
            blr_index=desired_index
        )
        assert isinstance(list_of_axes, list)
        assert isinstance(list_of_axes[0], list)
        assert np.shape(list_of_axes[0]) == expected_shape
        desired_index = [0]
        list_of_axes = self.my_populated_agn.get_blr_density_axes(
            blr_index=desired_index
        )
        assert isinstance(list_of_axes, list)
        assert isinstance(list_of_axes[0], list)
        assert np.shape(list_of_axes[0]) == expected_shape

        self.my_populated_agn.add_blr(blr_index=44.2, **self.my_blr_kwargs)

        all_axes = self.my_populated_agn.get_blr_density_axes()
        assert len(all_axes) > len(list_of_axes)

    def test_set_blr_efficiency_array(self):

        R = self.my_populated_agn.get_blr_density_axes()[0][0]
        test_efficiency = 1 / (1 + abs(R - 200))
        blr_index = 0
        assert np.shape(R) == np.shape(
            self.my_populated_agn.components["blr_0"].density_grid
        )
        assert not self.my_populated_agn.set_blr_efficiency_array()
        assert not self.my_populated_agn.set_blr_efficiency_array(
            efficiency_array=test_efficiency
        )
        assert not self.my_populated_agn.set_blr_efficiency_array(blr_index=blr_index)
        new_index = 22
        self.my_populated_agn.add_blr(blr_index=new_index)
        assert not self.my_populated_agn.set_blr_efficiency_array(blr_index=new_index)
        assert self.my_populated_agn.set_blr_efficiency_array(
            blr_index=blr_index,
            efficiency_array=test_efficiency,
        )
        assert not self.my_populated_agn.set_blr_efficiency_array(blr_index=blr_index)

    def test_add_torus(self):

        assert self.my_agn.add_torus(**self.my_torus_kwargs)

        assert isinstance(self.my_agn.components["torus"], Torus)

    def test_add_streamline_bounded_region_to_torus(self):

        self.my_agn.add_torus(**self.my_torus_kwargs)

        previous_density_grid_size = np.size(
            self.my_agn.components["torus"].density_grid
        )

        self.torus_streamline_kwargs = {"Streamline": self.test_torus_streamline_angled}

        assert self.my_agn.add_streamline_bounded_region_to_torus(
            **self.torus_streamline_kwargs
        )

        assert (
            np.size(self.my_agn.components["torus"].density_grid)
            > previous_density_grid_size
        )

        assert isinstance(self.my_agn.components["torus"], Torus)

        assert not self.my_agn.add_streamline_bounded_region_to_torus(
            **{"not_a_streamline": [1, 2, 3]}
        )

    def test_add_diffuse_continuum(self):

        additional_kwargs = {
            "cloud_density_radial_dependence": -0.5,
            "r_in_in_gravitational_radii": 400,
            "r_out_in_gravitational_radii": 5000,
        }

        self.my_agn.add_diffuse_continuum(**additional_kwargs)

        assert self.my_agn.components["diffuse_continuum"].responsivity_constant == 1
        assert (
            self.my_agn.components["diffuse_continuum"].rest_frame_wavelengths is None
        )
        assert self.my_agn.components["diffuse_continuum"].emissivity_etas is None

        additional_kwargs["responsivity_constant"] = 0.6

        assert self.my_agn.add_diffuse_continuum(**additional_kwargs)

        assert (
            self.my_agn.components["diffuse_continuum"].responsivity_constant
            == additional_kwargs["responsivity_constant"]
        )
        assert (
            self.my_agn.components["diffuse_continuum"].rest_frame_wavelengths is None
        )
        assert self.my_agn.components["diffuse_continuum"].emissivity_etas is None

        sample_wavelengths = [100, 500, 1000]
        sample_emissivities = [0.1, 0.4, 0.3]

        additional_kwargs["rest_frame_wavelengths"] = sample_wavelengths
        additional_kwargs["emissivity_etas"] = sample_emissivities

        assert self.my_agn.add_diffuse_continuum(**additional_kwargs)

        assert (
            self.my_agn.components["diffuse_continuum"].rest_frame_wavelengths
            is sample_wavelengths
        )

        assert (
            self.my_agn.components["diffuse_continuum"].emissivity_etas
            is sample_emissivities
        )

        # note the agn is prepared at redshift = 1
        test_emissivity_at_300_nm = self.my_agn.components[
            "diffuse_continuum"
        ].interpolate_spectrum_to_wavelength(300 * (1 + self.init_redshift_source))

        d_lam = (0.4 - 0.1) / (500 - 100)

        npt.assert_almost_equal(test_emissivity_at_300_nm, 0.1 + d_lam * 200)

        not_enough_kwargs = {"cloud_density_radial_dependence": -0.5}
        closer_kwargs = {
            "cloud_density_radial_dependence": -0.5,
            "radii_array": np.ones((100, 100)),
        }

        del self.my_agn.diffuse_continuum_kwargs["radii_array"]
        del self.my_agn.diffuse_continuum_kwargs["phi_array"]
        del self.my_agn.diffuse_continuum_kwargs["r_out_in_gravitational_radii"]

        with npt.assert_raises(AssertionError):
            self.my_agn.add_diffuse_continuum(**not_enough_kwargs)
        with npt.assert_raises(AssertionError):
            self.my_agn.add_diffuse_continuum(**closer_kwargs)

        extra_kwargs = self.my_dc_kwargs.copy()
        del extra_kwargs["radii_array"]
        assert self.my_agn.add_diffuse_continuum(**extra_kwargs)
        extra_kwargs = self.my_dc_kwargs.copy()
        del extra_kwargs["phi_array"]
        del self.my_agn.diffuse_continuum_kwargs["r_in_in_gravitational_radii"]
        assert "r_out_in_gravitational_radii" in extra_kwargs.keys()
        assert self.what_a_basic_agn.add_diffuse_continuum(**extra_kwargs)

        tiny_kwargs = {"r_out_in_gravitational_radii": 400}

        self.no_kwarg_agn.add_diffuse_continuum(**tiny_kwargs)

    def test_add_intrinsic_signal_parameters(self):

        length_of_light_curve = 1000
        cadence = 1

        # frequencies span up to nyquist frequency
        frequencies = np.linspace(
            1 / (2 * length_of_light_curve), 1 / (2 * cadence), length_of_light_curve
        )

        power_spectrum = (1 / frequencies) ** 2

        intrinsic_signal_kwargs = {
            "power_spectrum": power_spectrum,
            "frequencies": frequencies,
        }

        assert self.my_agn.add_intrinsic_signal_parameters(**intrinsic_signal_kwargs)

        assert self.my_agn.power_spectrum is power_spectrum
        assert self.my_agn.frequencies is frequencies
        assert self.my_agn.random_seed is None

        my_seed = 822

        intrinsic_signal_kwargs["random_seed"] = my_seed

        self.my_agn.add_intrinsic_signal_parameters(**intrinsic_signal_kwargs)

        assert self.my_agn.random_seed is my_seed

    def test_visualize_static_accretion_disk(self):

        wavelength_in_nm = 700

        # make sure it only works when you have an accretion disk
        with npt.assert_raises(KeyError):
            my_disk_flux_projection = self.my_agn.visualize_static_accretion_disk(
                wavelength_in_nm
            )

        self.my_agn.add_default_accretion_disk()

        my_disk_flux_projection = self.my_agn.visualize_static_accretion_disk(
            wavelength_in_nm
        )

        assert isinstance(my_disk_flux_projection, FluxProjection)
        assert my_disk_flux_projection.redshift_source == self.my_agn.redshift_source

        assert np.shape(my_disk_flux_projection.get_plotting_axes()) == np.shape(
            self.my_agn.components["accretion_disk"].get_plotting_axes()
        )

    def test_visualize_static_blr(self):

        with npt.assert_raises(KeyError):
            self.my_agn.visualize_static_blr()

        self.my_agn.add_blr(**self.my_blr_kwargs)
        self.blr.add_streamline_bounded_region(
            self.test_blr_streamline,
            self.test_blr_streamline_angled,
        )

        my_blr_projection = self.my_agn.visualize_static_blr()

        assert isinstance(my_blr_projection, FluxProjection)

        velocity_range = [0, 1]
        visualization_kwargs = {"velocity_range": velocity_range}

        my_approaching_blr_projection = self.my_agn.visualize_static_blr(
            **visualization_kwargs
        )

        assert isinstance(my_approaching_blr_projection, FluxProjection)
        assert np.sum(my_blr_projection.flux_array) != np.sum(
            my_approaching_blr_projection
        )

        # check superluminal velocities raise ValueError
        velocity_range = [1, 2]
        error_visualization_kwargs = {"velocity_range": velocity_range}

        with npt.assert_raises(ValueError):
            self.my_agn.visualize_static_blr(**error_visualization_kwargs)

    def test_visualize_torus_obscuration(self):

        with npt.assert_raises(KeyError):
            self.my_agn.visualize_torus_obscuration(50)

        self.my_agn.add_torus(**self.my_torus_kwargs)
        assert not self.my_agn.components[
            "torus"
        ].interpolate_to_extinction_at_wavelength(50)

        wavelengths = [100, 1000, 10000]
        absorption_strengths = [10**-10, 10**-10, 10**-5]

        self.my_torus_kwargs["rest_frame_wavelengths"] = wavelengths
        self.my_torus_kwargs["extinction_coefficients"] = absorption_strengths

        assert self.my_agn.add_torus(**self.my_torus_kwargs)

        self.torus_streamline_kwargs = {"Streamline": self.test_torus_streamline_angled}

        assert self.my_agn.add_streamline_bounded_region_to_torus(
            **self.torus_streamline_kwargs
        )

        wavelength = 1000

        my_torus_projection = self.my_agn.visualize_torus_obscuration(wavelength)

        assert isinstance(my_torus_projection, FluxProjection)

    def test_visualize_static_diffuse_continuum(self):

        sample_wavelengths = [100, 10000]
        sample_emissivities = [0.1, 1.0]

        additional_kwargs = {
            "cloud_density_radial_dependence": -0.5,
            "r_in_in_gravitational_radii": 400,
            "r_out_in_gravitational_radii": 5000,
        }

        additional_kwargs["rest_frame_wavelengths"] = sample_wavelengths
        additional_kwargs["emissivity_etas"] = sample_emissivities

        self.my_agn.add_diffuse_continuum(**additional_kwargs)

        observer_wavelength_in_nm_400 = 400
        observer_wavelength_in_nm_9001 = 9001

        projected_diffuse_continuum_400 = (
            self.my_agn.visualize_static_diffuse_continuum(
                observer_wavelength_in_nm_400
            )
        )

        projected_diffuse_continuum_9001 = (
            self.my_agn.visualize_static_diffuse_continuum(
                observer_wavelength_in_nm_9001
            )
        )

        assert isinstance(projected_diffuse_continuum_400, FluxProjection)

        assert np.shape(
            projected_diffuse_continuum_400.get_plotting_axes()
        ) == np.shape(projected_diffuse_continuum_9001.get_plotting_axes())

        assert (
            projected_diffuse_continuum_400.total_flux
            < projected_diffuse_continuum_9001.total_flux
        )

    def test_generate_intrinsic_signal(self):

        length_of_light_curve = 1000
        cadence = 1

        # frequencies span up to nyquist frequency
        frequencies = np.linspace(
            1 / (2 * length_of_light_curve), 1 / (2 * cadence), length_of_light_curve
        )

        power_spectrum = (1 / frequencies) ** 2

        intrinsic_signal_kwargs = {
            "power_spectrum": power_spectrum,
            "frequencies": frequencies,
        }

        assert self.my_agn.add_intrinsic_signal_parameters(**intrinsic_signal_kwargs)

        time_axis, my_signal = self.my_agn.generate_intrinsic_signal(
            length_of_light_curve
        )

        assert len(my_signal) == length_of_light_curve

        npt.assert_almost_equal(np.mean(my_signal), 0, 5)
        npt.assert_almost_equal(np.std(my_signal), 1, 5)

        time_axis, my_second_signal = self.my_agn.generate_intrinsic_signal(
            length_of_light_curve
        )

        assert np.sum((my_signal - my_second_signal) ** 2) != 0

        intrinsic_signal_kwargs["random_seed"] = 17
        self.my_agn.add_intrinsic_signal_parameters(**intrinsic_signal_kwargs)

        time_axis, my_seeded_signal = self.my_agn.generate_intrinsic_signal(
            length_of_light_curve
        )
        time_axis, my_second_seeded_signal = self.my_agn.generate_intrinsic_signal(
            length_of_light_curve, **intrinsic_signal_kwargs
        )

        assert np.sum((my_seeded_signal - my_second_seeded_signal) ** 2) == 0

        new_frequencies = np.linspace(
            1 / (length_of_light_curve), 1 / (cadence), 2 * length_of_light_curve
        )
        new_power_spectrum = (1 / new_frequencies) ** 2.5

        assert len(new_frequencies) != len(self.my_agn.frequencies)
        assert len(new_power_spectrum) != len(self.my_agn.power_spectrum)

        new_intrinsic_signal_kwargs = {
            "power_spectrum": new_power_spectrum,
            "frequencies": new_frequencies,
        }

        time_axis, my_third_signal = self.my_agn.generate_intrinsic_signal(
            2 * length_of_light_curve, **new_intrinsic_signal_kwargs
        )

        time_axis, my_fourth_signal = self.my_agn.generate_intrinsic_signal(
            2 * length_of_light_curve
        )

        assert np.sum((my_third_signal - my_fourth_signal) ** 2) == 0

        cadence = 0.2
        n_frequencies = int(length_of_light_curve / cadence)
        frequencies = np.linspace(
            1 / (2 * length_of_light_curve), 1 / (2 * cadence), n_frequencies
        )

        power_spectrum = (1 / frequencies) ** 2

        intrinsic_signal_kwargs = {
            "power_spectrum": power_spectrum,
            "frequencies": frequencies,
        }

        assert self.my_agn.add_intrinsic_signal_parameters(**intrinsic_signal_kwargs)

        time_axis, my_fifth_signal = self.my_agn.generate_intrinsic_signal(
            length_of_light_curve
        )

        assert len(my_fifth_signal) == length_of_light_curve / cadence

        # reset kwargs for testing purposes
        empty_dict_of_kwargs = {
            "power_spectrum": None,
            "frequencies": None,
            "random_seed": None,
        }

        self.my_agn.add_intrinsic_signal_parameters(**empty_dict_of_kwargs)

        assert not self.my_agn.generate_intrinsic_signal(length_of_light_curve)

    def test_update_smbh_mass_exponent(self):

        new_mass_exp = 3.3

        assert self.my_populated_agn.smbh_mass_exp != new_mass_exp

        for component in self.my_populated_agn.components.keys():
            assert (
                self.my_populated_agn.components[component].smbh_mass_exp
                != new_mass_exp
            )

        assert self.my_populated_agn.update_smbh_mass_exponent(new_mass_exp)

        assert self.my_populated_agn.smbh_mass_exp == new_mass_exp

        self.my_populated_agn.add_blr()

        for component in self.my_populated_agn.components.keys():
            assert (
                self.my_populated_agn.components[component].smbh_mass_exp
                == new_mass_exp
            )

        agn_kwargs = self.my_accretion_disk_kwargs.copy()

        agn_kwargs["radii_array"] = self.radii_array
        agn_kwargs["phi_array"] = self.phi_array
        agn_kwargs["temp_array"] = 10 * self.radii_array
        agn_kwargs["g_array"] = np.ones(np.shape(self.radii_array))

        self.my_agn.add_generic_accretion_disk(**agn_kwargs)

        assert self.my_agn.update_smbh_mass_exponent(new_mass_exp)

    def test_update_inclination(self):

        new_inclination = 42

        assert self.my_populated_agn.inclination_angle != new_inclination

        for component in self.my_populated_agn.components.keys():
            if (
                "inclination_angle"
                in vars(self.my_populated_agn.components[component]).keys()
            ):
                assert (
                    self.my_populated_agn.components[component].inclination_angle
                    != new_inclination
                )

        assert self.my_populated_agn.update_inclination(new_inclination)

        assert self.my_populated_agn.inclination_angle == new_inclination

        for component in self.my_populated_agn.components.keys():
            if (
                "inclination_angle"
                in vars(self.my_populated_agn.components[component]).keys()
            ):
                assert (
                    self.my_populated_agn.components[component].inclination_angle
                    == new_inclination
                )

        agn_kwargs = self.my_accretion_disk_kwargs.copy()

        agn_kwargs["radii_array"] = self.radii_array
        agn_kwargs["phi_array"] = self.phi_array
        agn_kwargs["temp_array"] = 10 * self.radii_array
        agn_kwargs["g_array"] = np.ones(np.shape(self.radii_array))

        self.my_agn.add_generic_accretion_disk(**agn_kwargs)

        assert not self.my_agn.update_inclination(new_inclination)

    def test_update_redshift(self):

        new_redshift = 200

        assert self.my_populated_agn.redshift_source != new_redshift

        for component in self.my_populated_agn.components.keys():
            assert self.my_populated_agn.redshift_source != new_redshift

        assert self.my_populated_agn.update_redshift(new_redshift)

        assert self.my_populated_agn.redshift_source == new_redshift

        for component in self.my_populated_agn.components.keys():
            assert self.my_populated_agn.redshift_source == new_redshift

    def test_update_h0(self):

        new_H0 = 1

        assert self.my_populated_agn.H0 != new_H0

        for component in self.my_populated_agn.components.keys():
            assert self.my_populated_agn.H0 != new_H0

        assert self.my_populated_agn.update_h0(new_H0)

        assert self.my_populated_agn.H0 == new_H0

        for component in self.my_populated_agn.components.keys():
            assert self.my_populated_agn.H0 == new_H0

    def test_update_omega_m(self):

        new_omega_m = 0.999

        assert self.my_populated_agn.OmM != new_omega_m

        for componet in self.my_populated_agn.components.keys():
            assert self.my_populated_agn.OmM != new_omega_m

        assert self.my_populated_agn.update_omega_m(new_omega_m)

        assert self.my_populated_agn.OmM == new_omega_m

        for componet in self.my_populated_agn.components.keys():
            assert self.my_populated_agn.OmM == new_omega_m

    def test_update_line_strength(self):

        new_line_strength = 40

        self.my_blr_kwargs["line_strength"] = 4
        self.my_populated_agn.add_blr(blr_index=88, **self.my_blr_kwargs)

        for index in self.my_populated_agn.blr_indicies:
            assert (
                self.my_populated_agn.components["blr_" + str(index)]
                != new_line_strength
            )

        for index in self.my_populated_agn.blr_indicies:
            assert self.my_populated_agn.update_line_strength(index, new_line_strength)

        for index in self.my_populated_agn.blr_indicies:
            assert (
                self.my_populated_agn.components["blr_" + str(index)].line_strength
                == new_line_strength
            )

    def test_calculate_accretion_disk_transfer_function(self):

        wavelength = 77

        assert not self.my_agn.calculate_accretion_disk_transfer_function(wavelength)

        my_disk_tf = self.my_populated_agn.calculate_accretion_disk_transfer_function(
            wavelength
        )

        assert isinstance(my_disk_tf, (list, np.ndarray))

        taus = np.linspace(0, len(my_disk_tf) - 1, len(my_disk_tf))
        mean = np.sum(my_disk_tf * taus) / np.sum(my_disk_tf)

        assert mean > 0

    def test_calculate_blr_transfer_function(self):

        wavelengths = [100, 2000]

        assert not self.my_agn.calculate_blr_transfer_function(wavelengths)

        my_blr_tfs = self.my_populated_agn.calculate_blr_transfer_function(wavelengths)

        weight, my_blr_tf = my_blr_tfs[0]

        assert isinstance(my_blr_tfs, (list, np.ndarray))
        assert isinstance(my_blr_tf, (list, np.ndarray))
        assert isinstance(weight, (float, int))

        taus = np.linspace(0, len(my_blr_tf) - 1, len(my_blr_tf))
        mean = np.sum(my_blr_tf * taus) / np.sum(my_blr_tf)
        assert mean > 0

    def test_calculate_diffuse_continuum_mean_time_lag(self):

        wavelength = 200 * np.pi

        assert not self.my_agn.calculate_diffuse_continuum_mean_time_lag(wavelength)

        lag_increase = self.my_populated_agn.calculate_diffuse_continuum_mean_time_lag(
            wavelength
        )

        assert lag_increase > 0

    def test_intrinsic_signal_propogation_pipeline(self):

        assert not self.my_populated_agn.intrinsic_signal_propagation_pipeline()
        assert not self.my_populated_agn.intrinsic_signal_propagation_pipeline(
            observer_frame_wavelengths_in_nm=500, speclite_filter="lsst2023-u"
        )

        assert not self.my_populated_agn.intrinsic_signal_propagation_pipeline(
            speclite_filter=["yellow"]
        )

        assert not self.my_populated_agn.intrinsic_signal_propagation_pipeline(
            observer_frame_wavelengths_in_nm=["cyan"]
        )

        assert not self.my_agn.intrinsic_signal_propagation_pipeline(
            speclite_filter="lsst2023-i"
        )

        assert not self.my_populated_agn.intrinsic_signal_propagation_pipeline(
            speclite_filter="lsst2023-*"
        )

        frequencies = np.linspace(1 / (2000), 1 / (2), 1000)

        power_spectrum = (1 / frequencies) ** 2

        intrinsic_signal_kwargs = {
            "power_spectrum": power_spectrum,
            "frequencies": frequencies,
        }

        assert self.my_populated_agn.add_intrinsic_signal_parameters(
            **intrinsic_signal_kwargs
        )

        all_my_signals = self.my_populated_agn.intrinsic_signal_propagation_pipeline(
            speclite_filter=["lsst2023-u", "lsst2023-z"], return_components=True
        )

        # I expect that this will return, for the u and z band, a list of the continuum, blr, and total signal
        # 3 components (continuum, blr, total)
        assert len(all_my_signals) == 3
        # each component has 2 bands (u, z)
        for jj in range(len(all_my_signals)):
            if isinstance(all_my_signals[jj], list):
                assert len(all_my_signals[jj]) == 2
            else:
                for key in all_my_signals[jj].keys():
                    # if we have a blr, we have instead 3 items in this list (time_axis, light_curve, weighting factor)
                    for band in range(len(all_my_signals[jj][key])):
                        assert len(all_my_signals[jj][key][band]) == 3
        # each component's band has 2 items (time_axis, light_curve), and each of these are equal in length
        # unless it's the blr, when there is a dictionary to allow for multiple blr components
        for jj in range(len(all_my_signals)):
            for kk in range(len(all_my_signals[jj])):
                if isinstance(all_my_signals[jj], list):
                    assert len(all_my_signals[jj][kk]) == 2
                    assert len(all_my_signals[jj][kk][0]) == len(
                        all_my_signals[jj][kk][1]
                    )
                else:
                    for key in all_my_signals[jj].keys():
                        assert len(all_my_signals[jj][key][0]) == len(
                            all_my_signals[jj][key][1]
                        )

        self.my_populated_agn.update_line_strength(blr_index=0, new_line_strength=0.3)

        # must also work for a set of wavelength ranges
        all_my_signals_2 = self.my_populated_agn.intrinsic_signal_propagation_pipeline(
            observer_frame_wavelengths_in_nm=[[100, 500], [500, 10000]],
            return_components=True,
        )

        assert len(all_my_signals_2) == 3
        # each component has 2 bands ([100-500], [500-10000])
        for jj in range(len(all_my_signals_2)):
            if isinstance(all_my_signals_2[jj], list):
                assert len(all_my_signals_2[jj]) == 2
            else:
                for key in all_my_signals_2[jj].keys():
                    for band in range(len(all_my_signals_2[jj][key])):
                        assert len(all_my_signals_2[jj][key][band]) == 3
        for jj in range(len(all_my_signals_2)):
            for kk in range(len(all_my_signals_2[jj])):
                if isinstance(all_my_signals_2[jj], list):
                    assert len(all_my_signals_2[jj][kk]) == 2
                    assert len(all_my_signals_2[jj][kk][0]) == len(
                        all_my_signals_2[jj][kk][1]
                    )
                else:
                    for key in all_my_signals_2[jj].keys():
                        assert len(all_my_signals_2[jj][key][0]) == len(
                            all_my_signals_2[jj][key][1]
                        )

        # must also work for a set of wavelengths
        all_my_signals_3 = self.my_populated_agn.intrinsic_signal_propagation_pipeline(
            observer_frame_wavelengths_in_nm=[100, 500, 10000, 750, 850],
            return_components=False,
        )

        assert len(all_my_signals_3) == 5
        # each component has 3 bands (100, 500, 10000, 750, 850)
        for jj in range(len(all_my_signals_3)):
            assert len(all_my_signals_3[jj]) == 2
        for jj in range(len(all_my_signals_3)):
            for kk in range(len(all_my_signals_3[jj])):
                assert len(all_my_signals_3[jj][0]) == len(all_my_signals_3[jj][1])

        # must also work for one wavelength range
        my_only_signal = self.my_populated_agn.intrinsic_signal_propagation_pipeline(
            observer_frame_wavelengths_in_nm=[[300, 700]]
        )

        assert len(my_only_signal) == 1
        assert len(my_only_signal[0]) == 2
        assert len(my_only_signal[0][0]) == len(my_only_signal[0][1])

        # and also one speclite filter
        my_other_only_signal = (
            self.my_populated_agn.intrinsic_signal_propagation_pipeline(
                speclite_filter="lsst2023-u"
            )
        )

        assert len(my_other_only_signal) == 1
        assert len(my_other_only_signal[0]) == 2
        assert len(my_other_only_signal[0][0]) == len(my_other_only_signal[0][1])

    def test_visualize_agn_pipeline(self):

        observer_frame_wavelengths_in_nm = [[1000, 1400]]
        # pick something very red which will have lower total flux
        no_blr_wavelengths = [[100000, 120000]]

        diffuse_continuum_wavelengths = [100, 10000]
        diffuse_continuum_spectrum = [1, 1]

        self.my_dc_kwargs["rest_frame_wavelengths"] = diffuse_continuum_wavelengths
        self.my_dc_kwargs["emissivity_etas"] = diffuse_continuum_spectrum

        self.my_populated_agn.add_diffuse_continuum(**self.my_dc_kwargs)

        components_with_blr = self.my_populated_agn.visualize_agn_pipeline(
            inclination_angle=None,
            observer_frame_wavelengths_in_nm=observer_frame_wavelengths_in_nm,
            return_components=True,
        )

        components_no_blr = self.my_populated_agn.visualize_agn_pipeline(
            inclination_angle=None,
            observer_frame_wavelengths_in_nm=no_blr_wavelengths,
            return_components=True,
        )

        projection_effect_with_blr = self.my_populated_agn.visualize_agn_pipeline(
            inclination_angle=None,
            observer_frame_wavelengths_in_nm=observer_frame_wavelengths_in_nm,
            return_components=False,
        )

        projection_effect_no_blr = self.my_populated_agn.visualize_agn_pipeline(
            inclination_angle=None,
            observer_frame_wavelengths_in_nm=no_blr_wavelengths,
            return_components=False,
        )

        assert isinstance(components_with_blr, list)
        assert isinstance(components_no_blr, list)
        assert isinstance(projection_effect_with_blr[0], FluxProjection)
        assert isinstance(projection_effect_no_blr[0], FluxProjection)
        assert isinstance(components_with_blr[0], list)
        assert isinstance(components_no_blr[0], list)
        assert isinstance(components_with_blr[0][0], FluxProjection)
        assert isinstance(components_no_blr[0][1], FluxProjection)

        assert len(components_with_blr[0]) == len(components_no_blr[0])
        assert np.shape(components_with_blr[0][-1].flux_array) > np.shape(
            components_no_blr[0][-1].flux_array
        )

        assert (
            projection_effect_with_blr[0].total_flux
            > projection_effect_no_blr[0].total_flux
        )

        # show that the pipeline works identically with and without returning components
        my_projection = self.my_populated_agn.visualize_agn_pipeline(
            inclination_angle=None,
            observer_frame_wavelengths_in_nm=observer_frame_wavelengths_in_nm,
            return_components=False,
        )
        assert my_projection[0].total_flux == projection_effect_with_blr[0].total_flux

        square_diff = np.sum(
            (projection_effect_with_blr[0].flux_array - my_projection[0].flux_array)
            ** 2
        )
        assert square_diff == 0

        # test that an agn without components will not return a projection
        my_null_projection = self.my_agn.visualize_agn_pipeline(
            observer_frame_wavelengths_in_nm=observer_frame_wavelengths_in_nm
        )
        assert my_null_projection is False

        # test that an agn with components but no wavelength will not return a projection
        my_null_projection = self.my_populated_agn.visualize_agn_pipeline()
        assert my_null_projection is False

        # test that we can project an agn in multiple bands with one call
        my_multitude_of_bands = ["lsst2023-u", "lsst2023-g", "lsst2023-z"]
        my_multi_components = self.my_populated_agn.visualize_agn_pipeline(
            speclite_filter=my_multitude_of_bands, return_components=True
        )
        my_multi_projection = self.my_populated_agn.visualize_agn_pipeline(
            speclite_filter=my_multitude_of_bands, return_components=False
        )
        assert len(my_multitude_of_bands) == len(my_multi_projection)
        # test we have the right number of components (no torus emission yet)
        assert len(self.my_populated_agn.components) == len(my_multi_components[0]) + 1
        # test each flux distribution is different
        assert my_multi_projection[0].total_flux != my_multi_projection[1].total_flux
        assert my_multi_projection[0].total_flux != my_multi_projection[2].total_flux

        # test that when inclination angle is updatable, you can update the agn object
        # directly with the projection pipeline
        new_inclination_angle = 40
        my_inclined_projection = self.my_populated_agn.visualize_agn_pipeline(
            inclination_angle=new_inclination_angle,
            speclite_filter=my_multitude_of_bands,
            return_components=False,
        )
        assert self.my_populated_agn.inclination_angle == new_inclination_angle

        for jj in range(len(my_multitude_of_bands)):
            assert (
                my_inclined_projection[jj].total_flux
                != my_multi_projection[jj].total_flux
            )
            assert (
                np.sum(
                    (
                        my_multi_projection[jj].flux_array
                        - my_inclined_projection[jj].flux_array
                    )
                    ** 2
                )
                != 0
            )
