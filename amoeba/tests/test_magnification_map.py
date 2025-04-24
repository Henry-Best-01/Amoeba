import numpy as np
import numpy.testing as npt
from amoeba.Classes.magnification_map import MagnificationMap, ConvolvedMap
from amoeba.Classes.flux_projection import FluxProjection
from amoeba.Classes.accretion_disk import AccretionDisk
from amoeba.Util.util import create_maps
import pytest


class TestMagnificationMap:
    def setup_method(self):
        magnification_array = [
            [1, 2, 1, 1, 1, 2, 3, 4, 3],
            [1, 3, 2, 1, 1, 2, 3, 4, 2],
            [1, 5, 8, 2, 1, 2, 3, 4, 1],
            [1, 6, 16, 8, 2, 2, 3, 4, 2],
            [1, 4, 8, 3, 1, 2, 3, 4, 3],
            [1, 3, 1, 1, 1, 2, 3, 5, 3],
            [4, 2, 6, 3, 3, 2, 5, 10, 5],
            [5, 2, 7, 20, 7, 2, 3, 5, 3],
            [4, 2, 12, 15, 12, 2, 3, 4, 3],
        ]
        flat_array = np.asarray(magnification_array.copy()).flatten()
        assert np.ndim(flat_array) == 1

        redshift_source = 2.0
        redshift_lens = 1.0
        convergence = 0.3
        shear = 0.1
        name = "silly test array"
        total_microlens_einstein_radii = 1

        self.magnification_map = MagnificationMap(
            redshift_source,
            redshift_lens,
            magnification_array,
            convergence,
            shear,
            total_microlens_einstein_radii=total_microlens_einstein_radii,
            name=name,
        )

        alt_magnification_map = MagnificationMap(
            redshift_source,
            redshift_lens,
            flat_array,
            convergence,
            shear,
            total_microlens_einstein_radii=total_microlens_einstein_radii,
            name=name,
        )
        # Note I am not including tests for the lines which open magnification
        # map files. To do so, I would have to include some test files or
        # design the test function to first write these files then delete them.

        MagnificationMap(
            redshift_source,
            redshift_lens,
            "definitely_not_a_file_path.invalid_type",
            convergence,
            shear,
            total_microlens_einstein_radii=total_microlens_einstein_radii,
            name=name,
        )

        assert self.magnification_map.name == name
        assert self.magnification_map.redshift_source == redshift_source
        assert self.magnification_map.redshift_lens == redshift_lens
        assert self.magnification_map.shear == shear
        assert self.magnification_map.convergence == convergence
        assert self.magnification_map.H0 == 70
        assert self.magnification_map.einstein_radius_in_meters > 0
        assert self.magnification_map.resolution == 9
        assert self.magnification_map.macro_magnification == (
            1 / ((1 - convergence) ** 2 - shear**2)
        )
        assert self.magnification_map.pixel_size > 0
        # demand that it was normalized
        assert np.sum(self.magnification_map.magnification_array) == np.size(
            self.magnification_map.magnification_array
        )
        assert np.argmax(self.magnification_map.ray_map) == np.argmax(
            self.magnification_map.magnification_array
        )

    def test_pull_value_from_grid(self):
        x_value = 3
        y_value = 5

        magnification_value = self.magnification_map.pull_value_from_grid(
            x_value, y_value
        )

        assert magnification_value > 0

        x_value = 1.1
        y_value = 6.8

        magnification_value_decimal = self.magnification_map.pull_value_from_grid(
            x_value, y_value
        )

        assert magnification_value_decimal > 0

        magnification_value_decimal_macromag = (
            self.magnification_map.pull_value_from_grid(
                x_value, y_value, weight_by_macromag=True
            )
        )

        assert magnification_value_decimal_macromag > 0

        assert magnification_value_decimal_macromag != magnification_value_decimal

    def test_pull_light_curve(self):

        stationary_transverse_velocity = 0.0
        effective_transverse_velocity = 4000
        light_curve_duration_in_years = 1

        x_start_position = 5
        y_start_position = 5

        random_seed = 4

        stationary_light_curve = self.magnification_map.pull_light_curve(
            stationary_transverse_velocity,
            light_curve_duration_in_years,
            x_start_position=x_start_position,
            y_start_position=y_start_position,
            random_seed=random_seed,
        )

        light_curve = self.magnification_map.pull_light_curve(
            effective_transverse_velocity,
            light_curve_duration_in_years,
            x_start_position=x_start_position,
            y_start_position=y_start_position,
            random_seed=random_seed,
        )

        # a light curve must be at least 2 points
        assert len(light_curve) >= 2
        # light curve should evolve, while stationary will not
        assert stationary_light_curve[0] == stationary_light_curve[-1]
        assert stationary_light_curve[-1] != light_curve[-1]
        # they have the same start point
        assert light_curve[0] == stationary_light_curve[0]

        # define a path which goes over a high magnification point
        y_start_position = 4
        x_start_position = 1
        phi_travel_direction = 90

        light_curve_rise_fall = self.magnification_map.pull_light_curve(
            effective_transverse_velocity,
            light_curve_duration_in_years,
            x_start_position=x_start_position,
            y_start_position=y_start_position,
            phi_travel_direction=phi_travel_direction,
        )

        assert len(light_curve_rise_fall) >= 4
        assert light_curve_rise_fall[2] > light_curve_rise_fall[0]

        light_curve_macromag = self.magnification_map.pull_light_curve(
            effective_transverse_velocity,
            light_curve_duration_in_years,
            x_start_position=x_start_position,
            y_start_position=y_start_position,
            random_seed=random_seed,
            weight_by_macromag=True,
        )
        light_curve_macromag_components = self.magnification_map.pull_light_curve(
            effective_transverse_velocity,
            light_curve_duration_in_years,
            x_start_position=x_start_position,
            y_start_position=y_start_position,
            random_seed=random_seed,
            return_track_coords=True,
            weight_by_macromag=True,
        )

        assert (
            np.sum(abs(light_curve_macromag - light_curve_macromag_components[0])) == 0
        )

    def test_convolve_with_flux_projection(self):

        # must generate a simple flux projection object to test this method
        flux_array = [[1 / 4, 1 / 2, 1 / 4], [1 / 2, 1, 1 / 2], [1 / 4, 1 / 2, 1 / 4]]
        observer_wavelength = 100
        smbh_mass_exp = 6.9
        redshift_source = self.magnification_map.redshift_source
        r_out_in_rg = 700
        inclination = 0

        test_flux_projection = FluxProjection(
            flux_array,
            observer_wavelength,
            smbh_mass_exp,
            redshift_source,
            r_out_in_rg,
            inclination,
        )

        self.test_convolution_object = (
            self.magnification_map.convolve_with_flux_projection(test_flux_projection)
        )

        # check inheritance
        assert isinstance(self.test_convolution_object, MagnificationMap)
        assert isinstance(self.test_convolution_object, ConvolvedMap)

        # check everything passed through properly
        assert (
            self.magnification_map.redshift_source
            == self.test_convolution_object.redshift_source
        )
        assert (
            self.magnification_map.pixel_size == self.test_convolution_object.pixel_size
        )
        assert (
            self.magnification_map.total_microlens_einstein_radii
            == self.test_convolution_object.total_microlens_einstein_radii
        )
        assert (
            self.magnification_map.mean_microlens_mass_in_kg
            == self.test_convolution_object.mean_microlens_mass_in_kg
        )
        assert (
            self.magnification_map.resolution == self.test_convolution_object.resolution
        )
        assert (
            test_flux_projection.smbh_mass_exp
            == self.test_convolution_object.smbh_mass_exp
        )
        assert (
            test_flux_projection.inclination_angle
            == self.test_convolution_object.inclination_angle
        )
        assert (
            test_flux_projection.observer_frame_wavelength_in_nm
            == self.test_convolution_object.observer_frame_wavelength_in_nm
        )
        assert (
            self.magnification_map.redshift_lens
            == self.test_convolution_object.redshift_lens
        )
        assert (
            self.magnification_map.macro_magnification
            == self.test_convolution_object.macro_magnification
        )

        # check the convolution actually changed the values
        x_value = 3
        y_value = np.pi

        assert self.test_convolution_object.pull_value_from_grid(
            x_value,
            y_value,
        ) != self.magnification_map.pull_value_from_grid(
            x_value,
            y_value,
        )

    def test_calculate_microlensed_transfer_function(self):

        # need to set up an accretion disk object
        smbh_mass_exp = 8.0
        redshift_source = 2.0
        inclination_angle = 0.0
        corona_height = 10
        number_grav_radii = 500
        resolution = 500
        spin = 0

        accretion_disk_data_1 = create_maps(
            smbh_mass_exp,
            redshift_source,
            number_grav_radii,
            inclination_angle,
            resolution,
            spin=spin,
            corona_height=corona_height,
        )

        self.AccretionDisk = AccretionDisk(**accretion_disk_data_1)

        # as a sanity check, prepare an identity array for microlensing

        big_magnification_ones = np.ones((resolution, resolution))

        self.identity_magnification_array = MagnificationMap(
            redshift_source,
            self.magnification_map.redshift_lens,
            big_magnification_ones,
            self.magnification_map.convergence,
            self.magnification_map.shear,
            mean_microlens_mass_in_kg=self.magnification_map.mean_microlens_mass_in_kg,
            total_microlens_einstein_radii=1,
            name="identity",
        )

        wavelength_1 = 100
        wavelength_2 = 300

        micro_tf_1_id = (
            self.identity_magnification_array.calculate_microlensed_transfer_function(
                self.AccretionDisk,
                wavelength_1,
            )
        )

        micro_tf_2_id = (
            self.identity_magnification_array.calculate_microlensed_transfer_function(
                self.AccretionDisk,
                wavelength_2,
            )
        )

        tf_1 = self.AccretionDisk.construct_accretion_disk_transfer_function(
            wavelength_1
        )

        tf_2 = self.AccretionDisk.construct_accretion_disk_transfer_function(
            wavelength_2
        )

        assert len(micro_tf_1_id) == len(micro_tf_2_id)
        assert len(tf_1) == len(tf_2)

        tau_ax = np.linspace(0, len(tf_1) - 1, len(tf_1))
        tau_ax_ml = np.linspace(0, len(micro_tf_1_id) - 1, len(micro_tf_1_id))

        mean_micro_tf_1_id = np.sum(tau_ax_ml * micro_tf_1_id)
        mean_micro_tf_2_id = np.sum(tau_ax_ml * micro_tf_2_id)
        mean_tf_1 = np.sum(tau_ax * tf_1)
        mean_tf_2 = np.sum(tau_ax * tf_2)

        npt.assert_almost_equal(np.sum(micro_tf_1_id), 1)
        npt.assert_almost_equal(np.sum(micro_tf_2_id), 1)

        npt.assert_almost_equal(round(mean_micro_tf_1_id, 0), round(mean_tf_1, 0), 1)
        npt.assert_almost_equal(round(mean_micro_tf_2_id, 0), round(mean_tf_2, 0), 1)
        assert mean_micro_tf_2_id > mean_micro_tf_1_id
