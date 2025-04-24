import numpy as np
import numpy.testing as npt
from amoeba.Classes.accretion_disk import AccretionDisk
from amoeba.Util.util import create_maps, convert_spin_to_isco_radius
from astropy import units as u


class TestAccretionDisk:

    def setup_method(self):

        # basic, face on disk
        smbh_mass_exp = 8.0
        redshift_source = 1.0
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

        self.FaceOnDisk1 = AccretionDisk(**accretion_disk_data_1)

        # inclined disk with less mass
        smbh_mass_exp = 7.0
        inclination_angle = 45.0

        accretion_disk_data_2 = create_maps(
            smbh_mass_exp,
            redshift_source,
            number_grav_radii,
            inclination_angle,
            resolution,
            spin=spin,
        )

        self.InclinedDisk = AccretionDisk(**accretion_disk_data_2)

        # high spin disk zoomed in
        inclination_angle = 0.0
        spin = 1.0
        number_grav_radii = 50
        resolution = 50

        accretion_disk_data_zoom_spin = create_maps(
            smbh_mass_exp,
            redshift_source,
            number_grav_radii,
            inclination_angle,
            resolution,
            spin=spin,
        )

        self.FaceOnDiskZoomSpin = AccretionDisk(**accretion_disk_data_zoom_spin)

        # retrograde spin
        spin = -1.0

        accretion_disk_data_zoom_spin_retro = create_maps(
            smbh_mass_exp,
            redshift_source,
            number_grav_radii,
            inclination_angle,
            resolution,
            spin=spin,
        )
        accretion_disk_data_zoom_spin_retro["name"] = "retrograde spin"

        self.FaceOnDiskZoomSpinRetro = AccretionDisk(
            **accretion_disk_data_zoom_spin_retro
        )

        accretion_disk_data_albedo_no_r_out = accretion_disk_data_1.copy()
        accretion_disk_data_albedo_no_r_out["albedo_array"] = 0.7
        del accretion_disk_data_albedo_no_r_out["r_out_in_gravitational_radii"]

        self.NewDisk = AccretionDisk(**accretion_disk_data_albedo_no_r_out)

    def test_initializtion(self):
        assert self.FaceOnDisk1.smbh_mass_exp == 8.0
        assert self.FaceOnDisk1.redshift_source == 1.0
        assert self.FaceOnDisk1.spin == 0.0
        assert np.shape(self.FaceOnDisk1.radii_array) == (500, 500)
        npt.assert_almost_equal(np.max(self.FaceOnDisk1.radii_array), 500 * 2**0.5)
        npt.assert_almost_equal(self.FaceOnDisk1.r_out_in_gravitational_radii, 500, 3)
        assert np.shape(self.FaceOnDisk1.radii_array) == np.shape(
            self.FaceOnDisk1.phi_array
        )
        assert np.shape(self.FaceOnDisk1.radii_array) == np.shape(
            self.FaceOnDisk1.g_array
        )
        assert np.shape(self.FaceOnDisk1.radii_array) == np.shape(
            self.FaceOnDisk1.temp_array
        )
        assert self.FaceOnDisk1.rg > 10**5
        assert self.FaceOnDisk1.corona_height == 10
        assert self.FaceOnDiskZoomSpinRetro.name == "retrograde spin"
        assert np.max(self.FaceOnDiskZoomSpinRetro.temp_array) < np.max(
            self.FaceOnDiskZoomSpin.temp_array
        )
        assert self.InclinedDisk.inclination_angle == 45.0

    def test_calculate_surface_intensity_map(self):

        wavelength_1 = 700  # nm

        intensity_map_1 = self.FaceOnDisk1.calculate_surface_intensity_map(wavelength_1)

        intensity_map_2 = self.InclinedDisk.calculate_surface_intensity_map(
            wavelength_1
        )

        intensity_map_zoom_spin = (
            self.FaceOnDiskZoomSpin.calculate_surface_intensity_map(wavelength_1)
        )

        intensity_map_zoom_retro = (
            self.FaceOnDiskZoomSpinRetro.calculate_surface_intensity_map(wavelength_1)
        )

        assert intensity_map_1.total_flux > intensity_map_2.total_flux
        assert intensity_map_zoom_spin.total_flux > intensity_map_zoom_retro.total_flux

        wavelength_2 = 500 * u.nm  # nm

        intensity_map_blue = self.FaceOnDisk1.calculate_surface_intensity_map(
            wavelength_2
        )

        assert intensity_map_blue.total_flux > intensity_map_1.total_flux

        intensities, wavelength_array = (
            self.FaceOnDisk1.calculate_surface_intensity_map(
                wavelength_1, return_wavelengths=True
            )
        )

        assert np.shape(intensities) == np.shape(wavelength_array)
        assert np.all(wavelength_array == wavelength_1 / 2)

    def test_calculate_db_dt_array(self):

        wavelength_1 = 700
        db_dt_array_1 = self.FaceOnDisk1.calculate_db_dt_array(wavelength_1)

        # the planck function is always increasing at constant wavelength
        assert np.all(db_dt_array_1 >= 0)

        wavelength_2 = 500
        db_dt_array_1_blue = self.FaceOnDisk1.calculate_db_dt_array(wavelength_2)

        deviation = db_dt_array_1 - db_dt_array_1_blue

        assert abs(np.sum(deviation)) > 0

        db_dt_array_zoom_spin = self.FaceOnDiskZoomSpin.calculate_db_dt_array(
            wavelength_1
        )
        db_dt_array_zoom_retro = self.FaceOnDiskZoomSpinRetro.calculate_db_dt_array(
            wavelength_1
        )

        assert np.sum(db_dt_array_zoom_spin) > np.sum(db_dt_array_zoom_retro)

        wavelength_astropy = 700 * u.nm

        db_dt_array_zoom_retro_astro = (
            self.FaceOnDiskZoomSpinRetro.calculate_db_dt_array(wavelength_astropy)
        )

        assert np.all(db_dt_array_zoom_retro_astro == db_dt_array_zoom_retro)

    def test_calculate_time_lag_array(self):

        time_lags_zoom_spin = self.FaceOnDiskZoomSpin.calculate_time_lag_array()

        assert np.all(time_lags_zoom_spin >= 0)

        # the center time lag will be smaller than the edges

        assert (
            time_lags_zoom_spin[
                int(np.size(self.FaceOnDiskZoomSpin.radii_array) ** 0.5) // 2,
                int(np.size(self.FaceOnDiskZoomSpin.radii_array) ** 0.5) // 2,
            ]
            < time_lags_zoom_spin[
                int(np.size(self.FaceOnDiskZoomSpin.radii_array) ** 0.5) // 2, 2
            ]
        )

        assert (
            time_lags_zoom_spin[
                int(np.size(self.FaceOnDiskZoomSpin.radii_array) ** 0.5) // 2,
                int(np.size(self.FaceOnDiskZoomSpin.radii_array) ** 0.5) // 2,
            ]
            < time_lags_zoom_spin[
                int(np.size(self.FaceOnDiskZoomSpin.radii_array) ** 0.5) // 2, -2
            ]
        )

        assert (
            time_lags_zoom_spin[
                int(np.size(self.FaceOnDiskZoomSpin.radii_array) ** 0.5) // 2,
                int(np.size(self.FaceOnDiskZoomSpin.radii_array) ** 0.5) // 2,
            ]
            < time_lags_zoom_spin[
                2, int(np.size(self.FaceOnDiskZoomSpin.radii_array) ** 0.5) // 2
            ]
        )

        assert (
            time_lags_zoom_spin[
                int(np.size(self.FaceOnDiskZoomSpin.radii_array) ** 0.5) // 2,
                int(np.size(self.FaceOnDiskZoomSpin.radii_array) ** 0.5) // 2,
            ]
            < time_lags_zoom_spin[
                -2, int(np.size(self.FaceOnDiskZoomSpin.radii_array) ** 0.5) // 2
            ]
        )

    def test_calculate_dt_dlx_array(self):

        dt_dlx_array = self.InclinedDisk.calculate_dt_dlx_array()

        # adding energy will always increase temperature
        # zero is included to include shadow of the black hole
        assert np.all(dt_dlx_array >= 0)

        center_pixel = np.size(self.InclinedDisk.radii_array, 0) // 2 + 1

        data_points_y = dt_dlx_array[center_pixel, :] > 0
        data_points_x = dt_dlx_array[:, center_pixel] > 0

        assert dt_dlx_array[center_pixel, center_pixel] == 0
        assert dt_dlx_array[center_pixel, center_pixel // 2] > 0

    def test_construct_accretion_disk_transfer_function(self):

        wavelength_1 = 700
        wavelength_2 = 1000

        transfer_function_face_on_1 = (
            self.FaceOnDisk1.construct_accretion_disk_transfer_function(
                wavelength_1,
            )
        )

        transfer_function_inclined_1 = (
            self.InclinedDisk.construct_accretion_disk_transfer_function(
                wavelength_1,
            )
        )

        tau_ax_face_on_1 = np.linspace(
            0, len(transfer_function_face_on_1) - 1, len(transfer_function_face_on_1)
        )
        mean_face_on_1 = np.sum(tau_ax_face_on_1 * transfer_function_face_on_1)

        tau_ax_inclined_1 = np.linspace(
            0, len(transfer_function_inclined_1) - 1, len(transfer_function_inclined_1)
        )
        mean_inclined_1 = np.sum(tau_ax_inclined_1 * transfer_function_inclined_1)

        assert mean_face_on_1 < mean_inclined_1

        transfer_function_face_on_2 = (
            self.FaceOnDisk1.construct_accretion_disk_transfer_function(
                wavelength_2,
            )
        )

        mean_face_on_2 = np.sum(tau_ax_face_on_1 * transfer_function_face_on_2)

        assert mean_face_on_1 < mean_face_on_2

    def test_generate_snapshots(self):

        driving_signal = np.sin(np.linspace(0, 100, 100) / np.pi) + 5
        time_stamps = [20, 25, 44.3, 80]
        observer_frame_wavelength = 250
        driving_signal_fractional_strength = 0.3

        snapshots = self.FaceOnDisk1.generate_snapshots(
            observer_frame_wavelength,
            time_stamps,
            driving_signal,
            driving_signal_fractional_strength,
        )

        assert len(snapshots) == len(time_stamps)

        assert np.sum(snapshots[0].flux_array ** 2 - snapshots[2].flux_array ** 2) > 0
