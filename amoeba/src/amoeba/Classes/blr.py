import numpy as np
from astropy import units as u
from astropy import constants as const
from amoeba.Util.util import (
    project_blr_to_source_plane,
    calculate_blr_transfer_function,
    determine_emission_line_velocities,
    calculate_keplerian_velocity,
    calculate_gravitational_radius,
    convert_speclite_filter_to_wavelength_range,
)
from amoeba.Classes.flux_projection import FluxProjection


class BroadLineRegion:

    def __init__(
        self,
        smbh_mass_exp,
        max_height,
        rest_frame_wavelength_in_nm,
        redshift_source,
        radial_step=10,
        height_step=10,
        max_radius=0,
        OmM=0.3,
        H0=70,
    ):
        # res holds the number of R_g each pixel is

        self.smbh_mass_exp = smbh_mass_exp
        self.rg = calculate_gravitational_radius(10**self.smbh_mass_exp)
        self.rest_frame_wavelength_in_nm = rest_frame_wavelength_in_nm
        self.redshift_source = redshift_source
        self.OmM = OmM
        self.H0 = H0
        self.max_height = max_height
        self.radial_step = radial_step
        self.height_step = height_step
        self.max_radius = max_radius
        self.density_grid = np.zeros(
            (max_radius // radial_step + 1, max_height // height_step + 1)
        )
        self.emission_efficiency_array = None
        self.current_calculated_inclination = None
        self.current_total_emission = None
        self.z_velocity_grid = np.zeros(np.shape(self.density_grid))
        self.r_velocity_grid = np.zeros(np.shape(self.density_grid))
        self.radii_values = np.linspace(0, max_radius, max_radius // radial_step + 1)
        self.height_values = np.linspace(0, max_height, max_height // height_step + 1)

        self.blr_array_shape = np.shape(self.density_grid)

        self.mass = 10 ** (self.smbh_mass_exp) * const.M_sun.to(u.kg)

    def add_streamline_bounded_region(
        self, InnerStreamline, OuterStreamline, density_initial_weighting=1
    ):

        # assure vertical coordinates are equal, otherwise interpolation is not well defined
        assert InnerStreamline.height_step == OuterStreamline.height_step
        assert InnerStreamline.max_height == OuterStreamline.max_height
        # assure vertical coordinates are equal to any previously added blr components
        assert InnerStreamline.height_step == self.height_step
        assert InnerStreamline.max_height == self.max_height
        assert density_initial_weighting > 0

        # Allow adaptive max radius
        if (
            np.max(
                [
                    np.max(InnerStreamline.radii_values),
                    np.max(OuterStreamline.radii_values),
                ]
            )
            > self.max_radius
        ):
            previous_maximum = self.max_radius // self.radial_step + 1
            self.max_radius = int(
                np.max(
                    [
                        np.max(InnerStreamline.radii_values),
                        np.max(OuterStreamline.radii_values),
                    ]
                )
            )
            dummygrid = np.zeros(
                (
                    self.max_radius // self.radial_step + 1,
                    self.max_height // self.height_step + 1,
                )
            )
            dummygrid[:previous_maximum, :] = self.density_grid
            self.density_grid = dummygrid
            dummygrid = np.zeros(np.shape(dummygrid))
            dummygrid[:previous_maximum, :] = self.z_velocity_grid
            self.z_velocity_grid = dummygrid
            dummygrid = np.zeros(np.shape(dummygrid))
            dummygrid[:previous_maximum, :] = self.r_velocity_grid
            self.r_velocity_grid = dummygrid

            self.radii_values = np.linspace(
                0, self.max_radius, self.max_radius // self.radial_step + 1
            )

        # loop over slabs
        for hh in range(np.size(self.height_values)):
            low_mask = self.radii_values >= min(
                InnerStreamline.radii_values[hh], OuterStreamline.radii_values[hh]
            )
            high_mask = self.radii_values <= max(
                InnerStreamline.radii_values[hh], OuterStreamline.radii_values[hh]
            )
            mask = np.logical_and(low_mask, high_mask) + 0
            if hh == 0:
                norm = sum(mask)
            # np.argmax(mask) returns the inner radius according to numpy documentation
            self.density_grid[
                np.argmax(mask) : np.argmax(mask) + sum(mask), hh
            ] *= 0  # overwrites these cells
            self.r_velocity_grid[np.argmax(mask) : np.argmax(mask) + sum(mask), hh] *= 0
            self.z_velocity_grid[np.argmax(mask) : np.argmax(mask) + sum(mask), hh] *= 0

            # kkspace is the radial space of transformed coordinates between the streamlines at any
            # constant height slab (e.g. looping index kk, cast to linear space)
            kkspace = np.linspace(0, sum(mask), sum(mask))
            self.z_velocity_grid[
                np.argmax(mask) : np.argmax(mask) + sum(mask), hh
            ] = InnerStreamline.poloidal_velocity[hh] * np.cos(
                InnerStreamline.launch_theta
            ) + (
                kkspace / sum(mask)
            ) * (
                OuterStreamline.poloidal_velocity[hh]
                * np.cos(OuterStreamline.launch_theta)
                - InnerStreamline.poloidal_velocity[hh]
                * np.cos(InnerStreamline.launch_theta)
            )
            self.r_velocity_grid[
                np.argmax(mask) : np.argmax(mask) + sum(mask), hh
            ] = InnerStreamline.poloidal_velocity[hh] * np.sin(
                InnerStreamline.launch_theta
            ) + (
                kkspace / sum(mask)
            ) * (
                OuterStreamline.poloidal_velocity[hh]
                * np.sin(OuterStreamline.launch_theta)
                - InnerStreamline.poloidal_velocity[hh]
                * np.sin(InnerStreamline.launch_theta)
            )

            del_pol_vels_on_vels = InnerStreamline.dpol_vel_dz_on_vel[hh] + (
                kkspace / sum(mask)
            ) * (
                OuterStreamline.dpol_vel_dz_on_vel[hh]
                - InnerStreamline.dpol_vel_dz_on_vel[hh]
            )
            self.density_grid[np.argmax(mask) : np.argmax(mask) + sum(mask), hh] += (
                density_initial_weighting
                * del_pol_vels_on_vels
                / self.radii_values[np.argmax(mask) : np.argmax(mask) + sum(mask)]
            )
        self.blr_array_shape = np.shape(self.density_grid)

    def project_blr_density_to_source_plane(self, inclination_angle):
        projection, projected_number_gravitational_radii = project_blr_to_source_plane(
            self.density_grid,
            self.z_velocity_grid,
            self.r_velocity_grid,
            inclination_angle,
            self.smbh_mass_exp,
            weighting_grid=self.density_grid,
            radial_resolution=self.radial_step,
            vertical_resolution=self.height_step,
        )

        # This needs to be fixed so I can return a proper fluxprojection object
        projection_object = FluxProjection(
            projection,
            self.rest_frame_wavelength_in_nm,
            self.smbh_mass_exp,
            self.redshift_source,
            projected_number_gravitational_radii,
            inclination_angle,
            OmM=self.OmM,
            H0=self.H0,
        )

        return projection

    def project_blr_total_intensity(
        self,
        inclination_angle,
        emission_efficiency_array=None,
    ):
        # essentially a weighted projection of the density
        # should be a FluxProjection object!

        if emission_efficiency_array is not None:
            self.emission_efficiency_array = emission_efficiency_array

        flux_map = project_blr_to_source_plane(
            self.density_grid,
            self.z_velocity_grid,
            self.r_velocity_grid,
            inclination_angle,
            self.smbh_mass_exp,
            weighting_grid=self.emission_efficiency_array,
            radial_resolution=self.radial_step,
            vertical_resolution=self.height_step,
        )

        max_radius = (
            self.max_height * np.tan(inclination_angle * np.pi / 180) + self.max_radius
        )

        # note that projecting the total intensity is equivalent to the integrated
        # flux over all wavelengths
        flux_projection = FluxProjection(
            flux_map[0],
            np.array([0, np.inf]),
            self.smbh_mass_exp,
            self.redshift_source,
            max_radius,
            inclination_angle,
            OmM=self.OmM,
            H0=self.H0,
        )

        self.current_calculated_inclination = inclination_angle
        self.current_total_emission = flux_projection.total_flux

        return flux_projection

    def project_blr_intensity_over_velocity_range(
        self,
        inclination_angle,
        velocity_range=None,
        observed_wavelength_range_in_nm=None,
        speclite_filter=None,
        emission_efficiency_array=None,
    ):

        if velocity_range is not None and observed_wavelength_range_in_nm is not None:
            print("Please only provide the velocities or wavelengths. Not both!")
        if (
            velocity_range is None
            and observed_wavelength_range_in_nm is None
            and speclite_filter is None
        ):
            print("Please provide the velocities or wavelengths.")
        if speclite_filter is not None:
            observed_wavelength_range_in_nm = (
                convert_speclite_filter_to_wavelength_range(
                    speclite_filter,
                )
            )
        if observed_wavelength_range_in_nm is not None:
            velocity_range = determine_emission_line_velocities(
                self.rest_frame_wavelength_in_nm,
                np.min(observed_wavelength_range_in_nm),
                np.max(observed_wavelength_range_in_nm),
                self.redshift_source,
            )

        if emission_efficiency_array is not None:
            self.emission_efficiency_array = emission_efficiency_array

        obs_plane_wavelength_in_nm = self.rest_frame_wavelength_in_nm * (
            1 + self.redshift_source
        )
        min_obs_plane_wavelength_in_nm = (
            obs_plane_wavelength_in_nm
            * ((1 - np.max(velocity_range)) / (1 + np.max(velocity_range))) ** 0.5
        )
        max_obs_plane_wavelength_in_nm = (
            obs_plane_wavelength_in_nm
            * ((1 - np.min(velocity_range)) / (1 + np.min(velocity_range))) ** 0.5
        )

        expected_broadening = self.estimate_doppler_broadening(inclination_angle)

        min_expected_wavelength_in_nm = (
            obs_plane_wavelength_in_nm
            * ((1 - np.max(expected_broadening)) / (1 + np.max(expected_broadening)))
            ** 0.5
        )
        max_expected_wavelength_in_nm = (
            obs_plane_wavelength_in_nm
            * ((1 - np.min(expected_broadening)) / (1 + np.min(expected_broadening)))
            ** 0.5
        )

        if min_expected_wavelength_in_nm > max_obs_plane_wavelength_in_nm:
            return FluxProjection(
                np.zeros((100, 100)),
                [min_obs_plane_wavelength_in_nm, max_obs_plane_wavelength_in_nm],
                self.smbh_mass_exp,
                self.redshift_source,
                100,
                inclination_angle,
                OmM=self.OmM,
                H0=self.H0,
            )
        if max_expected_wavelength_in_nm < min_obs_plane_wavelength_in_nm:
            return FluxProjection(
                np.zeros((100, 100)),
                [min_obs_plane_wavelength_in_nm, max_obs_plane_wavelength_in_nm],
                self.smbh_mass_exp,
                self.redshift_source,
                100,
                inclination_angle,
                OmM=self.OmM,
                H0=self.H0,
            )

        # Similar to above's density calculation, but this time only includes voxels within a velocity range
        flux_map, _ = project_blr_to_source_plane(
            self.density_grid,
            self.z_velocity_grid,
            self.r_velocity_grid,
            inclination_angle,
            self.smbh_mass_exp,
            velocity_range=velocity_range,
            weighting_grid=self.emission_efficiency_array,
            radial_resolution=self.radial_step,
            vertical_resolution=self.height_step,
        )

        max_radius = (
            self.max_height * np.tan(inclination_angle * np.pi / 180) + self.max_radius
        )

        flux_projection = FluxProjection(
            flux_map,
            [min_obs_plane_wavelength_in_nm, max_obs_plane_wavelength_in_nm],
            self.smbh_mass_exp,
            self.redshift_source,
            max_radius,
            inclination_angle,
            OmM=self.OmM,
            H0=self.H0,
        )

        return flux_projection

    def calculate_blr_scattering_transfer_function(self, inclination_angle):

        return calculate_blr_transfer_function(
            self.density_grid,
            self.z_velocity_grid,
            self.r_velocity_grid,
            inclination_angle,
            self.smbh_mass_exp,
            velocity_range=[-1, 1],
            weighting_grid=self.density_grid,
            radial_resolution=self.radial_step,
            vertical_resolution=self.height_step,
        )

    def calculate_blr_emission_line_transfer_function(
        self,
        inclination_angle,
        velocity_range=None,
        observed_wavelength_range_in_nm=None,
        speclite_filter=None,
        emission_efficiency_array=None,
    ):

        if velocity_range is not None and observed_wavelength_range_in_nm is not None:
            print("Please only provide the velocities or wavelengths. Not both!")
        if (
            velocity_range is None
            and observed_wavelength_range_in_nm is None
            and speclite_filter is None
        ):
            print("Please provide the velocities or wavelengths.")
        if speclite_filter is not None:
            observed_wavelength_range_in_nm = (
                convert_speclite_filter_to_wavelength_range(
                    speclite_filter,
                )
            )
        if observed_wavelength_range_in_nm is not None:
            velocity_range = determine_emission_line_velocities(
                self.rest_frame_wavelength_in_nm,
                np.min(observed_wavelength_range_in_nm),
                np.max(observed_wavelength_range_in_nm),
                self.redshift_source,
            )

        expected_broadening = self.estimate_doppler_broadening(inclination_angle)
        obs_plane_wavelength_in_nm = self.rest_frame_wavelength_in_nm * (
            1 + self.redshift_source
        )
        min_obs_plane_wavelength_in_nm = (
            obs_plane_wavelength_in_nm
            * ((1 - np.max(velocity_range)) / (1 + np.max(velocity_range))) ** 0.5
        )
        max_obs_plane_wavelength_in_nm = (
            obs_plane_wavelength_in_nm
            * ((1 - np.min(velocity_range)) / (1 + np.min(velocity_range))) ** 0.5
        )

        min_expected_wavelength_in_nm = (
            obs_plane_wavelength_in_nm
            * ((1 - np.max(expected_broadening)) / (1 + np.max(expected_broadening)))
            ** 0.5
        )
        max_expected_wavelength_in_nm = (
            obs_plane_wavelength_in_nm
            * ((1 - np.min(expected_broadening)) / (1 + np.min(expected_broadening)))
            ** 0.5
        )

        if min_expected_wavelength_in_nm > max_obs_plane_wavelength_in_nm:
            return 0, np.zeros(3)
        if max_expected_wavelength_in_nm < min_obs_plane_wavelength_in_nm:
            return 0, np.zeros(3)

        if emission_efficiency_array is not None:
            self.emission_efficiency_array = emission_efficiency_array

        emission_line_tf = calculate_blr_transfer_function(
            self.density_grid,
            self.z_velocity_grid,
            self.r_velocity_grid,
            inclination_angle,
            self.smbh_mass_exp,
            velocity_range=velocity_range,
            weighting_grid=self.emission_efficiency_array,
            radial_resolution=self.radial_step,
            vertical_resolution=self.height_step,
        )

        if self.current_calculated_inclination is not inclination_angle:
            self.current_calculated_inclination = inclination_angle
            self.emission_efficiency_array = emission_efficiency_array
            total_emission = self.project_blr_total_intensity(
                self.current_calculated_inclination,
                emission_efficiency_array=self.emission_efficiency_array,
            )
            self.current_total_emission = total_emission.total_flux

        current_emission = self.project_blr_intensity_over_velocity_range(
            inclination_angle,
            velocity_range=velocity_range,
            emission_efficiency_array=self.emission_efficiency_array,
        )

        tf_weighting_factor = current_emission.total_flux / self.current_total_emission

        # similar to previous function, but selects a region based on line-of-sight velocity range
        return tf_weighting_factor, emission_line_tf

    def estimate_doppler_broadening(self, inclination_angle):

        assert inclination_angle < 90
        assert inclination_angle >= 0

        inclination_angle *= np.pi / 180
        # radial velocities are symmetrically projected
        max_range_v_r = np.max(
            [abs(np.max(self.r_velocity_grid)), abs(np.min(self.r_velocity_grid))]
        )
        # z velocities are not symmetric
        min_range_v_z = np.min(self.z_velocity_grid)
        max_range_v_z = np.max(self.z_velocity_grid)
        # phi velocities are symmetric, but max depends on min radius
        densities_as_function_of_radius = np.sum(self.density_grid, axis=1)
        min_radius_argument = np.argmax(
            np.isfinite(1 / densities_as_function_of_radius)
        )
        min_radius_in_rg = self.radii_values[min_radius_argument]

        max_phi_velocity = calculate_keplerian_velocity(
            min_radius_in_rg * self.rg, 10**self.smbh_mass_exp
        )

        max_receeding_velocity = np.cos(inclination_angle) * min_range_v_z - np.sin(
            inclination_angle
        ) * np.sqrt(max_range_v_r**2 + max_phi_velocity**2)
        max_approaching_velocity = np.cos(inclination_angle) * max_range_v_z + np.sin(
            inclination_angle
        ) * np.sqrt(max_range_v_r**2 + max_phi_velocity**2)

        return np.asarray([max_receeding_velocity, max_approaching_velocity])
