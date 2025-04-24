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
        line_strength=1,
        **kwargs
    ):
        """Generate a broad line region object. This object contains the R-Z
        distributions of clouds that lead to emission. Note that once initialized, you
        must add information to it using the Streamline objects.

        :param smbh_mass_exp: solution to log_10(m_smbh / m_sun)
        :param max_height: max Z coordinate of the BLR in units of R_g
        :param rest_frame_wavelength_in_nm: the natural wavelength of emission in nm.
            This will get Doppler shifted and redshifted to the observer's reference
            frame.
        :param redshift_source: redshift of the BLR object.
        :param radial_step: resolution of the BLR in the R dimension, in R_g
        :param height_step: resolution of the BLR in the Z direction, in R_g
        :param max_radius: maximum radius of the BLR in R_g
        :param OmM: mass component of the energy budget of the universe
        :param H0: Hubble constant in units of km/s/Mpc
        :param line_strength: float/int representing how strong the total emission line
            is w.r.t. the continuum emission.
        """

        self.smbh_mass_exp = smbh_mass_exp
        self.rg = calculate_gravitational_radius(10**self.smbh_mass_exp)
        self.rest_frame_wavelength_in_nm = rest_frame_wavelength_in_nm
        self.redshift_source = redshift_source
        self.OmM = OmM
        self.H0 = H0
        self.line_strength = line_strength
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
        """Add a region to the BLR which represents the line emitting region.

        :param InnerStreamline: Streamline object representative of the inner boundary
            conditions.
        :param OuterStreamline: Streamline object representative of the outer boundary
            conditions.
        :param density_initial_weighting: weighting factor for the density grid. Useful
            when defining multiple regions in the same BLR object. All weighting is
            relative.
        :return: True if successful
        """

        assert InnerStreamline.height_step == OuterStreamline.height_step
        assert InnerStreamline.max_height == OuterStreamline.max_height
        assert InnerStreamline.height_step == self.height_step
        assert InnerStreamline.max_height == self.max_height
        assert density_initial_weighting > 0

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
            self.density_grid[np.argmax(mask) : np.argmax(mask) + sum(mask), hh] *= 0
            self.r_velocity_grid[np.argmax(mask) : np.argmax(mask) + sum(mask), hh] *= 0
            self.z_velocity_grid[np.argmax(mask) : np.argmax(mask) + sum(mask), hh] *= 0

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

        if self.emission_efficiency_array is not None:
            old_efficiency_array = self.emission_efficiency_array
            new_efficiency_array = np.zeros(self.blr_array_shape)
            new_efficiency_array[
                : np.size(old_efficiency_array, 0), : np.size(old_efficiency_array, 1)
            ] = old_efficiency_array

            self.set_emission_efficiency_array(new_efficiency_array)

        return True

    def project_blr_density_to_source_plane(self, inclination_angle):
        """Project the total density of the BLR into the source plane. Creates a
        FluxProjection object where the flux_array is representative of the density.

        :param inclination_angle: inclination of the source in degrees
        :return: FluxProjection containing metadata from the BLR and the emission array
        """

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
        """Project the BLR into the source plane. Similar to
        project_blr_density_to_source_plane, but can be weighted by the emission
        efficiency which is an array of equal size to the BLR's density array.

        :param inclination_angle: inclination which we view the source at, in degrees
        :param emission_efficiency_array: a 2d grid of weighting factors in the R-Z
            plane.
        :return: FluxProjection containing metadata from the BLR and the emission array
        """

        self.set_emission_efficiency_array(emission_efficiency_array)

        flux_map, _ = project_blr_to_source_plane(
            self.density_grid,
            self.z_velocity_grid,
            self.r_velocity_grid,
            inclination_angle,
            self.smbh_mass_exp,
            weighting_grid=self.emission_efficiency_array,
            radial_resolution=self.radial_step,
            vertical_resolution=self.height_step,
        )

        flux_map *= self.line_strength

        max_radius = (
            self.max_height * np.tan(inclination_angle * np.pi / 180) + self.max_radius
        )

        flux_projection = FluxProjection(
            flux_map,
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
        """Project a portion of the BLR to the source plane by determining the Doppler
        shifting at each coordinate in R-Z-phi space.

        :param inclination_angle: orientation of the source with respect to the observer
            in degrees
        :param velocity_range: a list representing the min and max velocities to choose.
            Cannot be used with observed_wavelength_range_in_nm or speclite_filter. Note
            that positive velocity is toward the observer.
        :param observed_wavelength_range_in_nm: a list representing the min and max
            wavelengths in nanometers which are observed. Cannot be used with
            velocity:range or speclite_filter.
        :param speclite_filter: Speclite filter object or string representing a loadable
            Speclite filter. Cannot be used with velocity_range or
            observed_wavelength_range_in_nm.
        :param emission_efficiency_array: a 2d grid of weighting factors in the R-Z
            plane
        :return: FluxProjection containing metadata from the BLR and the velocity
            selected emission array
        """

        if velocity_range is not None and observed_wavelength_range_in_nm is not None:
            print("Please only provide the velocities or wavelengths. Not both!")
            return False
        if (
            velocity_range is None
            and observed_wavelength_range_in_nm is None
            and speclite_filter is None
        ):
            print("Please provide the velocities or wavelengths.")
            velocity_range = [-1, 1]
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

        self.set_emission_efficiency_array(emission_efficiency_array)

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

        flux_map *= self.line_strength

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
        """Calculate the transfer function of the BLR under the assumption of basic
        scattering by the particles. Essentially a density-weighted scattering with no
        wavelength dependence.

        :param inclination_angle: inclination of the source with respect to the observer
            in degrees
        :return: list representing the BLR's transfer function under the assumption that
            all particles can scatter photons of any wavelength with time units R_g / c
        """

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
        """Calculate the transfer function of the BLR under the assumption that the
        emission is related to the density * emission_efficiency_array. Unlike
        calculate_blr_scattering_transfer_function, this has wavelength dependence.
        Assumes the major time delay is related to the light travel time and the
        relaxation time of an excited particle is negligable.

        :param inclination_angle: angle of orientation of the source with respect to the
            observer in degrees.
        :param velocity_range: a list representing the min and max velocities to choose.
            Cannot be used with observed_wavelength_range_in_nm or speclite_filter. Note
            that positive velocity is toward the observer.
        :param observed_wavelength_range_in_nm: a list representing the min and max
            wavelengths in nanometers which are observed. Cannot be used with
            velocity:range or speclite_filter.
        :param speclite_filter: Speclite filter object or string representing a loadable
            Speclite filter. Cannot be used with velocity_range or
            observed_wavelength_range_in_nm.
        :param emission_efficiency_array: a 2d grid of weighting factors in the R-Z
            plane. Primarily used to selectively weight regions in the local optimally
            emitting cloud model.
        :return: list representing the BLR's transfer function under only a subset of
            the BLR particles can scatter photons in time units of R_g / c
        """

        if velocity_range is not None and observed_wavelength_range_in_nm is not None:
            print("Please only provide the velocities or wavelengths. Not both!")
            return False
        if (
            velocity_range is None
            and observed_wavelength_range_in_nm is None
            and speclite_filter is None
        ):
            print("Please provide the velocities or wavelengths.")
            return False
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

        self.set_emission_efficiency_array(emission_efficiency_array)

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
        tf_weighting_factor *= self.line_strength

        return tf_weighting_factor, emission_line_tf

    def estimate_doppler_broadening(self, inclination_angle):
        """Estimate the maximum and minimum velocities of the BLR using the maximum
        values stored in velocity arrays. This is a helper method which determines when
        it's worth integrating through the BLR and when the integration will return 0.

        :param inclination_angle: inclination of the source w.r.t the observer in
            degrees.
        :return: array containing the estimated maximum receeding and approaching
            velocities in units v / c
        """

        assert inclination_angle < 90
        assert inclination_angle >= 0

        inclination_angle *= np.pi / 180
        max_range_v_r = np.max(
            [abs(np.max(self.r_velocity_grid)), abs(np.min(self.r_velocity_grid))]
        )
        min_range_v_z = np.min(self.z_velocity_grid)
        max_range_v_z = np.max(self.z_velocity_grid)
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

    def update_line_strength(self, line_strength):
        """Update the emission line strength stored in the BLR

        :param line_strength: updated strength of the emisison line w.r.t. the
            continuum emission
        :return: True if successful
        """
        prev_line_strength = self.line_strength
        assert isinstance(line_strength, (int, float))
        self.line_strength = line_strength
        if self.current_total_emission is not None and prev_line_strength != 0:
            self.current_total_emission *= self.line_strength / prev_line_strength
        return True

    def get_density_axis(self):
        """Get a meshgrid representation of the R-Z coordinates to get an
        array to apply local optimally emitting cloud region weighting. The spherical
        distance may be computed as:

        r_spherical = np.sqrt(R**2 * Z**2)

        :return: R and Z coordinates in a numpy meshgrid
        """

        R, Z = np.meshgrid(self.radii_values, self.height_values, indexing="ij")
        return R, Z

    def set_emission_efficiency_array(self, emission_efficiency_array=None):
        """Set the weighting of each (R, Z) coordinate. This is used to compute the
        emission and response of the BLR for spatially distinct regions.

        :param emission_efficiency_array: Array of int/float values representing
            how efficient the BLR responds at coordinate (R, Z)
        :return: True if successful
        """
        R, Z = self.get_density_axis()
        if emission_efficiency_array is None:
            self.emission_efficiency_array = np.ones(np.shape(R))
        else:
            assert np.shape(emission_efficiency_array) == np.shape(R)
            self.emission_efficiency_array = emission_efficiency_array
        return True
