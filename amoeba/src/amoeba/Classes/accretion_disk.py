from astropy import units as u
from astropy import constants as const
import numpy as np
from amoeba.Classes.flux_projection import FluxProjection
from amoeba.Util.util import (
    calculate_luminosity_distance,
    calculate_gravitational_radius,
    planck_law,
    planck_law_derivative,
    calculate_time_lag_array,
    calculate_dt_dlx,
    construct_accretion_disk_transfer_function,
    generate_snapshots_of_radiation_pattern,
)


class AccretionDisk:

    def __init__(
        self,
        smbh_mass_exp=None,
        redshift_source=None,
        inclination_angle=None,
        corona_height=None,
        temp_array=None,
        phi_array=None,
        g_array=None,
        radii_array=None,
        height_array=None,
        albedo_array=None,
        spin=0,
        OmM=0.3,
        H0=70,
        r_out_in_gravitational_radii=None,
        name="",
        **kwargs
    ):
        """Object representing an accretion disk which is assumed to be optically thick.

        :param smbh_mass_exp: mass exponent of the sumpermassive black hole at the
            center of the disk expressed as log_10(M / M_sun). Typical ranges are 6-11
            for AGN.
        :param redshift_source: positive float representing the redshift of the AGN.
        :param inclination_angle: inclination of the accretion disk with respect to the
            observer, in degrees
        :param corona_height: height of the corona in the lamppost model in units of R_g
            = GM/c^2
        :param temp_array: a 2d representation of the effective temperature distribution
            of the accretion disk
        :param phi_array: a 2d representation of the azimuth angles on the accretion
            disk
        :param g_array: a 2d representation of the g-factors (relativistic correction
            factors) on the accretion disk. Requires general relativistic ray tracing.
        :param radii_array: a 2d representation of the radii on the accretion disk, in
            R_g = GM/c^2
        :param height_array: a 2d representation of the height of the accretion disk in
            units of R_g = GM/c^2
        :param spin: float representing the dimensionless spin component of the SMBH
            (often denoted a_{*}) which may range from -1 to 1. Negative values
            represent accretion disks anti-aligned with the spin of the black hole.
        :param OmM: Cosmological parameter representing the mass fraction of the
            universe
        :param H0: Hubble constant in units of km/s/Mpc
        :param r_out_in_gravitational_radii: maximum radius of the accretion disk, in
            R_g = GM/c^2
        :param name: Name space
        """

        self.name = name
        self.smbh_mass_exp = smbh_mass_exp
        self.mass = 10**smbh_mass_exp * const.M_sun.to(u.kg)
        self.redshift_source = redshift_source
        self.inclination_angle = inclination_angle
        self.spin = spin
        self.radii_array = radii_array
        self.height_array = height_array
        if albedo_array is None:
            albedo_array = np.zeros(np.shape(radii_array))
        elif isinstance(albedo_array, (int, float)):
            albedo_array = np.ones(np.shape(radii_array)) * albedo_array
        self.albedo_array = albedo_array
        if r_out_in_gravitational_radii is None:
            r_out_in_gravitational_radii = radii_array[0, np.size(radii_array, 0) // 2]
        self.r_out_in_gravitational_radii = r_out_in_gravitational_radii
        self.radial_mask = radii_array <= r_out_in_gravitational_radii
        self.temp_array = temp_array * self.radial_mask
        self.phi_array = phi_array * self.radial_mask
        self.g_array = g_array
        self.OmM = OmM
        self.H0 = H0
        self.lum_dist = calculate_luminosity_distance(
            self.redshift_source, OmM=self.OmM, H0=self.H0
        )
        self.rg = calculate_gravitational_radius(10**self.smbh_mass_exp)
        self.pixel_size = (
            self.rg
            * self.r_out_in_gravitational_radii
            * 2
            / np.size(self.temp_array, 0)
        )
        self.corona_height = corona_height

    def calculate_surface_intensity_map(
        self, observer_frame_wavelength_in_nm, return_wavelengths=False
    ):
        """Method to calculate the surface flux distribution at some wavelength.

        :param observer_frame_wavelength_in_nm: Wavelength in nanometers used to
            determine black body flux
        :param return_wavelengths: Bool used to return the map of wavelengths used at
            each pixel
        :return: A 2d array representing the surface flux density at desired wavelength.
            If returnwavelength is True, returns a tuple of 2d arrays.
        """

        if isinstance(observer_frame_wavelength_in_nm, u.Quantity):
            dummy = observer_frame_wavelength_in_nm.to(u.nm)
            observer_frame_wavelength_in_nm = dummy.value

        redshift_factor = 1 / (1 + self.redshift_source)
        total_redshift_factor = redshift_factor * self.g_array
        rest_frame_wavelength = total_redshift_factor * observer_frame_wavelength_in_nm

        output = (
            np.nan_to_num(
                planck_law(self.temp_array, rest_frame_wavelength)
                * pow(self.g_array, 4.0)
            )
            * self.pixel_size**2
        )
        if return_wavelengths == True:
            return output, rest_frame_wavelength
        flux_projection = FluxProjection(
            output,
            observer_frame_wavelength_in_nm,
            self.smbh_mass_exp,
            self.redshift_source,
            self.r_out_in_gravitational_radii,
            self.inclination_angle,
            OmM=self.OmM,
            H0=self.H0,
        )
        return flux_projection

    def calculate_db_dt_array(self, observer_frame_wavelength_in_nm):
        """Calculate the rate of change of surface flux density with respect to a
        fluctuation in temperature.

        :param observer_frame_wavelength_in_nm: Wavelength in nanometers in the observer
            frame
        :return: 2d array representing the partial derivative of the Planck function
            with respect to a small change in temperature at each pixel
        """

        if isinstance(observer_frame_wavelength_in_nm, u.Quantity):
            dummy = observer_frame_wavelength_in_nm.to(u.nm)
            observer_frame_wavelength_in_nm = dummy.value

        redshift_factor = 1 / (1 + self.redshift_source)
        total_redshift_factor = redshift_factor * self.g_array
        rest_frame_wavelength = total_redshift_factor * observer_frame_wavelength_in_nm

        output = (
            planck_law_derivative(self.temp_array, rest_frame_wavelength)
            * pow(self.g_array, 4.0)
            * self.pixel_size**2
        )
        return np.nan_to_num(output)

    def calculate_time_lag_array(
        self,
        corona_height=None,
        axis_offset_in_gravitational_radii=0,
        angle_offset_in_degrees=0,
    ):
        """Calculate the time delay between a point source and the accretion disk.

        :param corona_height: None or int / float. If None, use the corona height in the
            accretion disk. Otherwise, value represents height of the flare in units R_g
            = GM/c^2
        :param axis_offset_in_gravitational_radii: Radial offset of the flare with
            repsect to the axis of symmetry in units R_g = GM/c^2
        :param angle_offset_in_degrees: Angular rotation of the offset flare in degrees.
            Zero degrees represents a flare on the nearer side of the accretion disk,
            while 180 degrees represents the far side.
        :return: A 2d array of time delays representing the extra path length from the
            corona to the midplane of the accretion disk in units R_g / c.
        """

        if corona_height is None:
            corona_height = self.corona_height

        time_lag_array = calculate_time_lag_array(
            self.radii_array,
            self.phi_array,
            self.inclination_angle,
            corona_height=corona_height,
            axis_offset_in_gravitational_radii=axis_offset_in_gravitational_radii,
            angle_offset_in_degrees=angle_offset_in_degrees,
            height_array=self.height_array,
        )

        time_lag_array *= self.radial_mask

        return time_lag_array

    def calculate_dt_dlx_array(
        self,
        corona_height=None,
        axis_offset_in_gravitational_radii=0,
        angle_offset_in_degrees=0,
    ):
        """Calculate the change in temperature with respect to a change in the X-ray
        luminosity.

        :param corona_height: None or int / float. If None, the initialized corona
            height will be used. Otherwise, represents the height of the flare in units
            R_g = GM/c^2.
        :param axis_offset_in_gravitational_radii: Axis offset of the flaring event in
            units R_g.
        :param angle_offset_in_gravitational_radii: Degree rotation around the axis of
            symmetry of the flaring event. Zero degrees represents the flare nearer to
            the observer for inclined disks, while 180 degrees represnts the far side of
            the accretion disk.
        :return: a 2d array of values representing the partial derivative of accretion
            disk's temperature with respect to irradiating luminosity.
        """

        if corona_height is None:
            corona_height = self.corona_height

        dt_dlx_array = calculate_dt_dlx(
            self.temp_array,
            self.radii_array,
            self.phi_array,
            self.smbh_mass_exp,
            corona_height,
            axis_offset_in_gravitational_radii=axis_offset_in_gravitational_radii,
            angle_offset_in_degrees=angle_offset_in_degrees,
            height_array=self.height_array,
            albedo_array=self.albedo_array,
        )

        return dt_dlx_array

    def construct_accretion_disk_transfer_function(
        self,
        observer_frame_wavelength_in_nm,
        corona_height=None,
        axis_offset_in_gravitational_radii=0,
        angle_offset_in_degrees=0,
        return_response_array_and_lags=False,
    ):
        """Calculate the transfer function of the accretion disk within the lamppost
        model.

        :param observer_frame_wavelength_in_nm: Wavelength in nm with respect to the
            observer to calculate the transfer function at
        :param corona_height: None or int / float. If None, the initialized corona
            height will be used. Otherwise, represents the height of the flare in units
            R_g = GM/c^2
        :param axis_offset_in_gravitational_radii: Axis offset of the flaring event in
            units R_g = GM/c^2
        :param angle_offset_in_gravitational_radii: Degree rotation around the axis of
            symmetry of the flaring event. Zero degrees represents the flare nearer to
            the observer for inclined disks, while 180 degrees represnts the far side of
            the accretion disk
        :param return_response_array_and_lags: Boolean flag to return the accretion disk
            response and time delay arrays instead of the computed transfer function
        :return: 1d array representing the transfer function of the accretion disk with
            respect to some change in flux at a particular wavelength
        """
        if corona_height is None:
            corona_height = self.corona_height

        rest_frame_wavelength_in_nm = observer_frame_wavelength_in_nm / (
            1 + self.redshift_source
        )

        rest_frame_transfer_function = construct_accretion_disk_transfer_function(
            rest_frame_wavelength_in_nm,
            self.temp_array,
            self.radii_array,
            self.phi_array,
            self.g_array,
            self.inclination_angle,
            self.smbh_mass_exp,
            corona_height,
            axis_offset_in_gravitational_radii=axis_offset_in_gravitational_radii,
            angle_offset_in_degrees=angle_offset_in_degrees,
            height_array=self.height_array,
            albedo_array=self.albedo_array,
            return_response_array_and_lags=return_response_array_and_lags,
        )

        return rest_frame_transfer_function

    def generate_snapshots(
        self,
        observer_frame_wavelength_in_nm,
        time_stamps,
        driving_signal,
        driving_signal_fractional_strength,
        corona_height=None,
        axis_offset_in_gravitational_radii=0,
        angle_offset_in_degrees=0,
    ):
        """Method to generate snapshots of the accretion disk's surface brightness under
        the assumption that a driving signal is actively being reprocessed.

        :param observer_frame_wavelength_in_nm: Wavelength in nm in the observer's frame
            which we are observing the source at
        :param time_stamps: 1d array or list of times to pull the snapshots at
        :param driving_signal: 1d array or list representing the driving light curve
        :param driving_signal_fractional_strength: float representing how strong the
            reprocessed signal is with respect to the continuum emission
        :param corona_height: None or int/float. If None, the initialized corona height
            will be used. Otherwise, represents the height of the flare in units R_g =
            GM/c^2
        :param axis_offset_in_gravitational_radii: Axis offset of the flaring event in
            units R_g = GM/c^2
        :param angle_offset_in_gravitational_radii: Degree rotation around the axis of
            symmetry of the flaring event. Zero degrees represents the flare nearer to
            the observer for inclined disks, while 180 degrees represnts the far side of
            the accretion disk
        :return: a list of snapshots of the accretion disk at each time step. Note that
            this is an experimental method.
        """

        rest_frame_wavelength_in_nm = (
            observer_frame_wavelength_in_nm / (1 + self.redshift_source) / self.g_array
        )

        if corona_height is None:
            corona_height = self.corona_height

        radiation_patterns = generate_snapshots_of_radiation_pattern(
            rest_frame_wavelength_in_nm,
            time_stamps,
            self.temp_array,
            self.radii_array,
            self.phi_array,
            self.g_array,
            self.smbh_mass_exp,
            driving_signal,
            driving_signal_fractional_strength,
            corona_height,
            self.inclination_angle,
            axis_offset_in_gravitational_radii,
            angle_offset_in_degrees,
            self.height_array,
            self.albedo_array,
        )

        radiation_patterns_flux_projections = []

        for current_pattern in radiation_patterns:
            current_projection = FluxProjection(
                current_pattern,
                observer_frame_wavelength_in_nm,
                self.smbh_mass_exp,
                self.redshift_source,
                self.r_out_in_gravitational_radii,
                self.inclination_angle,
                OmM=self.OmM,
                H0=self.H0,
            )

            radiation_patterns_flux_projections.append(current_projection)

        return radiation_patterns_flux_projections

    def get_plotting_axes(self):
        """Method to get plotting axes for the accretion disk. Useful for plotting any
        of the generated arrays i.e. from calculate_time_lag_array,
        construct_accretion_disk_transfer_function( ...,

        return_response_array_and_lags=True ), self.temp_array, etc.

        :return: X, Y arrays to be used with plt.contourf()
        """

        x_ax = np.linspace(
            -self.r_out_in_gravitational_radii,
            self.r_out_in_gravitational_radii,
            np.size(self.radii_array, 0),
        )
        X, Y = np.meshgrid(x_ax, x_ax)

        return X, Y
