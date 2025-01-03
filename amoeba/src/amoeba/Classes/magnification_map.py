from astropy import units as u
from astropy import constants as const
import numpy as np
from astropy.io import fits
from amoeba.src.amoeba.Util.util import (
    calculate_angular_diameter_distance,
    calculate_einstein_radius_in_meters,
    convert_1d_array_to_2d_array,
    perform_microlensing_convolution,
    pull_value_from_grid,
    extract_light_curve,
    calculate_microlensed_transfer_function,
)

# Make a decision if I weight all output maps / light curves by macromagnification
# Right now, it's not implimented.


class MagnificationMap:

    def __init__(
        self,
        redshift_source,
        redshift_lens,
        magnification_array,
        convergence,
        shear,
        mean_microlens_mass_in_kg=1 * const.M_sun.to(u.kg),
        total_microlens_einstein_radii=25,
        OmM=0.3,
        H0=70,
        name="",
    ):
        """This class is for raw magnification maps, which should be 2d representations
        of the magnification at each pixel. The magnification map can be input as either
        magnification_array or ray counts.

        :param redshift_source: Redshift of the distant source being lensed
        :param redshift_lens: Redshift of the lensing galaxy
        :param magnification_array: 2d numpy array, fits file, or binary file
            representing the magnification map
        :param convergence: Convergence of the lensing potential at the location of the
            image
        :param shear: Shear of the lensing potential at the location of the image
        :param mean_microlens_mass_in_kg: Average microlens mass
        :param total_microlens_einstein_radii: Number of Einstein radii the
            magnification map covers along one square edge.
        :param OmM: Cosmological fraction of mass
        :param OmL: Cosmological fraction of dark energy
        :param H0: Hubble constant in units km/s/Mpc
        :param name: Name space.
        """

        self.name = name
        self.redshift_source = redshift_source
        self.redshift_lens = redshift_lens
        self.convergence = convergence
        self.shear = shear
        self.total_microlens_einstein_radii = total_microlens_einstein_radii
        self.mean_microlens_mass_in_kg = mean_microlens_mass_in_kg
        self.OmM = OmM
        self.H0 = H0
        self.little_h = H0 / 100
        self.einstein_radius_in_meters = calculate_einstein_radius_in_meters(
            self.redshift_lens,
            self.redshift_source,
            mean_microlens_mass_in_kg=self.mean_microlens_mass_in_kg,
            OmM=OmM,
            little_h=self.little_h,
        )

        # support opening stored magnification maps
        if isinstance(magnification_array, (np.ndarray, list)):
            if np.ndim(magnification_array) == 1:
                self.ray_map = convert_1d_array_to_2d_array(magnification_array)
            elif np.ndim(magnification_array) == 2:
                self.ray_map = magnification_array
        elif magnification_array[-4:] == "fits":
            with fits.open(magnification_array) as f:
                self.ray_map = f[0].data
        elif magnification_array[-4:] == ".dat" or magnification_array[-4:] == ".bin":
            with open(magnification_array, "rb") as f:
                extracted_magnification_array = np.fromfile(f, "i", count=-1, sep="")
                self.ray_map = convert_1d_array_to_2d_array(
                    extracted_magnification_array
                )
        else:
            print(
                "Invalid file name / format. Please pass in the pathway to a .fits or .dat file or a Numpy array"
            )

        self.resolution = np.size(self.ray_map, 0)
        self.macro_magnification = 1 / ((1 - self.convergence) ** 2.0 - self.shear**2.0)

        self.ray_to_mag_ratio = np.sum(self.ray_map) / np.size(self.ray_map)
        self.magnification_array = self.ray_map / self.ray_to_mag_ratio

        self.pixel_size = (
            self.einstein_radius_in_meters
            * self.total_microlens_einstein_radii
            / self.resolution
        )
        self.pixel_shift = 0

    def convolve_with_flux_projection(
        self, FluxProjection, relative_orientation=False, random_seed=None
    ):
        """Prepare the convolution between this magnification map and a FluxProjection.

        :param FluxProjection: FluxProjection object representing the source plane flux
        density.
        :param relative_orientation: Angle to rotate the projection with respect to the magnification map
            in degrees.
        :return: Tuple representing...
            Convolution between magnification map and surface flux density
            Pixel shift in order to keep all positions with respect to smbh
        """

        convolved_map = ConvolvedMap(
            self, FluxProjection, relative_orientation=False, random_seed=random_seed
        )

        return convolved_map

    def pull_value_from_grid(self, x_val, y_val):
        """Read off the magnification at some position :param x_val: X value on the
        magnification map in pixels :param y_val: Y value on the magnification map in
        pixels :return: Magnification at location (x_val, y_val)."""

        return pull_value_from_grid(
            self.magnification_array, x_val + self.pixel_shift, y_val + self.pixel_shift
        )

    def pull_light_curve(
        self,
        effective_transverse_velocity,
        light_curve_duration_in_years,
        x_start_position=None,
        y_start_position=None,
        phi_travel_direction=None,
        return_track_coords=False,
        random_seed=None,
    ):
        """Calculate the values along a light curve :param vtrans: effective transverse
        velocity in km/s :param time: duration of light curve in years :param x_start: x
        coordinate to start at :param y_start: y coordinate to start at :param
        phi_angle: direction to travel in :param returntrack: Bool to return the pixel
        locations as well as the light curve :return: Either a list representing the
        light curve or (for returntrack) a tuple representing a list for the light
        curve, and a list for the positions."""

        return extract_light_curve(
            self.magnification_array,
            self.pixel_size,
            effective_transverse_velocity,
            light_curve_duration_in_years,
            pixel_shift=self.pixel_shift,
            x_start_position=x_start_position,
            y_start_position=y_start_position,
            phi_travel_direction=phi_travel_direction,
            return_track_coords=return_track_coords,
            random_seed=random_seed,
        )

    def calculate_microlensed_transfer_function(
        self,
        AccretionDisk,
        observer_frame_wavelength_in_nm,
        corona_height=None,
        relative_orientation=0,
        x_position=None,
        y_position=None,
        axis_offset_in_gravitational_radii=0,
        angle_offset_in_degrees=0,
        OmM=0.3,
        little_h=0.7,
        return_response_array_and_lags=False,
        return_descaled_response_array_and_lags=False,
        random_seed=None,
    ):
        """Calculate the microlensed response map of an accretion disk.

        :param AccretionDisk: AccretionDisk object
        :param wavelength: wavelength in nanometers to calculate the response at
        :param corona_height: If none, will take the value stored in the AccretionDisk
            object. Otherwise will override the coronaheight in units rg = GM/c^2
        :param relative_orientation: Degree rotation of the AccretionDisk with respect
            to the magnification map
        :param x_position: None or pixel position of the disk. If None, a random
            position is used
        :param y_position: None or pixel position of the disk. If None, a random
            position is used
        :param axis_offset_in_gravitational_radii: Off axis position to calculate the
            response map from
        :param angle_offset_in_degrees: Degree rotation of the axis offset
        :param unit: String or astropy unit representing the units of the response map
        :param scaleratio: Smoothing factor used to calculate the response map
        :param smooth: Bool to use a smoothing kernel on the response function. Note
            this can change the properties of the response kernel.
        :param returnmaps: Bool used to return the spatially resolved maps associated
            with the response function.
        :param jitters: Bool used to allow for calculation of the response map at a
            random point in each pixel on the source plane as opposed to on the grid
            axes.
        :param source_plane: Bool to signify if time delays are calculated in the source
            or observer's frame of reference.
        :return: By default, a microlensed transfer function represented by a list.
            Using returnmaps will return a tuple of 2d maps representing the magnified
            surface response function and the time lags with respect to the lamppost.
        """

        if corona_height is None:
            corona_height = AccretionDisk.corona_height

        rest_wavelength_in_nm = observer_frame_wavelength_in_nm / (
            1 + self.redshift_source
        )

        return calculate_microlensed_transfer_function(
            self.magnification_array,
            self.redshift_lens,
            AccretionDisk.redshift_source,
            rest_wavelength_in_nm,
            AccretionDisk.temp_array,
            AccretionDisk.radii_array,
            AccretionDisk.phi_array,
            AccretionDisk.g_array,
            AccretionDisk.inclination_angle,
            AccretionDisk.smbh_mass_exp,
            corona_height,
            mean_microlens_mass_in_kg=self.mean_microlens_mass_in_kg,
            number_of_microlens_einstein_radii=self.total_microlens_einstein_radii,
            number_of_smbh_gravitational_radii=AccretionDisk.r_out_in_gravitational_radii,
            relative_orientation=relative_orientation,
            OmM=OmM,
            little_h=little_h,
            axis_offset_in_gravitational_radii=axis_offset_in_gravitational_radii,
            angle_offset_in_degrees=angle_offset_in_degrees,
            height_array=AccretionDisk.height_array,
            albedo_array=AccretionDisk.albedo_array,
            x_position=x_position,
            y_position=y_position,
            return_response_array_and_lags=return_response_array_and_lags,
            return_descaled_response_array_and_lags=return_descaled_response_array_and_lags,
            random_seed=random_seed,
        )


class ConvolvedMap(MagnificationMap):
    """This class represents the convoultion of a magnification map with a surface flux
    density."""

    def __init__(
        self,
        magnification_map,
        projected_flux_distribution,
        relative_orientation=False,
        random_seed=None,
    ):

        output_convolution, pixel_shift = perform_microlensing_convolution(
            magnification_map.magnification_array,
            projected_flux_distribution.flux_array,
            redshift_lens=magnification_map.redshift_lens,
            redshift_source=magnification_map.redshift_source,
            smbh_mass_exponent=projected_flux_distribution.smbh_mass_exp,
            mean_microlens_mass_in_kg=magnification_map.mean_microlens_mass_in_kg,
            number_of_microlens_einstein_radii=magnification_map.total_microlens_einstein_radii,
            number_of_smbh_gravitational_radii=projected_flux_distribution.r_out_in_gravitational_radii,
            relative_orientation=relative_orientation,
            OmM=magnification_map.OmM,
            little_h=magnification_map.H0 / 100,
            random_seed=random_seed,
        )

        self.pixel_size = magnification_map.pixel_size
        self.total_microlens_einstein_radii = (
            magnification_map.total_microlens_einstein_radii
        )
        self.mean_microlens_mass_in_kg = magnification_map.mean_microlens_mass_in_kg
        self.resolution = magnification_map.resolution
        self.magnification_array = output_convolution
        self.pixel_shift = pixel_shift
        self.macro_magnification = magnification_map.macro_magnification
        self.redshift_lens = magnification_map.redshift_lens

        self.smbh_mass_exp = projected_flux_distribution.smbh_mass_exp
        self.inclination_angle = projected_flux_distribution.inclination_angle
        self.rg = projected_flux_distribution.rg
        self.observer_frame_wavelength_in_nm = (
            projected_flux_distribution.observer_frame_wavelength_in_nm
        )
        self.relative_orientation = relative_orientation
        self.redshift_source = projected_flux_distribution.redshift_source
