from astropy import units as u
from astropy import constants as const
import numpy as np
from astropy.io import fits
from amoeba.Util.util import (
    calculate_angular_diameter_distance,
    calculate_einstein_radius_in_meters,
    convert_1d_array_to_2d_array,
    perform_microlensing_convolution,
    pull_value_from_grid,
    extract_light_curve,
    calculate_microlensed_transfer_function,
)


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
        **kwargs
    ):
        """This class is for microlensing magnification maps, which should be 2d
        representations of the magnification at each pixel. The magnification map can be
        input as either magnification_array or ray counts.

        :param redshift_source: Redshift of the distant source being lensed
        :param redshift_lens: Redshift of the lensing galaxy
        :param magnification_array: 2d numpy array, fits file, or binary file
            representing the magnification map
        :param convergence: Convergence of the lensing potential at the location of the
            image
        :param shear: Shear of the lensing potential at the location of the image
        :param mean_microlens_mass_in_kg: Average microlens mass in kg
        :param total_microlens_einstein_radii: Number of Einstein radii the
            magnification map covers along one square edge.
        :param OmM: Cosmological fraction of mass in the energy budget of the universe
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
        self.einstein_radius_in_meters = calculate_einstein_radius_in_meters(
            self.redshift_lens,
            self.redshift_source,
            mean_microlens_mass_in_kg=self.mean_microlens_mass_in_kg,
            OmM=OmM,
            H0=self.H0,
        )

        if isinstance(magnification_array, (np.ndarray, list)):
            if np.ndim(magnification_array) == 1:
                self.ray_map = convert_1d_array_to_2d_array(magnification_array)
            elif np.ndim(magnification_array) == 2:
                self.ray_map = magnification_array
        elif magnification_array[-4:] == "fits":  # pragma: no cover
            with fits.open(magnification_array) as f:
                self.ray_map = f[0].data
        elif (
            magnification_array[-4:] == ".dat" or magnification_array[-4:] == ".bin"
        ):  # pragma: no cover
            with open(magnification_array, "rb") as f:
                extracted_magnification_array = np.fromfile(f, "i", count=-1, sep="")
                self.ray_map = convert_1d_array_to_2d_array(
                    extracted_magnification_array
                )
        else:
            print(
                "Invalid file name / format. Please pass in the pathway to a .fits or .dat file or a Numpy array"
            )
            return None

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
        Note that once the convolution is performed, all orientations are defined. In
        particular, this solidifies the orientation of the source w.r.t. the direction
        of shear of the magnification map.

        :param FluxProjection: FluxProjection object representing the source plane flux
            density.
        :param relative_orientation: Angle to rotate the projection with respect to the
            magnification map in degrees.
        :param random_seed: A random seed to set random parameters consistently
        :return: Child magnification map object representing all magnifications of the
            flux projection. Normalization of coordinates w.r.t. the smbh is considered,
            and you will probably use this primarily for its "pull_light_curve" method.
        """

        convolved_map = ConvolvedMap(
            self, FluxProjection, relative_orientation=False, random_seed=random_seed
        )

        return convolved_map

    def pull_value_from_grid(self, x_val, y_val, weight_by_macromag=False):
        """Read off the magnification at some position.

        :param x_val: X value on the magnification map in pixels
        :param y_val: Y value on the magnification map in pixels
        :param weight_by_macromag: boolean toggle to weight the values by the
            magnification map's macro magnification. This macro magnification is
            determined by the convergence and shear.
        :return: Magnification at location (x_val, y_val).
        """

        value = pull_value_from_grid(
            self.magnification_array, x_val + self.pixel_shift, y_val + self.pixel_shift
        )

        if weight_by_macromag:
            value *= self.macro_magnification
        return value

    def pull_light_curve(
        self,
        effective_transverse_velocity,
        light_curve_duration_in_years,
        x_start_position=None,
        y_start_position=None,
        phi_travel_direction=None,
        weight_by_macromag=False,
        return_track_coords=False,
        random_seed=None,
    ):
        """Calculate the values of a light curve. By default, the light curve will be
        randomly generated from the region of the magnification map not affected by
        convolutional artifacts.

        :param effective_transverse_velocity: effective transverse velocity in km/s. This
            is typically on the order of ~10^2 - 10^3 km/s. However, exceptional cases
            may exceeed 10^4 km/s.
        :param light_curve_duration_in_years: duration of light curve in years
        :param x_start_position: x coordinate to start at in pixels
        :param y_start_position: y coordinate to start at in pixels
        :param phi_travel_direction: direction to travel in, in degrees
        :param weight_by_macromag: boolean toggle to weight the values by the magnification
            map's macro magnification. This macro magnification is determined by the
            convergence and shear.
        :param return_track_coords: Bool to return the pixel locations as well as the
            light curve
        :param random_seed: Set a random seed so random values are determined
            consistently

        :return: Either a list representing the light curve or (for return_track_coords) a
            tuple representing a list for the light curve, and two lists for the x, y positions
        """

        light_curve = extract_light_curve(
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
        if weight_by_macromag:
            if not return_track_coords:
                light_curve *= self.macro_magnification
            else:
                track_x = light_curve[1]
                track_y = light_curve[2]
                light_curve = self.macro_magnification * np.asarray(light_curve[0])
                light_curve = light_curve, track_x, track_y
        return light_curve

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
        H0=70,
        return_response_array_and_lags=False,
        return_descaled_response_array_and_lags=False,
        return_magnification_map_crop=False,
        random_seed=None,
    ):
        """Calculate the transfer function of an accretion disk when the response map is
        microlensed at a particular location.

        :param AccretionDisk: AccretionDisk object
        :param observer_frame_wavelength_in_nm: observer frame wavelength in nanometers
            to calculate the response at
        :param corona_height: None or int/float. If none, will take the value stored in
            the AccretionDisk object. Otherwise will override the coronaheight in units
            R_g = GM/c^2
        :param relative_orientation: Degree rotation of the AccretionDisk with respect
            to the magnification map
        :param x_position: None or pixel position of the disk. If None, a random
            position is used
        :param y_position: None or pixel position of the disk. If None, a random
            position is used
        :param axis_offset_in_gravitational_radii: Off axis position to calculate the
            response map from
        :param angle_offset_in_degrees: Degree rotation of the axis offset
        :param OmM: mass component of the energy budget of the universe
        :param H0: Hubble constant in units km/s/Mpc
        :param return_response_array_and_lags: Boolean used to return the spatially
            resolved maps associated with the response function.
        :param return_descaled_response_array_and_lags: Similar to above, but returns
            the projections at the resolution of the magnification map. Useful for
            debugging.
        :param return_magnification_map_crop: Boolean used to return the region of
            the magnification map used to amplify the response function.
        :param random_seed: allows the user to set a random seed for reproducibility
        :return: The microlensed transfer function represented by a list.
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
            H0=H0,
            axis_offset_in_gravitational_radii=axis_offset_in_gravitational_radii,
            angle_offset_in_degrees=angle_offset_in_degrees,
            height_array=AccretionDisk.height_array,
            albedo_array=AccretionDisk.albedo_array,
            x_position=x_position,
            y_position=y_position,
            return_response_array_and_lags=return_response_array_and_lags,
            return_descaled_response_array_and_lags=return_descaled_response_array_and_lags,
            return_magnification_map_crop=return_magnification_map_crop,
            random_seed=random_seed,
        )

class ConvolvedMap(MagnificationMap):
    """This child class represents the convoultion of a magnification map with a surface
    flux density.

    Has all of the same methods as MagnificationMap
    """

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
            smbh_mass_exp=projected_flux_distribution.smbh_mass_exp,
            mean_microlens_mass_in_kg=magnification_map.mean_microlens_mass_in_kg,
            number_of_microlens_einstein_radii=magnification_map.total_microlens_einstein_radii,
            number_of_smbh_gravitational_radii=projected_flux_distribution.r_out_in_gravitational_radii,
            relative_orientation=relative_orientation,
            OmM=magnification_map.OmM,
            H0=magnification_map.H0,
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
