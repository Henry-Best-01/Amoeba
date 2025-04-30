from astropy import units as u
from astropy import constants as const
import numpy as np
from amoeba.Util.util import (
    calculate_gravitational_radius,
    calculate_luminosity_distance,
    calculate_angular_diameter_distance,
)
from skimage.transform import rescale
from math import isinf


class FluxProjection:

    def __init__(
        self,
        flux_array,
        observer_frame_wavelength_in_nm,
        smbh_mass_exp,
        redshift_source,
        r_out_in_gravitational_radii,
        inclination_angle,
        OmM=0.3,
        H0=70,
        **kwargs
    ):
        """Initialize the projection. This object carries the projected array as well as
        important metadata.

        :param flux_array: a 2 dimensional representation of the flux array in arbitrary
            units. Spacing along the X and Y direction must be even.
        :param observer_frame_wavelength_in_nm: Wavelength or range of wavelengths this
            flux projection is associated with, in units nm, in the observer's frame of
            reference.
        :param smbh_mass_exp: solution to log10(m_smbh / m_sun), determines the distance
            of one gravitational radius and therefore sets the size scale of the
            flux_array
        :param redshift_source: redshift of the object
        :param r_out_in_gravitational_radii: maximum radius of the flux_array along each
            axis in units of R_g.
        :param inclination_angle: inclination of the source in degrees
        :param OmM: mass contribution to the energy budget of the universe
        :param H0: Hubble constant in units km/s/Mpc
        """

        self.flux_array = np.asarray(flux_array)
        self.observer_frame_wavelength_in_nm = observer_frame_wavelength_in_nm
        if isinstance(observer_frame_wavelength_in_nm, (list, np.ndarray)):
            min_wavelength = round(
                observer_frame_wavelength_in_nm[0] / (1 + redshift_source)
            )
            if not isinf(observer_frame_wavelength_in_nm[1]):
                max_wavelength = round(
                    observer_frame_wavelength_in_nm[1] / (1 + redshift_source)
                )
            else:
                max_wavelength = observer_frame_wavelength_in_nm[1]
            rest_frame_wavelength_in_nm = [
                min_wavelength,
                max_wavelength,
            ]
        else:
            rest_frame_wavelength_in_nm = round(
                observer_frame_wavelength_in_nm / (1 + redshift_source)
            )
        self.rest_frame_wavelength_in_nm = rest_frame_wavelength_in_nm
        self.smbh_mass_exp = smbh_mass_exp
        self.mass = 10**smbh_mass_exp * const.M_sun.to(u.kg)
        self.r_out_in_gravitational_radii = r_out_in_gravitational_radii
        self.redshift_source = redshift_source
        self.inclination_angle = inclination_angle
        self.rg = calculate_gravitational_radius(10**self.smbh_mass_exp)
        self.pixel_size = (
            2 * self.rg * self.r_out_in_gravitational_radii / np.size(flux_array, 0)
        )
        self.total_flux = np.sum(self.flux_array * self.pixel_size**2)
        self.OmM = OmM
        self.H0 = H0
        self.lum_dist = calculate_luminosity_distance(
            self.redshift_source, OmM=self.OmM, H0=self.H0
        )
        self.ang_diam_dist = calculate_angular_diameter_distance(
            self.redshift_source, OmM=self.OmM, H0=self.H0
        )

    def add_flux_projection(self, SecondProjection):
        """Join this FluxProjection with another FluxProjection via simple addition. The
        resulting FluxProjection will have pixels the size of the larger FluxProjection
        and a maximum radius of the larger FluxProjection. Note that these are not
        necessarily the same (a smaller and coursely spaced projection joined with a
        large and finely spaced projection will result in a large and coursely spaced
        projection).

        :param SecondProjection: FluxProjection object to add to the first projection
        :return: True if successful. Note that this method modifies the parent
            FluxProjection.
        """
        assert self.redshift_source == SecondProjection.redshift_source
        assert self.smbh_mass_exp == SecondProjection.smbh_mass_exp
        assert self.inclination_angle == SecondProjection.inclination_angle

        desired_pixel_size_in_rg = np.max(
            [self.pixel_size / self.rg, SecondProjection.pixel_size / self.rg]
        )

        working_flux_projection = SecondProjection.flux_array.copy()

        self_resolution_ratio = (self.pixel_size / self.rg) / desired_pixel_size_in_rg
        second_resolution_ratio = (
            SecondProjection.pixel_size / self.rg
        ) / desired_pixel_size_in_rg

        self.flux_array = rescale(self.flux_array, self_resolution_ratio)
        self.pixel_size = desired_pixel_size_in_rg * self.rg

        working_flux_projection = rescale(
            working_flux_projection, second_resolution_ratio
        )

        new_max_radius = np.max(
            [
                self.r_out_in_gravitational_radii,
                SecondProjection.r_out_in_gravitational_radii,
            ]
        )

        desired_flux_array_shape = (
            int(2 * new_max_radius / desired_pixel_size_in_rg),
            int(2 * new_max_radius / desired_pixel_size_in_rg),
        )

        if self.r_out_in_gravitational_radii != new_max_radius:
            increase_in_pixels = int(
                (new_max_radius - self.r_out_in_gravitational_radii)
                // desired_pixel_size_in_rg
            )
            self.flux_array = np.pad(self.flux_array, increase_in_pixels)
        if SecondProjection.r_out_in_gravitational_radii != new_max_radius:
            increase_in_pixels = int(
                (new_max_radius - SecondProjection.r_out_in_gravitational_radii)
                // desired_pixel_size_in_rg
            )
            working_flux_projection = np.pad(
                working_flux_projection, increase_in_pixels
            )

        if np.shape(self.flux_array) != np.shape(working_flux_projection):
            x_diff = np.size(self.flux_array, 0) - np.size(working_flux_projection, 0)
            y_diff = np.size(self.flux_array, 1) - np.size(working_flux_projection, 1)

            if x_diff < 0:  # pragma: no cover
                working_flux_projection = working_flux_projection[:x_diff, :]
            elif x_diff > 0:  # pragma: no cover
                working_flux_projection = np.pad(
                    working_flux_projection, ((0, x_diff), (0, 0))
                )
            if y_diff < 0:  # pragma: no cover
                working_flux_projection = working_flux_projection[:, :y_diff]
            elif y_diff > 0:  # pragma: no cover
                working_flux_projection = np.pad(
                    working_flux_projection, ((0, 0), (0, y_diff))
                )

        self.r_out_in_gravitational_radii = new_max_radius
        self.flux_array += working_flux_projection
        self.total_flux = np.sum(self.flux_array * self.pixel_size**2)

        if not isinstance(self.observer_frame_wavelength_in_nm, list):
            lams_1 = [self.observer_frame_wavelength_in_nm]
        else:
            lams_1 = self.observer_frame_wavelength_in_nm
        if not isinstance(SecondProjection.observer_frame_wavelength_in_nm, list):
            lams_2 = [SecondProjection.observer_frame_wavelength_in_nm]
        else:
            lams_2 = SecondProjection.observer_frame_wavelength_in_nm

        concatenated_lams = np.concatenate((lams_1, lams_2))
        self.observer_frame_wavelength_in_nm = [
            np.min(concatenated_lams),
            np.max(concatenated_lams),
        ]

        return True

    def get_plotting_axes(self):
        """Convenience method to get the X and Y axes to plot the flux_projection in
        physical units.

        :return: X and Y arrays in units of R_g used to create contour plots via
            plt.contourf(X, Y, self.flux_array).
        """
        xax = np.linspace(
            -self.r_out_in_gravitational_radii,
            self.r_out_in_gravitational_radii,
            np.size(self.flux_array, 0),
        )
        X, Y = np.meshgrid(xax, xax)
        return X, Y
