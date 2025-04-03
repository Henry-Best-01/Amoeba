import numpy as np
from astropy import units as u
from astropy import constants as const
from amoeba.Classes.blr_streamline import Streamline
from amoeba.Classes.flux_projection import FluxProjection
from amoeba.Util.util import project_blr_to_source_plane


class Torus:

    def __init__(
        self,
        smbh_mass_exp,
        max_height,
        redshift_source,
        radial_step=10,
        height_step=10,
        power_law_density_dependence=0,
        OmM=0.3,
        H0=70,
        **kwargs
    ):
        """Generate a model torus designed to be projected into the plane of the
        accretion disk. Using this column density can be used as an estimate for how
        much light is absorbed by the dust. This is an experimental class.

        Assumes that there is absorption such that:
            flux_abs = flux_emit * (1 - e^{-\rho_{col} / abs_strength})

        where:
            flux_abs is the absorbed flux (an array)
            flux_emit is the emitted flux (an array)
            e is Euler's number
            \rho_{col} is the column density (an array)
            abs_strength is the absorption strength (int/float)

        Note that this can only absorb if there is source flux to absorb. Therefore, the
        FluxProjection of project_extinction_to_source_plane() is cast to an approximate
        magnitude change via -2.5 * np.log10(flux_abs)

        :param smbh_mass_exp: solution to log10(m_smbh / m_sun), which sets
            the size scales of all components in the torus.
        :param max_height: maximum height of the torus in the Z direction, in
            units R_g
        :param redshift_source: redshift of the torus
        :param radial_step: radial resolution in units R_g
        :param height_step: resolution in the Z-axis in units R_g
        :param power_law_density_dependence: defines the density profile as a function
            of radius such that \rho \propto r^{power_law_density_dependence}. Zero means
            the density is assumed to be constant throughout the torus. Positive values
            represent increasing densities as you go further out.
        :param OmM: mass component of the universe's energy budget
        :param H0: Hubble constant in units km/s/Mpc
        """
        self.smbh_mass_exp = smbh_mass_exp
        self.redshift_source = redshift_source
        self.OmM = OmM
        self.H0 = H0
        self.max_height = max_height
        self.radial_step = radial_step
        self.height_step = height_step
        self.max_radius = 0
        self.density_grid = np.zeros(
            (self.max_radius // radial_step + 1, self.max_height // height_step + 1)
        )
        self.radii_values = np.linspace(
            0, self.max_radius, self.max_radius // radial_step + 1
        )
        self.height_values = np.linspace(
            0, self.max_height, self.max_height // height_step + 1
        )

        self.torus_array_shape = np.shape(self.density_grid)
        self.mass = 10 ** (self.smbh_mass_exp) * const.M_sun.to(u.kg)
        self.power_law_density_dependence = power_law_density_dependence

        if (
            "rest_frame_wavelengths" in kwargs.keys()
            and "extinction_coefficients" in kwargs.keys()
        ):
            self.define_extinction_coefficients(
                rest_frame_wavelengths=kwargs["rest_frame_wavelengths"],
                extinction_coefficients=kwargs["extinction_coefficients"],
            )
        else:
            self.define_extinction_coefficients()

    def add_streamline_bounded_region(self, Streamline):
        """Add a streamline representing the boundary of the dust to the torus.

        :param Streamline: Streamline object to use as the inner boundary condition
        :return: True if successful
        """

        assert Streamline.height_step == self.height_step
        assert Streamline.max_height == self.max_height

        new_max_radius = np.sqrt(
            np.max(Streamline.radii_values) ** 2 + np.max(Streamline.height_values) ** 2
        )

        if new_max_radius > self.max_radius:
            self.max_radius = int(new_max_radius)

            dummygrid = np.zeros(
                (
                    self.max_radius // self.radial_step + 1,
                    self.max_height // self.height_step + 1,
                )
            )
            previous_maximum = self.max_radius // self.radial_step + 1
            dummygrid[:previous_maximum, :] = self.density_grid
            self.density_grid = dummygrid

            self.radii_values = np.linspace(
                0, self.max_radius, self.max_radius // self.radial_step + 1
            )

        for hh in range(np.size(self.height_values)):
            mask = self.radii_values >= Streamline.radii_values[hh]
            self.density_grid[np.argmax(mask) :, hh] = 1

        R, Z = np.asarray(
            np.meshgrid(self.radii_values, self.height_values, indexing="ij")
        )
        radii = np.sqrt(R**2 + Z**2)
        self.density_grid *= radii**self.power_law_density_dependence

        self.torus_array_shape = np.shape(self.density_grid)

        return True

    def define_extinction_coefficients(
        self, rest_frame_wavelengths=None, extinction_coefficients=None
    ):
        """Define the extinction coefficients for a list of wavelengths.

        :param rest_frame_wavelengths: list/array of wavelengths with representative
            extinction coefficients.
        :param extinction_coefficients: list/array of extinction coefficients. Must be
            the same shape as rest_frame_wavelengths.
        :return: True if successful
        """
        self.rest_frame_wavelengths = rest_frame_wavelengths
        self.extinction_coefficients = extinction_coefficients

        return True

    def interpolate_to_extinction_at_wavelength(self, observer_frame_wavelength):
        """Use linear interpolation of the extinction curve to determine the extinction
        at a particular wavelength.

        :param observer_frame_wavelength: observer frame wavelength in nm
        :return: int representing the extinction coefficient at the given wavelength
        """

        rest_frame_wavelength = observer_frame_wavelength / (1 + self.redshift_source)

        if self.rest_frame_wavelengths is None:
            print(
                "please provide an array of extinction coefficients and an array of wavelengths"
            )
            return False

        extinction_interpolation = np.interp(
            rest_frame_wavelength,
            self.rest_frame_wavelengths,
            self.extinction_coefficients,
        )

        return extinction_interpolation

    def project_density_to_source_plane(self, inclination_angle):
        """Project the torus' density into a FluxProjection object.

        :param inclination_angle: inclination of the torus w.r.t. the observer in
            degrees
        :return: a 2 dimensional array representing the projected column density of the
            torus. Note that this is not representative of an emission or absorption, so
            a FluxProjection object is not returned.
        """

        projection = project_blr_to_source_plane(
            self.density_grid,
            np.zeros(np.shape(self.density_grid)),
            np.zeros(np.shape(self.density_grid)),
            inclination_angle,
            self.smbh_mass_exp,
            weighting_grid=self.density_grid,
            radial_resolution=self.radial_step,
            vertical_resolution=self.height_step,
        )[0]

        return projection

    def project_extinction_to_source_plane(
        self, inclination_angle, observer_frame_wavelength
    ):
        """Project the torus' modelled extinction to the source plane.

        :param inclination_angle: inclination of the torus w.r.t. the observer in
            degrees
        :param observer_frame_wavelength: observer frame wavelength in nm
        :return: FluxProjection object representing the amount of flux absorbed by the
            torus
        """

        extinction_strength = self.interpolate_to_extinction_at_wavelength(
            observer_frame_wavelength
        )
        projected_extinction_array = self.project_density_to_source_plane(
            inclination_angle
        )

        extinction_mask = projected_extinction_array > 0

        projected_extinction_array = np.nan_to_num(
            -2.5
            * np.log10(1 - np.exp(-projected_extinction_array / extinction_strength))
            * extinction_mask
        )

        projected_extinction = FluxProjection(
            projected_extinction_array,
            observer_frame_wavelength,
            self.smbh_mass_exp,
            self.redshift_source,
            np.max(self.radii_values),
            inclination_angle,
            OmM=self.OmM,
            H0=self.H0,
        )

        return projected_extinction
