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

    # unlike the blr, we only need one boundary
    def add_streamline_bounded_region(self, Streamline):

        # assure vertical coordinates are equal, otherwise interpolation is not well defined
        assert Streamline.height_step == self.height_step
        assert Streamline.max_height == self.max_height

        # Allow adaptive max radius to relevant radii (due to power law dependence, use limit of (r**2+z**2)**0.5
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

        # all density_grid values are 0 or 1, so give them weighting based on desired power law dependence
        R, Z = np.asarray(
            np.meshgrid(self.radii_values, self.height_values, indexing="ij")
        )
        radii = np.sqrt(R**2 + Z**2)
        self.density_grid *= radii**self.power_law_density_dependence

        self.torus_array_shape = np.shape(self.density_grid)

    def define_extinction_coefficients(
        self, rest_frame_wavelengths=None, extinction_coefficients=None
    ):

        self.rest_frame_wavelengths = rest_frame_wavelengths
        self.extinction_coefficients = extinction_coefficients

    def interpolate_to_extinction_at_wavelength(self, observer_frame_wavelength):
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
