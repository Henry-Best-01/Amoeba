import numpy as np
from astropy import units as u
from astropy import constants as const
from amoeba.Classes.blr_streamline import Streamline
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
        Om0=0.3,
        H0=70,
    ):
        self.smbh_mass_exp = smbh_mass_exp
        self.redshift_source = redshift_source
        self.Om0 = Om0
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

    def project_density_to_source_plane(self, inclination_angle):
        return project_blr_to_source_plane(
            self.density_grid,
            np.zeros(np.shape(self.density_grid)),
            np.zeros(np.shape(self.density_grid)),
            inclination_angle,
            self.smbh_mass_exp,
            weighting_grid=self.density_grid,
            radial_resolution=self.radial_step,
            vertical_resolution=self.height_step,
        )

    # add support for extinction calculation
