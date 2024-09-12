from astropy import units as u
from astropy import constants as const
import numpy as np
from amoeba.Util.util import (
    calculate_gravitational_radius,
    calculate_luminosity_distance,
    calculate_angular_diameter_distance,
)


class FluxProjection:

    def __init__(
        self,
        flux_array,
        observer_frame_wavelength_in_nm,
        smbh_mass_exp,
        redshift_source,
        r_out_in_gravitational_radii,
        inclination_angle,
        Om0=0.3,
        H0=70,
    ):
        """Initialize the projection."""

        self.flux_array = flux_array
        self.total_flux = np.sum(self.flux_array)
        if not isinstance(observer_frame_wavelength_in_nm, (u.Quantity, list)):
            observer_frame_wavelength_in_nm *= u.nm
        self.observer_frame_wavelength_in_nm = observer_frame_wavelength_in_nm
        self.rest_frame_wavelength_in_nm = (
            round(observer_frame_wavelength_in_nm.value / (1 + redshift_source), 1)
            * u.nm
        )

        self.smbh_mass_exp = smbh_mass_exp
        self.mass = 10**smbh_mass_exp * const.M_sun.to(u.kg)
        self.r_out_in_gravitational_radii = r_out_in_gravitational_radii
        self.redshift_source = redshift_source
        self.inclination_angle = inclination_angle
        self.rg = calculate_gravitational_radius(10**self.smbh_mass_exp)
        self.pixel_size = (
            2 * self.rg * self.r_out_in_gravitational_radii / np.size(flux_array, 0)
        )
        self.Om0 = Om0
        self.little_h = H0 / 100
        self.lum_dist = calculate_luminosity_distance(
            self.redshift_source, Om0=self.Om0, little_h=self.little_h
        )
        self.ang_diam_dist = calculate_angular_diameter_distance(
            self.redshift_source, Om0=self.Om0, little_h=self.little_h
        )
