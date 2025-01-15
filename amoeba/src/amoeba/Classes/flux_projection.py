from astropy import units as u
from astropy import constants as const
import numpy as np
from amoeba.Util.util import (
    calculate_gravitational_radius,
    calculate_luminosity_distance,
    calculate_angular_diameter_distance,
)
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
    ):
        """Initialize the projection."""

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
        self.little_h = H0 / 100
        self.lum_dist = calculate_luminosity_distance(
            self.redshift_source, OmM=self.OmM, little_h=self.little_h
        )
        self.ang_diam_dist = calculate_angular_diameter_distance(
            self.redshift_source, OmM=self.OmM, little_h=self.little_h
        )

    def get_plotting_axes(self):
        xax = np.linspace(
            -self.r_out_in_gravitational_radii,
            self.r_out_in_gravitational_radii,
            np.size(self.flux_array, 0),
        )
        X, Y = np.meshgrid(xax, xax)
        return X, Y
