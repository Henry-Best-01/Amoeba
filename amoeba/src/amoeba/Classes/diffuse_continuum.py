import numpy as np
from amoeba.Util.util import (
    calculate_luminosity_distance,
    calculate_gravitational_radius,
    calculate_time_lag_array,
)
from amoeba.Classes.flux_projection import FluxProjection
import astropy.constants as const
import astropy.units as u


class DiffuseContinuum:

    def __init__(
        self,
        smbh_mass_exp=None,
        redshift_source=None,
        inclination_angle=None,
        radii_array=None,
        phi_array=None,
        cloud_density_radial_dependence=0,
        cloud_density_array=None,
        OmM=0.3,
        H0=70,
        r_in_in_gravitational_radii=None,
        r_out_in_gravitational_radii=None,
        name="",
        **kwargs
    ):
        """Object representing the diffuse continuum component of the AGN. The diffuse
        continuum model follows Korista+Goad 2019, where the increase in time lag
        is defined as:

        tau_lam(inci + DC) \approx tau_lam(DC) (1 - A)x / (1 - Ax)

        where:
        tau_lam(DC) is the time delay of the diffuse continuum and depends
            heavily on the cloud_density_array
        A is a constant on the range (0, 1)
        x is the fractional contribution of the DC to the total light

        These must be set using methods:
        A = self.set_responsivity_constant()
        x = self.set_emissivity()


        :param smbh_mass_exp: mass exponent of the sumpermassive black hole at the
            center of the disk expressed as log_10(M / M_sun). Typical ranges are 6-11
            for AGN.
        :param redshift_source: redshift of the diffuse continuum
        :param inclination_angle: inclination of the object w.r.t. the observer, in degrees
        :param radii_array: a 2d representation of the radii of the diffuse continuum, in
            gravitational radii R_g = GM/c^2
        :param phi_array: a 2d representation of the azimuths of the diffuse continuum
            in radians.
        :param cloud_density_radial_dependence: radial dependence of the clound density
            according to \rho \propto r^{alpha} where alpha is cloud_density_radial_dependence
        :param cloud_density_array: either an int, float, or 2d representation of the
            diffuse continuum cloud density.
        :param OmM: Cosmological parameter representing the mass fraction of the
            universe
        :param H0: Hubble constant in units of km/s/Mpc
        :param r_in_in_gravitational_radii: inner radius of the diffuse continuum in R_g.
            Note that this parameter dominates the time lag of the diffuse continuum.
        :param r_out_in_gravitational_radii: maximum radius of the diffuse continuum, in
            gravitational radii
        :param name: Name space
        """

        self.name = name
        self.smbh_mass_exp = smbh_mass_exp
        self.mass = 10**smbh_mass_exp * const.M_sun.to(u.kg)
        self.redshift_source = redshift_source
        self.inclination_angle = inclination_angle
        self.r_out_in_gravitational_radii = r_out_in_gravitational_radii
        self.r_in_in_gravitational_radii = r_in_in_gravitational_radii

        if r_out_in_gravitational_radii is not None:
            radial_mask = radii_array <= r_out_in_gravitational_radii
        else:
            radial_mask = np.ones(np.shape(radii_array))
            self.r_out_in_gravitational_radii = np.max(radii_array)

        if r_in_in_gravitational_radii is not None:
            radial_mask = radial_mask * (radii_array >= r_in_in_gravitational_radii)
            self.r_in_in_gravitational_radii = r_in_in_gravitational_radii
        else:
            self.r_in_in_gravitational_radii = 10

        self.radii_array = radii_array * radial_mask
        self.phi_array = phi_array
        self.radial_mask = radial_mask

        if cloud_density_array is not None:
            self.cloud_density_radial_dependence = None
            self.cloud_density_array = cloud_density_array * radial_mask
        elif isinstance(cloud_density_radial_dependence, (float, int)):
            self.cloud_density_array = (
                np.nan_to_num(self.radii_array**cloud_density_radial_dependence)
                * self.radial_mask
            )
            self.cloud_density_radial_dependence = cloud_density_radial_dependence
        else:
            raise ValueError(
                "Cloud density array and cloud density radial dependence cannot both be None type"
            )

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
            / np.size(self.radii_array, 0)
        )
        self.kwargs = kwargs

        if "emissivity_etas" in self.kwargs and "rest_frame_wavelengths" in self.kwargs:
            self.set_emissivity(
                rest_frame_wavelengths=self.kwargs["rest_frame_wavelengths"],
                emissivity_etas=self.kwargs["emissivity_etas"],
            )
        else:
            self.set_emissivity()

        if "responsivity_constant" in self.kwargs:
            self.set_responsivity_constant(
                responsivity_constant=self.kwargs["responsivity_constant"]
            )
        else:
            self.set_responsivity_constant()

    def set_emissivity(self, rest_frame_wavelengths=None, emissivity_etas=None):
        """Define the diffuse continuum's emission spectrum according to a rest frame
        wavelength spectrum. Required for estimating the increase in time lags.
        Determines the value of "x" in the equation in the init function.

        :param rest_frame_wavelengths: list representing the rest frame wavelengths in
            nanometers. Should cover the range of wavelengths expected to be observed.
        :param emissivity_etas: list representing the emissivities of the diffuse
            continuum clouds at each wavelength in rest_frame_wavelengths. Must be the
            same length as rest_frame_wavelengths.
        :return: True if successful
        """

        self.rest_frame_wavelengths = rest_frame_wavelengths
        self.emissivity_etas = emissivity_etas

        return True

    def set_responsivity_constant(self, responsivity_constant=1):
        """Define the responsivity constant of the BLR according to Korista+Goad 2019.
        Sets the value of "A" in the equation in the init function.

        :param responsivity_constant: define A, the percentage of flux that comes from
            the diffuse continuum w.r.t. the total flux. Must be between 0 and 1.
        :return: True if successful
        """

        assert responsivity_constant >= 0
        assert responsivity_constant <= 1

        self.responsivity_constant = responsivity_constant

        return True

    def interpolate_spectrum_to_wavelength(self, observer_frame_wavelength_in_nm):
        """Interpolates known spectra to a particular wavelength via linear
        interpolation.

        :param observer_frame_wavelength_in_nm: observer frame wavelength in nm
        :return: emissivity at observer frame wavenegth
        """

        if self.rest_frame_wavelengths is None:
            print("please initialize the diffuse continuum spectrum")
            return False

        rest_frame_wavelength_in_nm = observer_frame_wavelength_in_nm / (
            1 + self.redshift_source
        )
        emissivity_interpolation = np.interp(
            rest_frame_wavelength_in_nm,
            self.rest_frame_wavelengths,
            self.emissivity_etas,
        )

        return emissivity_interpolation

    def get_diffuse_continuum_emission(
        self, observer_frame_wavelength_in_nm, incident_continuum_weighting=1
    ):
        """Produce a FluxProjection object of the diffuse continuum under the assumption
        that there is no inclination dependence. Primarily used for the AGN projection
        pipeline. Note that there is a geometric weighting of r^{-2} from the incident
        light.

        :param observer_frame_wavelength_in_nm: wavelength the diffuse continuum is
            observed at in nanometers
        :param incident_continuum_weighting: optional weighting factor to scale the flux
            array by
        :return: FluxProjection object containing metadata of the diffuse continuum and
            the emission array
        """
        rest_frame_wavelength_in_nm = observer_frame_wavelength_in_nm / (
            1 + self.redshift_source
        )
        emission_array = self.cloud_density_array * np.nan_to_num(
            self.radii_array ** (-2)
        )
        emission_array /= np.sum(emission_array)
        emission_array *= (
            self.interpolate_spectrum_to_wavelength(rest_frame_wavelength_in_nm)
            * incident_continuum_weighting
        )

        projection = FluxProjection(
            emission_array * self.pixel_size ** (-2),
            observer_frame_wavelength_in_nm,
            self.smbh_mass_exp,
            self.redshift_source,
            self.r_out_in_gravitational_radii,
            self.inclination_angle,
            OmM=self.OmM,
            H0=self.H0,
        )

        return projection

    def get_diffuse_continuum_mean_lag(self, observer_frame_wavelength_in_nm):
        """Calculate the diffuse continuum's mean time lag at a given wavelength in the
        observer's frame. The integration constant is assumed to enforce tau_min = R_in
        / c.

        :param observer_frame_wavelength_in_nm: observer frame wavelength in nanometers
        :return: float representing the mean increase in time delay in units R_g / c
        """

        if self.cloud_density_radial_dependence is not None:

            integration_constant = self.r_in_in_gravitational_radii

            if self.cloud_density_radial_dependence < 0:
                integration_multiplicative_constant = (2 * np.pi) / (
                    self.cloud_density_radial_dependence
                )

                return (
                    integration_multiplicative_constant
                    * (
                        self.r_out_in_gravitational_radii
                        ** (self.cloud_density_radial_dependence)
                        - self.r_in_in_gravitational_radii
                        ** (self.cloud_density_radial_dependence)
                    )
                    + integration_constant
                )

            elif self.cloud_density_radial_dependence == 0:
                integration_multiplicative_constant = 2 * np.pi
                return (
                    integration_multiplicative_constant
                    * np.log(
                        self.r_out_in_gravitational_radii
                        / self.r_in_in_gravitational_radii
                    )
                    + integration_constant
                )

            else:
                integration_multiplicative_constant = (2 * np.pi) / (
                    self.cloud_density_radial_dependence
                )
                return (
                    integration_multiplicative_constant
                    * (
                        self.r_out_in_gravitational_radii
                        ** (self.cloud_density_radial_dependence)
                        - self.r_in_in_gravitational_radii
                        ** (self.cloud_density_radial_dependence)
                    )
                    + integration_constant
                )

        else:
            time_delays = calculate_time_lag_array(
                self.radii_array, self.phi_array, 0, 0
            )
            emissivity = self.get_diffuse_continuum_emission(
                observer_frame_wavelength_in_nm
            ).flux_array

            diffuse_continuum_transfer_function = np.histogram(
                time_delays,
                range=(0, np.max(time_delays)),
                bins=int(np.max(time_delays)),
                weights=np.nan_to_num(emissivity),
                density=True,
            )[0]

            diffuse_continuum_transfer_function[0] = 0

            diffuse_continuum_transfer_function /= np.sum(
                diffuse_continuum_transfer_function
            )

            mean_delay = np.sum(
                np.linspace(0, np.max(time_delays) - 1, int(np.max(time_delays)))
                * diffuse_continuum_transfer_function
            )

            return mean_delay

    def get_diffuse_continuum_lag_contribution(
        self,
        observer_frame_wavelength_in_nm,
    ):
        """Estimate the mean increase in time lag for an accretion disk due to the
        diffuse continuum, which must be added by the mean lag of the continuum. Solves
        the equation from Korista+Goad 2019 in the init function.

        :param observer_frame_wavelength_in_nm: observer frame wavelength in nanometers
        :return: increase in mean time lag at an observer frame wavelength in R_g / c
        """

        tau_diffuse = self.get_diffuse_continuum_mean_lag(
            observer_frame_wavelength_in_nm
        )
        emissivity = self.interpolate_spectrum_to_wavelength(
            observer_frame_wavelength_in_nm
        )
        const_a = self.responsivity_constant

        tau_dc_vs_continuum = (
            tau_diffuse * (1 - const_a) * emissivity / (1 - const_a * emissivity)
        )

        return tau_dc_vs_continuum
