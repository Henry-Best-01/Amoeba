import numpy as np
import astropy.units as u
import astropy.constants as const
from amoeba.Classes.accretion_disk import AccretionDisk
from amoeba.Classes.blr import BroadLineRegion
from amoeba.Classes.blr_streamline import Streamline
from amoeba.Classes.torus import Torus
from amoeba.Classes.diffuse_continuum import DiffuseContinuum
from amoeba.Util.util import (
    create_maps,
    generate_signal_from_psd,
    calculate_gravitational_radius,
)


class Agn:

    def __init__(
        self,
        smbh_mass_exp=None,
        inclination_angle=None,
        redshift_source=None,
        OmM=0.3,
        H0=70,
        name="",
        **kwargs
    ):
        """
        This is the main class which connects each component of the AGN together
        and allows for a consistent calculation of each component.
        """

        self.smbh_mass_exp = smbh_mass_exp
        self.smbh_mass = 10**self.smbh_mass_exp * const.M_sun.to(u.kg)
        self.gravitational_radius = calculate_gravitational_radius(
            10**self.smbh_mass_exp
        )
        self.redshift_source = redshift_source
        self.inclination_angle = inclination_angle
        self.OmM = OmM
        self.H0 = H0
        self.name = name
        self.kwargs = kwargs

        self.disk_is_updatable = True
        self.has_blr = False

        self.generate_kwarg_dictionaries_for_individual_components()

        self.components = {}

    def add_default_accretion_disk(self, **kwargs):
        """use create_maps to generate an accretion disk. then store it in AGN"""

        all_kwargs = [
            "mass_exp",
            "redshift",
            "number_grav_radii",
            "inc_ang",
            "resolution",
            "spin",
            "eddington_ratio",
            "temp_beta",
            "corona_height",
            "albedo",
            "eta",
            "generic_beta",
            "disk_acc",
            "height_array",
            "albedo_array",
            "OmM",
            "H0",
            "efficiency",
            "visc_temp_prof",
        ]

        for kwarg in all_kwargs:
            # update if overridden
            if kwarg in kwargs:
                self.default_accretion_disk_kwargs[kwarg] = kwargs[kwarg]

        agn_disk_dictionary = create_maps(**self.default_accretion_disk_kwargs)

        # store dictionary in generic_acc_disk_kwargs to facilitate later updates
        for kwarg in agn_disk_dictionary.keys:
            self.generic_accretion_disk_kwargs[kwarg] = agn_disk_dictionary[kwarg]

        agn_disk = AccretionDisk(**agn_disk_dictionary)

        self.disk_is_updatable = True

        self.components["accretion_disk"] = agn_disk

        return True

    def add_generic_accretion_disk(self, **kwargs):
        """give me an AGN parameter dictionary to store a disk in the AGN"""

        all_kwargs = [
            "smbh_mass_exp",
            "redshift_source",
            "inclination_angle",
            "corona_height",
            "temp_array",
            "phi_array",
            "g_array",
            "radii_array",
            "height_array",
            "albedo_array",
            "spin",
            "OmM",
            "H0",
            "r_out_in_gravitational_radii",
        ]

        for kwarg in all_kwargs:
            if kwarg in kwargs:
                self.generic_accretion_disk_kwargs[kwarg] = kwargs[kwarg]

        self.disk_is_updatable = False

        agn_disk = AccretionDisk(**self.generic_accretion_disk_kwargs)

        self.components["accretion_disk"] = agn_disk

        return True

    def add_blr(self, blr_index=0, **kwargs):
        """give me a dictionary containing information about a BLR. Append to a list of BLRs"""

        all_kwargs = [
            "smbh_mass_exp",
            "max_height",
            "rest_frame_wavelength_in_nm",
            "redshift_source",
            "radial_step",
            "height_step",
            "max_radius",
            "OmM",
            "H0",
        ]

        for kwarg in all_kwargs:
            if kwarg in kwargs:
                self.blr_kwargs[kwarg] = kwargs[kwarg]

        agn_blr = BroadLineregion(**self.blr_kwargs)

        self.components["blr_" + str(blr_index)] = agn_blr

        self.has_blr = True

        return True

    def add_streamline_bounded_region_to_blr(self, blr_index, **kwargs):
        """for BLR with blr_index, add a region of particles using
        add_streamline_bounded_region"""

        required_kwargs = [
            "InnerStreamline",
            "OuterStreamline",
        ]

        for kwarg in required_kwargs:
            if kwarg not in kwargs:
                print("two streamlines are required, blr was not updated")
                return False

        self.components["blr_" + str(blr_index)].add_streamline_bounded_region(**kwargs)

        return True

    def add_torus(self, **kwargs):
        """add an obscuring torus to the AGN model"""

        all_kwargs = [
            "smbh_mass_exp",
            "max_height",
            "redshift_source",
            "radial_step",
            "height_step",
            "power_law_density_dependence",
            "OmM",
            "H0",
        ]

        for kwarg in all_kwargs:
            if kwarg in kwargs:
                self.torus_kwargs[kwarg] = kwargs[kwarg]

        agn_torus = Torus(**self.torus_kwargs)

        self.components["torus"] = agn_torus

    def add_diffuse_continuum(self, **kwargs):
        """add a diffuse continuum to the AGN"""

        all_kwargs = [
            "smbh_mass_exp",
            "inclination_angle",
            "redshift_source",
            "radii_array",
            "phi_array",
            "cloud_density_radial_dependence",
            "cloud_density_array",
            "OmM",
            "H0",
            "r_in_in_gravitational_radii",
            "r_out_in_gravitational_radii",
        ]

        for kwarg in all_kwargs:
            if kwarg in kwargs:
                self.diffuse_continuum_kwargs[kwarg] = kwargs[kwarg]

        agn_dc = DiffuseContinuum(**self.diffuse_continuum_kwargs)

        self.components["diffuse_continuum"] = agn_dc

    def add_intrinsic_signal_parameters(self, **kwargs):
        """define the parameters which will be used to generate the intrinsic signal"""

        if "power_spectrum" in kwargs.keys:
            self.power_spectrum = kwargs["psd"]
        if "frequencies" in kwargs.keys:
            self.frequencies = kwargs["frequencies"]

        # random seed is optional
        if "random_seed" in kwargs.keys:
            self.random_seed = kwargs["random_seed"]
        else:
            self.random_seed = None

    def visualize_static_accretion_disk(self, observed_wavelength_in_nm):
        """create FluxProjection of accretion disk"""

        accretion_disk_flux_projection = self.components[
            "accretion_disk"
        ].calculate_surface_intensity_map(observed_wavelength_in_nm)

        return accretion_disk_flux_projection

    def visualize_static_blr(self, blr_index, **kwargs):
        """create FluxProjection of BLR with index blr_index"""

        blr_flux_projection = self.components[
            "blr_" + str(blr_index)
        ].project_blr_intensity_over_velocity_range(self.inclination_angle, **kwargs)

        return blr_flux_projection

    def visualize_torus_obscuration(self, obscuration_strength, **kwargs):
        """create FluxProjection in magnitudes to use as extinction"""

        torus_extinction_projection = self.components[
            "torus"
        ].project_density_to_source_plane(self.inclination_angle)

        # need to still add support for this calculation in the torus code
        torus_extinction.flux_array = np.exp(
            -torus_extinction.flux_array * obscuration_strength
        )

        return torus_extinction_projection

    def visualize_static_diffuse_continuum(self, observed_wavelength_in_nm, **kwargs):
        """create FluxProjection of diffuse continuum"""

        diffuse_continuum_projection = self.components[
            "diffuse_continuum"
        ].get_diffuse_continuum_emission(observed_wavelength_in_nm, **kwargs)

        return diffuse_continuum_projection

    def visualize_intrinsic_signal(self, max_time_in_days, **kwargs):
        """generate a driving signal based on stored variability parameters"""

        signal = generate_signal_from_psd(
            max_time_in_days, self.power_spectrum, self.frequencies, self.random_seed
        )

        return signal

    def update_smbh_mass_exponent(self, new_smbh_mass_exponent):
        """update the black hole mass in all components"""

        self.smbh_mass_exp = new_smbh_mass_exponent
        self.smbh_mass = 10**self.smbh_mass_exp * const.M_sun.to(u.kg)
        self.gravitational_radius = calculate_gravitational_radius(
            10**self.smbh_mass_exp
        )

        if self.disk_is_updatable == False:
            print(
                "accretion disk is not updatable, temperature profile will not be consistent"
            )

        self.default_accretion_disk_kwargs["mass_exp"] = self.smbh_mass_exp
        self.generic_accretion_disk_kwargs["mass_exp"] = self.smbh_mass_exp
        self.blr_kwargs["mass_exp"] = self.smbh_mass_exp
        self.diffuse_continuum_kwargs["mass_exp"] = self.smbh_mass_exp
        self.torus_kwargs["mass_exp"] = self.smbh_mass_exp

        if "accretion_disk" in self.components.keys and self.disk_is_updatable == True:
            self.add_default_accretion_disk()
        elif (
            "accretion_disk" in self.components.keys and self.disk_is_updatable == False
        ):
            self.add_generic_accretion_disk()
        if "diffuse_continuum" in self.components.keys:
            self.add_diffuse_continuum()
        if "torus" in self.components.keys:
            self.add_torus()

        if self.has_blr == True:
            blr_keys = []
            for key in self.components.keys:
                if key[:4] == "blr_":
                    blr_keys.append(key[3:])
            for key in blr_keys:
                self.add_blr(blr_index=key)
        return True

    def update_inclination(self, new_inclination):
        """update inclination in all components"""

        if self.disk_is_updatable == False:
            print("accretion disk is not updatable, new maps must be calculated")
            return False

        self.inclination_angle = new_inclination

        self.default_accretion_disk_kwargs["inclination_angle"] = self.inclination_angle
        self.generic_accretion_disk_kwargs["inclination_angle"] = self.inclination_angle
        self.blr_kwargs["inclination_angle"] = self.inclination_angle
        self.diffuse_continuum_kwargs["inclination_angle"] = self.inclination_angle
        self.torus_kwargs["inclination_angle"] = self.inclination_angle

        if "accretion_disk" in self.components.keys and self.disk_is_updatable == True:
            self.add_default_accretion_disk()
        return True

    def update_redshift(self, new_redshift):
        """update redshift in all components"""

        self.redshift_source = new_redshift

        self.default_accretion_disk_kwargs["redshift_source"] = self.redshift_source
        self.generic_accretion_disk_kwargs["redshift_source"] = self.redshift_source
        self.blr_kwargs["redshift_source"] = self.redshift_source
        self.diffuse_continuum_kwargs["redshift_source"] = self.redshift_source
        self.torus_kwargs["redshift_source"] = self.redshift_source

        return True

    def update_h0(self, new_H0):
        """update H0 in all components"""

        self.H0 = new_H0

        self.default_accretion_disk_kwargs["H0"] = self.H0
        self.generic_accretion_disk_kwargs["H0"] = self.H0
        self.blr_kwargs["H0"] = self.H0
        self.diffuse_continuum_kwargs["H0"] = self.H0
        self.torus_kwargs["H0"] = self.H0

        return True

    def update_omega_m(self, new_omega_m):
        """update OmM in all components"""

        self.OmM = new_omega_m

        self.default_accretion_disk_kwargs["OmM"] = self.OmM
        self.generic_accretion_disk_kwargs["OmM"] = self.OmM
        self.blr_kwargs["OmM"] = self.OmM
        self.diffuse_continuum_kwargs["OmM"] = self.OmM
        self.torus_kwargs["OmM"] = self.OmM

        return True

    def calculate_accretion_disk_transfer_function(
        self, observed_wavelength_in_nm, **kwargs
    ):
        """calculate the transfer function of the accretion disk in units Rg"""

        if "accretion_disk" not in self.components.keys:
            return False

        accretion_disk_transfer_function = self.components[
            "accretion_disk"
        ].construct_accretion_disk_transfer_function(
            observed_wavelength_in_nm, **kwargs
        )

        return accretion_disk_transfer_function

    def calculate_blr_transfer_function(self, observed_wavelengths_in_nm, **kwargs):
        """calculate the transfer function of the BLR in units Rg"""

        if self.has_blr is False:
            return False

        for key in self.components.keys:
            if key[:4] == "blr_":
                blr_keys.append(key)

        blr_transfer_function_list = []
        for key in blr_keys:
            current_transfer_function = calculate_blr_emission_line_transfer_function(
                inclination=self.inclination_angle,
                observed_wavelength_range_in_nm=observed_wavelength_range_in_nm,
                **kwargs
            )
            blr_transfer_function_list.append(current_transfer_function)

        return blr_transfer_function_list

    def calculate_diffuse_continuum_mean_time_lag(self, **kwargs):
        """calculate the diffuse continuum's mean time lag contribution to
        the time lags in the continuum"""

        # code that runs agn.components["diffuse_continuum"].calculate_mean_time_lag_increase()

        return dc_mean_time_lag

    def intrinsic_signal_propogation_pipeline(self, **kwargs):
        """run the pipeline to generate the full AGN intrinsic signal"""

        # code that runs sequential operations

        return full_agn_light_curve

    def visualize_agn_pipeline(self, **kwargs):
        """run the pipeline to generate the full flux distribution of
        each of the AGN components"""

        # different code that runs sequential operations

        return full_agn_flux_projection

    def generate_kwarg_dictionaries_for_individual_components(self):
        """generates keyword dictionaries for each component based on the
        AGN dictionary given"""

        default_accretion_disk_kwargs = [
            "number_grav_radii",
            "inclination_angle",
            "resolution",
            "spin",
            "eddington_ratio",
            "temp_beta",
            "corona_height",
            "albedo",
            "eta",
            "generic_beta",
            "disk_acc",
            "height_array",
            "albedo_array",
            "efficiency",
            "visc_temp_prof",
        ]

        generic_accretion_disk_kwargs = [
            "temp_array",
            "phi_array",
            "g_array",
            "radii_array",
            "height_array",
            "albedo_array",
            "r_out_in_gravitational_radii",
        ]
        blr_kwargs = [
            "max_height",
            "rest_frame_wavelength_in_nm",
            "radial_step",
            "height_step",
            "max_radius",
        ]
        diffuse_continuum_kwargs = [
            "radii_array",
            "phi_array",
            "cloud_density_radial_dependence",
            "cloud_density_array",
            "r_in_in_gravitational_radii",
            "r_out_in_gravitational_radii",
        ]
        intrinsic_signal_kwargs = [
            "power_spectrum",
            "frequencies",
        ]
        torus_kwargs = [
            "max_height",
            "radial_step",
            "height_step",
            "power_law_density_dependence",
        ]

        self.default_accretion_disk_kwargs = {}
        self.generic_accretion_disk_kwargs = {}
        self.blr_kwargs = {}
        self.diffuse_continuum_kwargs = {}
        self.intrinsic_signal_kwargs = {}
        self.torus_kwargs = {}

        for kwarg in self.kwargs.keys:
            if kwarg in default_accretion_disk_kwargs:
                self.default_accretion_disk_kwargs[kwarg] = self.kwargs[kwarg]
            if kwarg in generic_accretion_disk_kwargs:
                self.generic_accretion_disk_kwargs[kwarg] = self.kwargs[kwarg]
            if kwarg in blr_kwargs:
                self.blr_kwargs[kwarg] = self.kwargs[kwarg]
            if kwarg in diffuse_continuum_kwargs:
                self.diffuse_continuum_kwargs[kwarg] = self.kwargs[kwarg]
            if kwarg in intrinsic_signal_kwargs:
                self.intrinsic_signal_kwargs[kwarg] = self.kwargs[kwarg]
            if kwarg in torus_kwargs:
                self.torus_kwargs[kwarg] = self.kwargs[kwarg]

        if self.smbh_mass_exp is not None:
            self.default_accretion_disk_kwargs["mass_exp"] = self.smbh_mass_exp
            self.generic_accretion_disk_kwargs["mass_exp"] = self.smbh_mass_exp
            self.blr_kwargs["mass_exp"] = self.smbh_mass_exp
            self.diffuse_continuum_kwargs["mass_exp"] = self.smbh_mass_exp
            self.torus_kwargs["mass_exp"] = self.smbh_mass_exp

        if self.redshift_source is not None:
            self.default_accretion_disk_kwargs["redshift_source"] = self.redshift_source
            self.generic_accretion_disk_kwargs["redshift_source"] = self.redshift_source
            self.blr_kwargs["redshift_source"] = self.redshift_source
            self.diffuse_continuum_kwargs["redshift_source"] = self.redshift_source
            self.torus_kwargs["redshift_source"] = self.redshift_source

        if self.inclination_angle is not None:
            self.default_accretion_disk_kwargs["inclination_angle"] = (
                self.inclination_angle
            )
            self.generic_accretion_disk_kwargs["inclination_angle"] = (
                self.inclination_angle
            )
            self.blr_kwargs["inclination_angle"] = self.inclination_angle
            self.diffuse_continuum_kwargs["inclination_angle"] = self.inclination_angle
            self.torus_kwargs["inclination_angle"] = self.inclination_angle

        if self.OmM is not None:
            self.default_accretion_disk_kwargs["OmM"] = self.OmM
            self.generic_accretion_disk_kwargs["OmM"] = self.OmM
            self.blr_kwargs["OmM"] = self.OmM
            self.diffuse_continuum_kwargs["OmM"] = self.OmM
            self.torus_kwargs["OmM"] = self.OmM

        if self.H0 is not None:
            self.default_accretion_disk_kwargs["H0"] = self.H0
            self.generic_accretion_disk_kwargs["H0"] = self.H0
            self.blr_kwargs["H0"] = self.H0
            self.diffuse_continuum_kwargs["H0"] = self.H0
            self.torus_kwargs["H0"] = self.H0
