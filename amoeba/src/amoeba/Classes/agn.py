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
    convolve_signal_with_transfer_function,
    convert_cartesian_to_polar,
)
from amoeba.Util.pipeline_util import (
    intrinsic_signal_propagation_pipeline_for_agn,
    visualization_pipeline,
)

from speclite.filters import load_filter, load_filters
import speclite


class Agn:

    def __init__(self, agn_name="", **kwargs):
        """This is the main class which connects each component of the AGN together and
        allows for a consistent calculation of each component.

        :param agn_name: name space for Agn object
        :kwarg smbh_mass_exp: solution to log10(M_smbh / M_sun)
        :kwarg redshift_source: redshift of the Agn object
        :kwarg inclination_angle: inclination of the Agn with respect to our point of view
        :kwarg OmM: Energy budget of matter in the universe
        :kwarg H0: Hubble constant in units km/s/Mpc

        ----- accretion disk kwargs -----

        "number_grav_radii", "inclination_angle", "resolution", "spin", "eddington_ratio",
        "temp_beta", "corona_height", "albedo", "eta", "generic_beta", "disk_acc",
        "height_array", "albedo", "efficiency", "visc_temp_prof"

        ----- broad line region kwargs -----

        "smbh_mass_exp", "mass_height", "rest_frame_wavelength_in_nm",
        "redshift_source", "radial_step", "height_step", "max_radius"
        "line_strength"

        ----- torus kwargs -----

        "smbh_mass_exp", "max_height", "redshift_source", "radial_step",
        "height_step", "power_law_density_dependence"
        """

        self.kwargs = kwargs
        if "smbh_mass_exp" in self.kwargs.keys():
            self.smbh_mass_exp = self.kwargs["smbh_mass_exp"]
            self.smbh_mass = 10**self.smbh_mass_exp * const.M_sun.to(u.kg)
            self.gravitational_radius = calculate_gravitational_radius(
                10**self.smbh_mass_exp
            )
        if "redshift_source" in self.kwargs.keys():
            self.redshift_source = self.kwargs["redshift_source"]
        if "inclination_angle" in self.kwargs.keys():
            self.inclination_angle = self.kwargs["inclination_angle"]
        if "OmM" in self.kwargs.keys():
            self.OmM = self.kwargs["OmM"]
        else:
            self.OmM = 0.3
        if "H0" in self.kwargs.keys():
            self.H0 = self.kwargs["H0"]
        else:
            self.H0 = 70
        self.name = agn_name

        self.disk_is_updatable = True
        self.intrinsic_light_curve = None
        self.intrinsic_light_curve_time_axis = None
        self.blr_indicies = []

        self.generate_kwarg_dictionaries_for_individual_components()

        self.components = {}

    def add_default_accretion_disk(self, **kwargs):
        """Store a basic accretion disk object which is generatable with the create_maps
        function in Util. Expected input is a dictionary of accretion disk parameters.

        :return: True if successful
        """

        all_kwargs = [
            "smbh_mass_exp",
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
            "albedo",
            "OmM",
            "H0",
            "efficiency",
            "visc_temp_prof",
        ]

        for kwarg in all_kwargs:
            if kwarg in kwargs:
                self.default_accretion_disk_kwargs[kwarg] = kwargs[kwarg]

        agn_disk_dictionary = create_maps(**self.default_accretion_disk_kwargs)

        for kwarg in agn_disk_dictionary.keys():
            self.generic_accretion_disk_kwargs[kwarg] = agn_disk_dictionary[kwarg]

        agn_disk = AccretionDisk(**agn_disk_dictionary)

        self.disk_is_updatable = True

        self.components["accretion_disk"] = agn_disk

        return True

    def add_generic_accretion_disk(self, **kwargs):
        """Create an accretion disk from an AGN parameter dictionary. This may be used
        to generate more specialized accretion disk objects. Expected input is a
        dictionary of accretion disk parameters.

        :return: True if successful
        """

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
            "albedo",
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
        """Add an initialization of a BroadLineRegion object which is defined with index
        blr_index. Expects a dictionary of BLR parameters in kwargs input.

        :param blr_index: int or float to specify which BLR we are using
        :param line_strength: int or float representing how strong the emission line is
            with respect to the accretion disk.
        :return: True if successful
        """

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
            "line_strength",
        ]

        for kwarg in all_kwargs:
            if kwarg in kwargs:
                self.blr_kwargs[kwarg] = kwargs[kwarg]

        agn_blr = BroadLineRegion(**self.blr_kwargs)

        self.components["blr_" + str(blr_index)] = agn_blr
        if blr_index not in self.blr_indicies:
            self.blr_indicies.append(blr_index)

        return True

    def add_streamline_bounded_region_to_blr(self, blr_index=0, **kwargs):
        """Add a region of particles using the add_streamline_bounded_region method of
        the BroadLineRegion object with index blr_index. Expects two streamlines in
        a dictionary for the kwargs argument.

        :param blr_index: int or float representing which BLR to add to
        :return: True if successful
        """

        if blr_index not in self.blr_indicies:
            print("blr_index not found, please initialize a BLR with this index")
            return False

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

    def get_blr_density_axes(self, blr_index=None):
        """Get the meshgrid representations of the R-Z coordinates of the
        BroadLineRegion object(s).

        :param blr_index: None or specific index / list of indicies to
            return the axis(axes) of. If None, a list of axes will be
            returned. If specified, only the requrested axes will be returned.
        :return: list of lists containing the R, Z meshgrid coordinates of
            each BLR object.
        """
        if blr_index is None:
            blr_index = self.blr_indicies
        elif isinstance(blr_index, (int, float, str)):
            blr_index = [blr_index]

        output_axes = []
        for index in blr_index:
            current_R, current_Z = self.components[
                "blr_" + str(index)
            ].get_density_axis()
            output_axes.append([current_R, current_Z])
        return output_axes

    def set_blr_efficiency_array(self, efficiency_array=None, blr_index=None):
        """Set the emission efficiency array of the BroadLineRegion object
        with index blr_index. If no arguments are passed, this will check
        which BroadLineRegion components do not have efficiency arrays
        associated with them.

        :param efficiency_array: Array representing the weighted emission
            efficiency at each position in R, Z coordinates.
        :param blr_index: index representing which BLR to update
        :return: True if successful
        """
        if efficiency_array is None and blr_index is None:
            for index in self.blr_indicies:
                if (
                    self.components["blr_" + str(index)].emission_efficiency_array
                    is None
                ):
                    print("indexes without efficiency arrays:", index)
            return False
        if blr_index is None:
            print("Please give the index to associate this efficiency array with.")
            print("Note that the default blr index is '0'")
            return False
        if efficiency_array is None:
            if (
                self.components["blr_" + str(blr_index)].emission_efficiency_array
                is None
            ):
                print("this index does not have an efficiency array associated with it")
            else:
                print("this index has an efficiency array associated with it")
            return False

        self.components["blr_" + str(blr_index)].set_emission_efficiency_array(
            emission_efficiency_array=efficiency_array
        )
        return True

    def add_torus(self, **kwargs):
        """Initialize an obscuring torus in the AGN model. Note this is still an
        experimental feature. After initialization, the torus must be defined by the
        add_streamline_bounded_region_to_torus() method. Expects a dictionary of
        torus parameters in the kwargs argument.

        :return: True if successful
        """

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

        return True

    def add_streamline_bounded_region_to_torus(self, **kwargs):
        """Define the torus with a Streamline object. Expects a streamline in a
        dictionary for the kwargs argument.

        :return: True if successful
        """

        required_kwargs = ["Streamline"]

        for kwarg in required_kwargs:
            if kwarg not in kwargs:
                print("streamline required with keyword Streamline, torus not updated")
                return False

        self.components["torus"].add_streamline_bounded_region(**kwargs)

        return True

    def add_diffuse_continuum(self, **kwargs):
        """Add a diffuse continuum to the AGN. Expects a dictionary of diffuse
        continuum parameters in the kwargs argument.

        :return: True if successful
        """

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
            "responsivity_constant",
            "rest_frame_wavelengths",
            "emissivity_etas",
        ]

        for kwarg in all_kwargs:
            if kwarg in kwargs:
                self.diffuse_continuum_kwargs[kwarg] = kwargs[kwarg]

        if "r_out_in_gravitational_radii" in self.diffuse_continuum_kwargs.keys():
            xax = np.linspace(
                -self.diffuse_continuum_kwargs["r_out_in_gravitational_radii"],
                self.diffuse_continuum_kwargs["r_out_in_gravitational_radii"],
                100,
            )
            X, Y = np.meshgrid(xax, xax)
            R, Phi = convert_cartesian_to_polar(X, Y)
        else:
            assert "radii_array" in self.diffuse_continuum_kwargs.keys()
            assert "phi_array" in self.diffuse_continuum_kwargs.keys()

        if "radii_array" not in self.diffuse_continuum_kwargs.keys():
            self.diffuse_continuum_kwargs["radii_array"] = R

        if "phi_array" not in self.diffuse_continuum_kwargs.keys():
            self.diffuse_continuum_kwargs["phi_array"] = Phi

        agn_dc = DiffuseContinuum(**self.diffuse_continuum_kwargs)

        self.components["diffuse_continuum"] = agn_dc

        return True

    def add_intrinsic_signal_parameters(self, **kwargs):
        """Define the parameters which will be used to generate the intrinsic signal.
        Expects a dictionary of intrinsic signal parameters to be passed into the
        kwargs argument.

        These may be:
        :param power_spectrum: list or array representing the power spectral density (PSD)
            of the intrinsic signal
        :param frequencies: list or array representing the frequencies associated with
            the PSD. Must be identical in size to power_spectrum.
        :param random_seed: optional int to set a random seed.
        :return: True if successful
        """

        if "power_spectrum" in kwargs.keys():
            self.power_spectrum = kwargs["power_spectrum"]
        if "frequencies" in kwargs.keys():
            self.frequencies = kwargs["frequencies"]

        if "random_seed" in kwargs.keys():
            self.random_seed = kwargs["random_seed"]
        else:
            self.random_seed = None

        return True

    def visualize_static_accretion_disk(self, observed_wavelength_in_nm, **kwargs):
        """Create FluxProjection of the accretion disk component.

        :param observed_wavelength_in_nm: observer frame wavelength in nm
        :return: FluxProjection object of the accretion disk component
        """

        accretion_disk_flux_projection = self.components[
            "accretion_disk"
        ].calculate_surface_intensity_map(observed_wavelength_in_nm)

        return accretion_disk_flux_projection

    def visualize_static_blr(self, blr_index=0, **kwargs):
        """Create FluxProjection of BLR with index blr_index.

        :param blr_index: index of the BroadLineRegion to project.
        :param velocity_range: array or list representing the velocity range to project
            in units v/c. If not given, will project all velocities. Note positive is
            towards the observer.
        :param speclite_filters: (list of) speclite filters to project the BLR into.
            Cannot be used with velocity_range.
        :return: FluxProjection object of the BLR in the AGN with index blr_index.
        """

        if "velocity_range" not in kwargs and "speclite_filters" not in kwargs:
            kwargs["velocity_range"] = [-1, 1]

        blr_flux_projection = self.components[
            "blr_" + str(blr_index)
        ].project_blr_intensity_over_velocity_range(self.inclination_angle, **kwargs)

        return blr_flux_projection

    def visualize_torus_obscuration(self, observed_wavelength_in_nm, **kwargs):
        """Create FluxProjection in magnitudes to use as extinction. Note this is in the
        experimental phase.

        :param observed_wavelength_in_nm: observer frame wavelength in nm
        :return: FluxProjection object representing the column density projected to the
            source plane. Can be used as a proxy for the attenuation by the dusty torus.
        """

        torus_extinction_projection = self.components[
            "torus"
        ].project_extinction_to_source_plane(
            self.inclination_angle, observed_wavelength_in_nm
        )

        return torus_extinction_projection

    def visualize_static_diffuse_continuum(self, observed_wavelength_in_nm, **kwargs):
        """Create FluxProjection of diffuse continuum.

        :param observed_wavelength_in_nm: observer frame wavelength in nm
        :return: FluxProjection object representing the emission of the diffuse
            continuum. Note that the diffuse continuum object is not assumed to be
            inclination dependent, so this will always project into an annulus.
        """

        diffuse_continuum_projection = self.components[
            "diffuse_continuum"
        ].get_diffuse_continuum_emission(observed_wavelength_in_nm, **kwargs)

        return diffuse_continuum_projection

    def generate_intrinsic_signal(self, max_time_in_days, **kwargs):
        """Generate a driving signal based on stored variability parameters. This stores
        the signal in the AGN, and also returns the driving light curve. If variability
        parameters are not stored, they must be passed into kwargs.

        :param max_time_in_days: duration of the light curve to produce in days.
        :param power_spectrum: list or array representing the power spectral density to
            use. Note that this may be predefined in the AGN with the
            self.add_intrinsic_signal_parmeters() method.
        :param frequencies: list or array representing the frequencies associated with
            the PSD to use. Note that this may be predefined in the AGN with the
            self.add_intrinsic_signal_parmeters() method.
        :param random_seed: optional int to use as a random seed.
        :return: time axis and light curve associated with the generated driving signal
        """

        for kwarg in kwargs.keys():
            if kwarg == "power_spectrum":
                self.power_spectrum = kwargs[kwarg]
            if kwarg == "frequencies":
                self.frequencies = kwargs[kwarg]
            if kwarg == "random_seed":
                self.random_seed = kwargs[kwarg]

        if self.power_spectrum is None or self.frequencies is None:
            print(
                "please provide psd and frequencies with add_intrinsic_signal_parameters or kwargs"
            )
            return False

        time_axis, light_curve = generate_signal_from_psd(
            max_time_in_days, self.power_spectrum, self.frequencies, self.random_seed
        )

        self.intrinsic_light_curve = light_curve
        self.intrinsic_light_curve_time_axis = time_axis

        return time_axis, light_curve

    def update_smbh_mass_exponent(self, new_smbh_mass_exponent):
        """Update the black hole mass in all components. Note that the accretion disk
        may not be updatable if a specialized disk was created, and a new AGN object
        must be constructed.

        :param new_smbh_mass_exponent: Updated solution to log10(m_smbh / m_sun)
        :return: True if successful
        """

        self.smbh_mass_exp = new_smbh_mass_exponent
        self.smbh_mass = 10**self.smbh_mass_exp * const.M_sun.to(u.kg)
        self.gravitational_radius = calculate_gravitational_radius(
            10**self.smbh_mass_exp
        )

        if self.disk_is_updatable == False:
            print(
                "accretion disk is not updatable, temperature profile will not be consistent"
            )

        self.default_accretion_disk_kwargs["smbh_mass_exp"] = self.smbh_mass_exp
        self.generic_accretion_disk_kwargs["smbh_mass_exp"] = self.smbh_mass_exp
        self.blr_kwargs["smbh_mass_exp"] = self.smbh_mass_exp
        self.diffuse_continuum_kwargs["smbh_mass_exp"] = self.smbh_mass_exp
        self.torus_kwargs["smbh_mass_exp"] = self.smbh_mass_exp

        if (
            "accretion_disk" in self.components.keys()
            and self.disk_is_updatable == True
        ):
            self.add_default_accretion_disk()

        elif (
            "accretion_disk" in self.components.keys()
            and self.disk_is_updatable == False
        ):
            self.add_generic_accretion_disk()

        if "diffuse_continuum" in self.components.keys():
            self.add_diffuse_continuum()

        if "torus" in self.components.keys():
            self.add_torus()

        if len(self.blr_indicies) > 0:
            for key in self.blr_indicies:
                self.add_blr(blr_index=key)
        return True

    def update_inclination(self, new_inclination):
        """Update inclination in all components. Note that the accretion disk may not be
        updatable if a specialized disk was created, and a new AGN object must be
        constructed.

        :param new_inclination: Updated inclination in degrees
        :return: True if successful
        """

        if self.disk_is_updatable == False:
            print("accretion disk is not updatable, new maps must be calculated")
            return False

        self.inclination_angle = new_inclination

        self.default_accretion_disk_kwargs["inclination_angle"] = self.inclination_angle
        self.generic_accretion_disk_kwargs["inclination_angle"] = self.inclination_angle
        self.diffuse_continuum_kwargs["inclination_angle"] = self.inclination_angle

        if (
            "accretion_disk" in self.components.keys()
            and self.disk_is_updatable == True
        ):
            self.add_default_accretion_disk()

        if "diffuse_continuum" in self.components.keys():
            self.add_diffuse_continuum()

        return True

    def update_redshift(self, new_redshift):
        """Update redshift in all components.

        :param new_redshift: new redshift of the AGN
        :return: True if successful
        """

        self.redshift_source = new_redshift

        self.default_accretion_disk_kwargs["redshift_source"] = self.redshift_source
        self.generic_accretion_disk_kwargs["redshift_source"] = self.redshift_source
        self.blr_kwargs["redshift_source"] = self.redshift_source
        self.diffuse_continuum_kwargs["redshift_source"] = self.redshift_source
        self.torus_kwargs["redshift_source"] = self.redshift_source

        if (
            "accretion_disk" in self.components.keys()
            and self.disk_is_updatable == True
        ):
            self.add_default_accretion_disk()
        for component in self.components:
            self.components[component].redshift_source = self.redshift_source

        return True

    def update_h0(self, new_H0):
        """Update H0 in all components. Gives you the power to change cosmology.

        :param new_H0: updated Hubble constant in units km/s/Mpc
        :return: True if successful
        """

        self.H0 = new_H0

        self.default_accretion_disk_kwargs["H0"] = self.H0
        self.generic_accretion_disk_kwargs["H0"] = self.H0
        self.blr_kwargs["H0"] = self.H0
        self.diffuse_continuum_kwargs["H0"] = self.H0
        self.torus_kwargs["H0"] = self.H0

        return True

    def update_omega_m(self, new_omega_m):
        """Update OmM in all components. Gives you the power to change cosmology.

        :param new_omega_m: new mass component of the energy budget of the universe
        :return: True if successful
        """

        self.OmM = new_omega_m

        self.default_accretion_disk_kwargs["OmM"] = self.OmM
        self.generic_accretion_disk_kwargs["OmM"] = self.OmM
        self.blr_kwargs["OmM"] = self.OmM
        self.diffuse_continuum_kwargs["OmM"] = self.OmM
        self.torus_kwargs["OmM"] = self.OmM

        return True

    def update_line_strength(self, blr_index, new_line_strength):
        """Update line strength of a BLR emission line.

        :param blr_index: index associated with a particular BroadLineRegion object in
            the AGN model
        :param new_line_strength: int or float representing the new (relative) strength
            of the emission line
        :return: True if successful
        """

        self.components["blr_" + str(blr_index)].update_line_strength(new_line_strength)

        return True

    def calculate_accretion_disk_transfer_function(
        self, observed_wavelength_in_nm, **kwargs
    ):
        """Calculate the transfer function of the accretion disk in units R_g / c.

        :param observed_wavelength_in_nm: observer frame wavelength in nm.
        :return: a list represneting the accretion disk's transfer function in time lag
            units R_g / c
        """

        if "accretion_disk" not in self.components.keys():
            print("please add an accretion disk component to the model")
            return False

        accretion_disk_transfer_function = self.components[
            "accretion_disk"
        ].construct_accretion_disk_transfer_function(
            observed_wavelength_in_nm, **kwargs
        )

        return accretion_disk_transfer_function

    def calculate_blr_transfer_function(self, observed_wavelengths_in_nm, **kwargs):
        """Calculate the transfer function of all BroadLineRegion components within the
        AGN in units R_g / c.

        :param observed_wavelengths_in_nm: observer frame wavelength in nm
        :return: list of BroadLineRegion transfer functions, which are lists of
            responses in units of R_g / c.
        """

        if len(self.blr_indicies) == 0:
            print("please add a broad line region component to the model")
            return False

        blr_keys = []
        for key in self.components.keys():
            if key[:4] == "blr_":
                blr_keys.append(key)

        blr_transfer_function_list = []
        for key in blr_keys:
            fractional_weight, current_transfer_function = self.components[
                key
            ].calculate_blr_emission_line_transfer_function(
                inclination_angle=self.inclination_angle,
                observed_wavelength_range_in_nm=observed_wavelengths_in_nm,
                **kwargs,
            )

            blr_transfer_function_list.append(
                [fractional_weight, current_transfer_function]
            )

        return blr_transfer_function_list

    def calculate_diffuse_continuum_mean_time_lag(
        self, observed_wavelength_in_nm, **kwargs
    ):
        """Calculate the diffuse continuum's mean time lag contribution to the time lags
        in the continuum.

        :param observed_wavelength_in_nm: observer frame wavelength in nm
        :return: increase in mean time lag due to diffue continuum in units R_g / c
        """

        if "diffuse_continuum" not in self.components.keys():
            print("please add a diffuse continuum component to the model")
            return False

        return self.components["diffuse_continuum"].get_diffuse_continuum_mean_lag(
            observed_wavelength_in_nm
        )

    def intrinsic_signal_propagation_pipeline(
        self,
        intrinsic_light_curve=None,
        time_axis=None,
        observer_frame_wavelengths_in_nm=None,
        speclite_filter=None,
        blr_weightings=None,
        return_components=False,
        **kwargs,
    ):
        """Runs the intrinsic signal propagation pipeline by generating a light curve
        based on intrinsic signal parameters if stored, propagating this through the
        accretion disk, then increasing these transfer functions' tau axis by the
        increase due to the diffuse continuum, using these light curves as the driving
        signal for all broad line region components, and joining the light curves
        together.

        :param intrinsic_light_curve: None or list/array representing the driving light
            curve to propagate through the system.
        :param time_axis: None or list/array representing the time stamps of the driving
            light curve if provided. If None, the intrinsic_light_curve will be assumed
            to have daily cadence.
        :param observer_frame_wavelengths_in_nm: list or int of observer frame
            wavelength(s) in nm to propagate the driving signal to. Cannot be used with
            speclite_filter.
        :param speclite_filter: (list of) speclite filter(s) to propagate the driving
            signal to. Cannot be used with observer_frame_wavelengths_in_nm.
        :param blr_weightings: list of BroadLineRegion weightings to use with the local
            optimally emitting cloud model. Each weighting must be a 2 dimensional array
            of shape (R, Z).
        :param return_components: boolean toggle to return individual light curves in
            addition to the fully joined light curves.
        :return: list of light curves in each wavelength/filter
        """

        output_signals = intrinsic_signal_propagation_pipeline_for_agn(
            self,
            intrinsic_light_curve=intrinsic_light_curve,
            time_axis=time_axis,
            observer_frame_wavelengths_in_nm=observer_frame_wavelengths_in_nm,
            speclite_filter=speclite_filter,
            blr_weightings=blr_weightings,
            return_components=return_components,
            **kwargs,
        )

        return output_signals

    def visualize_agn_pipeline(
        self,
        inclination_angle=None,
        observer_frame_wavelengths_in_nm=None,
        speclite_filter=None,
        blr_weightings=None,
        return_components=False,
        **kwargs,
    ):
        """Runs the pipeline to generate FluxProjection objects of all components in the
        AGN model.

        :param inclination_angle: None or int/float, defines the inclination of the AGN
            to project with respect to, in degrees. If None, uses the AGN's stored
            inclination angle. If int/float, the accretion disk must be defined as a
            basic accretion disk.
        :param observer_frame_wavelengths_in_nm: (list of) int(s)/float(s) of observer
            frame wavelengths to project the AGN at. Cannot be used with
            speclite_filter.
        :param speclite_filter: (list of) speclite filter(s) to calculate the projected
            components to.
        :param blr_weightings: list of efficiency arrays to be used with each
            BroadLineRegion component to simulate the local optimally emitting cloud
            model. Each weighting must be a 2 dimensional array of shape (R, Z).
        :param return_components: boolean toggle to return a list of all FluxProjection
            objects
        :return: FluxProjection representing the sum of all projectable components
        """

        output_projections = visualization_pipeline(
            self,
            inclination_angle=inclination_angle,
            observer_frame_wavelengths_in_nm=observer_frame_wavelengths_in_nm,
            speclite_filter=speclite_filter,
            blr_weightings=blr_weightings,
            return_components=return_components,
            **kwargs,
        )

        return output_projections

    def generate_kwarg_dictionaries_for_individual_components(self):
        """Generates keyword dictionaries for each component based on the AGN dictionary
        given.

        Helper method for init, not designed to be called manually.
        """

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
            "albedo",
            "efficiency",
            "visc_temp_prof",
        ]

        generic_accretion_disk_kwargs = [
            "temp_array",
            "phi_array",
            "g_array",
            "radii_array",
            "height_array",
            "albedo",
            "r_out_in_gravitational_radii",
        ]
        blr_kwargs = [
            "max_height",
            "rest_frame_wavelength_in_nm",
            "radial_step",
            "height_step",
            "max_radius",
            "line_strength",
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

        for kwarg in self.kwargs.keys():
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
            self.default_accretion_disk_kwargs["smbh_mass_exp"] = self.smbh_mass_exp
            self.generic_accretion_disk_kwargs["smbh_mass_exp"] = self.smbh_mass_exp
            self.blr_kwargs["smbh_mass_exp"] = self.smbh_mass_exp
            self.diffuse_continuum_kwargs["smbh_mass_exp"] = self.smbh_mass_exp
            self.torus_kwargs["smbh_mass_exp"] = self.smbh_mass_exp

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
            self.diffuse_continuum_kwargs["inclination_angle"] = self.inclination_angle

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
