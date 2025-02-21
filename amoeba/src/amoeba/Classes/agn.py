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
)
from speclite.filters import load_filter, load_filters
import speclite


class Agn:

    def __init__(self, agn_name="", **kwargs):
        """
        This is the main class which connects each component of the AGN together
        and allows for a consistent calculation of each component.
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
        if "H0" in self.kwargs.keys():
            self.H0 = self.kwargs["H0"]
        self.name = agn_name

        self.disk_is_updatable = True
        self.intrinsic_light_curve = None
        self.blr_indicies = []
        self.line_strengths = {}
        self.line_widths = {}

        self.generate_kwarg_dictionaries_for_individual_components()

        self.components = {}

    def add_default_accretion_disk(self, **kwargs):
        """use create_maps to generate an accretion disk. then store it in AGN"""

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
        for kwarg in agn_disk_dictionary.keys():
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

    def add_blr(self, blr_index=0, line_strength=1, line_width=10, **kwargs):
        """give me a dictionary containing information about a BLR. Append to a list of BLRs, with weighting
        factor of line_strength."""

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

        agn_blr = BroadLineRegion(**self.blr_kwargs)

        self.components["blr_" + str(blr_index)] = agn_blr
        self.line_strengths[str(blr_index)] = line_strength
        self.line_widths[str(blr_index)] = line_width
        if blr_index not in self.blr_indicies:
            self.blr_indicies.append(blr_index)

        return True

    def add_streamline_bounded_region_to_blr(self, blr_index=0, **kwargs):
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

        return True

    def add_streamline_bounded_region_to_torus(self, **kwargs):
        """define the torus with a streamline"""

        required_kwargs = ["Streamline"]

        for kwarg in required_kwargs:
            if kwarg not in kwargs:
                print("streamline required with keyword Streamline, torus not updated")
                return False

        self.components["torus"].add_streamline_bounded_region(**kwargs)

        return True

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
            "responsivity_constant",
            "rest_frame_wavelengths",
            "emissivity_etas",
        ]

        for kwarg in all_kwargs:
            if kwarg in kwargs:
                self.diffuse_continuum_kwargs[kwarg] = kwargs[kwarg]

        agn_dc = DiffuseContinuum(**self.diffuse_continuum_kwargs)

        self.components["diffuse_continuum"] = agn_dc

        return True

    def add_intrinsic_signal_parameters(self, **kwargs):
        """define the parameters which will be used to generate the intrinsic signal"""

        if "power_spectrum" in kwargs.keys():
            self.power_spectrum = kwargs["power_spectrum"]
        if "frequencies" in kwargs.keys():
            self.frequencies = kwargs["frequencies"]

        # random seed is optional
        if "random_seed" in kwargs.keys():
            self.random_seed = kwargs["random_seed"]
        else:
            self.random_seed = None

        return True

    def visualize_static_accretion_disk(self, observed_wavelength_in_nm):
        """create FluxProjection of accretion disk"""

        accretion_disk_flux_projection = self.components[
            "accretion_disk"
        ].calculate_surface_intensity_map(observed_wavelength_in_nm)

        return accretion_disk_flux_projection

    def visualize_static_blr(self, blr_index=0, **kwargs):
        """create FluxProjection of BLR with index blr_index"""

        if "velocity_range" not in kwargs:
            kwargs["velocity_range"] = [-1, 1]

        blr_flux_projection = self.components[
            "blr_" + str(blr_index)
        ].project_blr_intensity_over_velocity_range(self.inclination_angle, **kwargs)

        return blr_flux_projection

    def visualize_torus_obscuration(self, observed_wavelength_in_nm, **kwargs):
        """create FluxProjection in magnitudes to use as extinction"""

        torus_extinction_projection = self.components[
            "torus"
        ].project_extinction_to_source_plane(
            self.inclination_angle, observed_wavelength_in_nm
        )

        return torus_extinction_projection

    def visualize_static_diffuse_continuum(self, observed_wavelength_in_nm, **kwargs):
        """create FluxProjection of diffuse continuum"""

        diffuse_continuum_projection = self.components[
            "diffuse_continuum"
        ].get_diffuse_continuum_emission(observed_wavelength_in_nm, **kwargs)

        return diffuse_continuum_projection

    def generate_intrinsic_signal(self, max_time_in_days, **kwargs):
        """generate a driving signal based on stored variability parameters"""

        for kwarg in kwargs.keys():
            if kwarg is "power_spectrum":
                self.power_spectrum = kwargs[kwarg]
            if kwarg is "frequencies":
                self.frequencies = kwargs[kwarg]
            if kwarg is "random_seed":
                self.random_seed = kwargs[kwarg]

        if self.power_spectrum is None or self.frequencies is None:
            print(
                "please provide psd and frequencies with add_intrinsic_signal_parameters or kwargs"
            )
            return False

        light_curve = generate_signal_from_psd(
            max_time_in_days, self.power_spectrum, self.frequencies, self.random_seed
        )

        self.intrinsic_light_curve = light_curve

        return light_curve

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
        """update inclination in all components"""

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
        """update redshift in all components"""

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

    def update_line_strength(self, blr_index, new_line_strength):
        """update line strength (emitted flux w.r.t. continuum) of a BLR emission line"""

        self.line_strengths[str(blr_index)] = new_line_strength

        return True

    def update_line_width(self, blr_index, new_line_width):
        """update line width (rest frame broadening) of a BLR emission line"""

        self.line_widths[str(blr_index)] = new_line_width

        return True

    def calculate_accretion_disk_transfer_function(
        self, observed_wavelength_in_nm, **kwargs
    ):
        """calculate the transfer function of the accretion disk in units Rg"""

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
        """calculate the transfer function of the BLR in units Rg"""

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
            fractional_weight *= self.line_strengths[key[4:]]
            blr_transfer_function_list.append(
                [fractional_weight, current_transfer_function]
            )

        return blr_transfer_function_list

    def calculate_diffuse_continuum_mean_time_lag(
        self, observed_wavelength_in_nm, **kwargs
    ):
        """calculate the diffuse continuum's mean time lag contribution to
        the time lags in the continuum"""

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
        """run the pipeline to generate the full AGN intrinsic signal
        driving_signal: the signal to propogate though the AGN model. Must be in units of days.
        time_axis: the time stamps of the driving signal to be specified if the driving signal is not
            evenly sampled every day.
        observer_frame_wavelengths_in_nm: a wavelength or list of wavelengths in nm.
        speclite_filter: a speclite filter, list of speclite filters, or list of speclite filter names.
        blr_weightings: a dictionary containing keys that are the blr_indicies, and values representing
            a 2d grid of response efficiencies.
        return_components: a bool which allows the return of each component light curve in addition to
            the combined light curve."""

        if observer_frame_wavelengths_in_nm is None and speclite_filter is None:
            print("please provide a range of wavelengths or a speclite filter to use")
            return False
        if observer_frame_wavelengths_in_nm is not None and speclite_filter is not None:
            print("only provide a range of wavelengths or a speclite filter to use")
            return False

        if speclite_filter is not None:
            if isinstance(speclite_filter, str):
                # want a list of filters in either singular or multiple cases
                try:
                    current_filters = [load_filter(speclite_filter)]
                except:
                    current_filters = load_filters(speclite_filter)
            elif isinstance(speclite_filter, speclite.filters.FilterResponse):
                current_filters = [speclite_filter]
            elif isinstance(speclite_filter, list):
                successful_filters = []
                for item in speclite_filter:
                    if isinstance(item, str):
                        try:
                            cur_filter = load_filter(item)
                            successful_filters.append(cur_filter)
                        except:
                            continue
                    elif isinstance(item, speclite.filters.FilterResponse):
                        successful_filters.append(item)
                    else:
                        print(f"{item} not recognized")
                if len(successful_filters) == 0:
                    print("no filters loaded, no propagation required")
                    return False
                current_filters = successful_filters

            mean_wavelengths = []
            wavelength_ranges = []

            for band in current_filters:
                mean_wavelengths.append(band.effective_wavelength.to(u.nm).value)
                min_wavelength = band.wavelength[np.argmax(band.response > 0.01)] / 10
                total_wavelengths = np.sum(band.response > 0.01)
                wavelength_ranges.append(
                    [
                        int(min_wavelength),
                        int((min_wavelength + total_wavelengths / 10)),
                    ]
                )

        else:
            if isinstance(observer_frame_wavelengths_in_nm, (int, float)):
                mean_wavelengths = [observer_frame_wavelengths_in_nm]
                wavelength_ranges = [
                    [
                        observer_frame_wavelengths_in_nm - 20,
                        observer_frame_wavelengths_in_nm + 20,
                    ]
                ]

            elif isinstance(observer_frame_wavelengths_in_nm, (list, np.ndarray)):
                mean_wavelengths = []
                wavelength_ranges = []

                for band in observer_frame_wavelengths_in_nm:
                    if isinstance(band, (int, float)):
                        mean_wavelengths.append(band)
                        wavelength_ranges.append([band - 20, band + 20])
                    elif isinstance(band, (list, np.ndarray)):
                        mean_wavelengths.append(np.mean(band))
                        wavelength_ranges.append([np.min(band), np.max(band)])

        if len(mean_wavelengths) == 0:
            print(
                "please provide a speclite filter, wavelength, wavelength range, or list containing \n previously mentioned types"
            )
            return False

        # check if there is an accretion disk to convert driving light curve to optical light curves
        if "accretion_disk" not in self.components.keys():
            print(
                "please add an accretion disk model to this agn, other components require the variable continuum."
            )
            return False

        # check if there's a signal to propagate
        if intrinsic_light_curve is None:
            if self.intrinsic_light_curve is None:
                try:
                    self.generate_intrinsic_signal(len(self.frequencies))
                except:
                    print(
                        "please provide a psd and set of frequencies, or a driving light curve"
                    )
                    return False
            # define it this way so a provided light curve overrides this propagation,
            # but does not override the stored light curve.
            intrinsic_light_curve = self.intrinsic_light_curve.copy()

        # generate the continuum signals
        reprocessed_signals = []
        for wavelength in mean_wavelengths:
            cur_tf = self.components[
                "accretion_disk"
            ].construct_accretion_disk_transfer_function(wavelength)

            if "diffuse_continuum" in self.components.keys():
                cur_dc_mean_lag_increase = self.components[
                    "diffuse_continuum"
                ].get_diffuse_continuum_lag_contribution(wavelength)
                lag_increase = np.zeros(int(cur_dc_mean_lag_increase))
                cur_tf = np.concatenate((lag_increase, cur_tf))

            t_ax, cur_signal = convolve_signal_with_transfer_function(
                smbh_mass_exp=self.smbh_mass_exp,
                driving_signal=intrinsic_light_curve,
                initial_time_axis=time_axis,
                transfer_function=cur_tf,
                redshift_source=0,
                desired_cadence_in_days=0.1,
            )
            reprocessed_signals.append([t_ax, cur_signal])
        output_signals = reprocessed_signals.copy()

        # generate the blr's response to the optical continuum, if any
        blr_signals = {}

        if len(self.blr_indicies) > 0:

            for index in self.blr_indicies:
                blr_signals[str(index)] = []
                # cur_contamination_signals = []
                observer_frame_emission_line_wavelength = self.components[
                    "blr_" + str(index)
                ].rest_frame_wavelength_in_nm * (1 + self.redshift_source)

                for jj, wavelength_range in enumerate(wavelength_ranges):
                    if (
                        observer_frame_emission_line_wavelength
                        < wavelength_range[0] - self.line_widths[str(index)]
                    ):
                        blr_signals[str(index)].append([0, 0, 0])
                        continue
                    if (
                        observer_frame_emission_line_wavelength
                        > wavelength_range[1] + self.line_widths[str(index)]
                    ):
                        blr_signals[str(index)].append([0, 0, 0])
                        continue

                    # note: the weighting_factor below is representative of how much of the broad line
                    # falls within the filter. The line_strength associated with the broad line represents
                    # the total relative strength of the emission line w.r.t. the continuum
                    if blr_weightings is not None:
                        weighting_factor, cur_blr_tf = self.components[
                            "blr_" + str(index)
                        ].calculate_blr_emission_line_transfer_function(
                            self.inclination_angle,
                            observed_wavelength_range_in_nm=wavelength_range,
                            emission_efficiency_array=blr_weightings[str(index)],
                        )
                    else:
                        weighting_factor, cur_blr_tf = self.components[
                            "blr_" + str(index)
                        ].calculate_blr_emission_line_transfer_function(
                            self.inclination_angle,
                            observed_wavelength_range_in_nm=wavelength_range,
                        )

                    t_ax, contaminated_signals = convolve_signal_with_transfer_function(
                        smbh_mass_exp=self.smbh_mass_exp,
                        driving_signal=reprocessed_signals[jj][1],
                        initial_time_axis=reprocessed_signals[jj][0],
                        transfer_function=cur_blr_tf,
                        redshift_source=0,
                        desired_cadence_in_days=0.1,
                    )

                    blr_signals[str(index)].append(
                        [t_ax, contaminated_signals, weighting_factor]
                    )

        # add blr contamination, if any. Also redshift the time axis to the observer's frame of reference.
        for jj, wavelength_range in enumerate(wavelength_ranges):

            cur_weighting = 1
            original_mean = np.mean(reprocessed_signals[jj][1])
            original_std = np.std(reprocessed_signals[jj][1])

            cur_signal = reprocessed_signals[jj][1] - original_mean
            if original_std != 0:
                cur_signal /= original_std

            if len(self.blr_indicies) > 0:
                for index in self.blr_indicies:
                    if isinstance(blr_signals[str(index)], list):
                        if not isinstance(blr_signals[str(index)][jj], list):
                            continue
                        cur_weighting += self.line_strengths[str(index)]

                        cur_blr_signal = blr_signals[str(index)][jj][1]
                        cur_blr_signal -= np.mean(cur_blr_signal)
                        if np.std(cur_blr_signal) != 0:
                            cur_blr_signal /= np.std(cur_blr_signal)

                        cur_signal += (
                            cur_blr_signal
                            * self.line_strengths[str(index)]
                            * blr_signals[str(index)][jj][2]
                        )
            if cur_weighting != 0:
                cur_signal /= cur_weighting
            if original_std != 0:
                cur_signal *= original_std
            cur_signal += original_mean

            output_signals[jj] = [
                reprocessed_signals[jj][0] * (1 + self.redshift_source),
                cur_signal,
            ]

        if return_components is True:
            return [reprocessed_signals, blr_signals, output_signals]
        return output_signals

    def visualize_agn_pipeline(self, **kwargs):
        """run the pipeline to generate the full flux distribution of
        each of the AGN components"""

        # code that runs sequential operations
        full_agn_flux_projection = 0

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
