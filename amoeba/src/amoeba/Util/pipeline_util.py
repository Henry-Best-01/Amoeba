import numpy as np
import astropy.units as u
import astropy.constants as const
from speclite.filters import (
    load_filter,
    load_filters,
)
import speclite
from amoeba.Util.util import convolve_signal_with_transfer_function


def intrinsic_signal_propagation_pipeline_for_agn(
    AGN,
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
    if "accretion_disk" not in AGN.components.keys():
        print(
            "please add an accretion disk model to this agn, other components require the variable continuum."
        )
        return False

    # check if there's a signal to propagate
    if intrinsic_light_curve is None:
        if AGN.intrinsic_light_curve is None:
            try:
                AGN.generate_intrinsic_signal(len(AGN.frequencies))
            except:
                print(
                    "please provide a psd and set of frequencies, or a driving light curve"
                )
                return False
        # define it this way so a provided light curve overrides this propagation,
        # but does not override the stored light curve.
        intrinsic_light_curve = AGN.intrinsic_light_curve.copy()
        time_axis = AGN.intrinsic_light_curve_time_axis.copy()

    # generate the continuum signals
    reprocessed_signals = []
    for wavelength in mean_wavelengths:
        cur_tf = AGN.components[
            "accretion_disk"
        ].construct_accretion_disk_transfer_function(wavelength)

        if "diffuse_continuum" in AGN.components.keys():
            cur_dc_mean_lag_increase = AGN.components[
                "diffuse_continuum"
            ].get_diffuse_continuum_lag_contribution(wavelength)
            lag_increase = np.zeros(int(cur_dc_mean_lag_increase))
            cur_tf = np.concatenate((lag_increase, cur_tf))

        t_ax, cur_signal = convolve_signal_with_transfer_function(
            smbh_mass_exp=AGN.smbh_mass_exp,
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

    if len(AGN.blr_indicies) > 0:

        for index in AGN.blr_indicies:
            blr_signals[str(index)] = []
            # cur_contamination_signals = []
            observer_frame_emission_line_wavelength = AGN.components[
                "blr_" + str(index)
            ].rest_frame_wavelength_in_nm * (1 + AGN.redshift_source)

            for jj, wavelength_range in enumerate(wavelength_ranges):
                # if (
                #    observer_frame_emission_line_wavelength
                #    < wavelength_range[0] - AGN.line_widths[str(index)]
                # ):
                #    blr_signals[str(index)].append([0, 0, 0])
                #    continue
                # if (
                #    observer_frame_emission_line_wavelength
                #    > wavelength_range[1] + AGN.line_widths[str(index)]
                # ):
                #    blr_signals[str(index)].append([0, 0, 0])
                #   continue

                # note: the weighting_factor below is representative of how much of the broad line
                # falls within the filter. The line_strength associated with the broad line represents
                # the total relative strength of the emission line w.r.t. the continuum
                if blr_weightings is not None:
                    weighting_factor, cur_blr_tf = AGN.components[
                        "blr_" + str(index)
                    ].calculate_blr_emission_line_transfer_function(
                        AGN.inclination_angle,
                        observed_wavelength_range_in_nm=wavelength_range,
                        emission_efficiency_array=blr_weightings[str(index)],
                    )
                else:
                    weighting_factor, cur_blr_tf = AGN.components[
                        "blr_" + str(index)
                    ].calculate_blr_emission_line_transfer_function(
                        AGN.inclination_angle,
                        observed_wavelength_range_in_nm=wavelength_range,
                    )

                t_ax, contaminated_signals = convolve_signal_with_transfer_function(
                    smbh_mass_exp=AGN.smbh_mass_exp,
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

        if len(AGN.blr_indicies) > 0:
            for index in AGN.blr_indicies:
                if isinstance(blr_signals[str(index)], list):
                    if not isinstance(blr_signals[str(index)][jj], list):
                        continue
                    cur_weighting += AGN.line_strengths[str(index)]

                    cur_blr_signal = blr_signals[str(index)][jj][1]
                    cur_blr_signal -= np.mean(cur_blr_signal)
                    if np.std(cur_blr_signal) != 0:
                        cur_blr_signal /= np.std(cur_blr_signal)

                    cur_signal += (
                        cur_blr_signal
                        * AGN.line_strengths[str(index)]
                        * blr_signals[str(index)][jj][2]
                    )
        if cur_weighting != 0:
            cur_signal /= cur_weighting
        if original_std != 0:
            cur_signal *= original_std
        cur_signal += original_mean

        output_signals[jj] = [
            reprocessed_signals[jj][0] * (1 + AGN.redshift_source),
            cur_signal,
        ]

    if return_components is True:
        return [reprocessed_signals, blr_signals, output_signals]
    return output_signals


def visualization_pipeline(
    AGN,
    inclination_angle=None,
    observer_frame_wavelengths_in_nm=None,
    speclite_filter=None,
    blr_weightings=None,
    return_components=None,
    **kwargs,
):
    """This pipeline produces a flux projection object of the combined emission of each
    agn component.

    Note that depending on flux ratios between various components, one may easily
    dominate the flux distribution
    """

    if inclination_angle is not None:
        update_output = AGN.update_inclination(inclination_angle)
        if update_output is False:
            return False

    if observer_frame_wavelengths_in_nm is None and speclite_filter is None:
        print("please provide a range of wavelengths or a speclite filter to use")
        return False
    if observer_frame_wavelengths_in_nm is not None and speclite_filter is not None:
        print("only provide a range of wavelengths or a speclite filter to use")
        return False

    if speclite_filter is not None:
        if isinstance(speclite_filter, str):
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
    if len(AGN.components.keys()) == 0:
        print(
            "please add at least one coponent model to this agn. You cannot project nothing!"
        )
        return False

    list_of_wavelength_resolved_projections = []
    for jj, wavelength in enumerate(mean_wavelengths):
        list_of_projections = []
        if "accretion_disk" in AGN.components.keys():
            cur_img = AGN.components["accretion_disk"].calculate_surface_intensity_map(
                wavelength
            )
            list_of_projections.append(cur_img)

        if "diffuse_continuum" in AGN.components.keys():
            cur_img = AGN.components[
                "diffuse_continuum"
            ].get_diffuse_continuum_emission(wavelength)
            list_of_projections.append(cur_img)

        if len(AGN.blr_indicies) > 0:

            for kk, index in enumerate(AGN.blr_indicies):

                cur_projection = AGN.components[
                    "blr_" + str(index)
                ].project_blr_intensity_over_velocity_range(
                    AGN.inclination_angle,
                    observed_wavelength_range_in_nm=[wavelength_ranges[jj]],
                )

                if blr_weightings is not None:
                    cur_projection.flux_array *= blr_weightings
                    cur_projection.total_flux *= blr_weightings

                if kk == 0:
                    output_blr_projection = cur_projection
                else:
                    output_blr_projection.add_flux_projection(cur_projection)

            list_of_projections.append(output_blr_projection)
        list_of_wavelength_resolved_projections.append(list_of_projections)
    if return_components:
        return list_of_wavelength_resolved_projections

    output_projections = []
    for list_of_projections in list_of_wavelength_resolved_projections:
        output_projections.append(list_of_projections[0])
        for jj in range(len(list_of_projections) - 1):
            index = jj + 1
            output_projections[-1].add_flux_projection(list_of_projections[index])

    return output_projections
