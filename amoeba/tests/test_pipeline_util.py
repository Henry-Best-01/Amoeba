import numpy as np
import numpy.testing as npt
from amoeba.Classes.accretion_disk import AccretionDisk
from amoeba.Classes.blr import BroadLineRegion
from amoeba.Classes.blr_streamline import Streamline
from amoeba.Classes.agn import Agn
from amoeba.Classes.diffuse_continuum import DiffuseContinuum
from amoeba.Classes.torus import Torus
from amoeba.Classes.flux_projection import FluxProjection
from amoeba.Util.util import (
    create_maps,
    convert_cartesian_to_polar,
)
from amoeba.Util.pipeline_util import (
    visualization_pipeline,
    intrinsic_signal_propagation_pipeline_for_agn,
)
from astropy import units as u
from astropy import constants as const
from speclite.filters import FilterResponse
from speclite.filters import load_filter


def test_intrinsic_signal_propogation_pipeline():

    init_smbh_mass_exp = 8.0
    init_redshift_source = 1.0
    init_inclination_angle = 0.0
    init_corona_height = 10
    init_number_grav_radii = 1000
    init_resolution = 1000
    init_spin = 0
    init_OmM = 0.3
    init_H0 = 70

    my_accretion_disk_kwargs = create_maps(
        init_smbh_mass_exp,
        init_redshift_source,
        init_number_grav_radii,
        init_inclination_angle,
        init_resolution,
        spin=init_spin,
        corona_height=init_corona_height,
    )
    accretion_disk = AccretionDisk(**my_accretion_disk_kwargs)

    x_vals = np.linspace(-2000, 2000, 100)
    X, Y = np.meshgrid(x_vals, x_vals / np.cos(init_inclination_angle * np.pi / 180))
    R, Phi = convert_cartesian_to_polar(X, Y)

    radii_array = R
    phi_array = Phi
    cloud_density_radial_dependence = 0
    cloud_density_array = None
    r_in_in_gravitational_radii = 800
    r_out_in_gravitational_radii = 1000
    name = "my diffuse continuum"

    my_dc_kwargs = {
        "smbh_mass_exp": init_smbh_mass_exp,
        "redshift_source": init_redshift_source,
        "inclination_angle": init_inclination_angle,
        "radii_array": radii_array,
        "phi_array": phi_array,
        "cloud_density_radial_dependence": cloud_density_radial_dependence,
        "cloud_density_array": cloud_density_array,
        "OmM": init_OmM,
        "H0": init_H0,
        "r_in_in_gravitational_radii": r_in_in_gravitational_radii,
        "r_out_in_gravitational_radii": r_out_in_gravitational_radii,
        "name": name,
    }

    my_continuum = DiffuseContinuum(**my_dc_kwargs)

    init_launch_radius = 500  # Rg
    init_launch_theta = 0  # degrees
    init_max_height = 500  # Rg
    init_height_step = 200
    init_rest_frame_wavelength_in_nm = 600
    init_characteristic_distance = init_max_height // 5
    init_asymptotic_poloidal_velocity = 0.2
    init_poloidal_launch_velocity = 10**-5

    test_blr_streamline = Streamline(
        init_launch_radius,
        init_launch_theta,
        init_max_height,
        init_characteristic_distance,
        init_asymptotic_poloidal_velocity,
        poloidal_launch_velocity=init_poloidal_launch_velocity,
        height_step=init_height_step,
    )

    init_launch_theta_angled = 45
    test_blr_streamline_angled = Streamline(
        init_launch_radius,
        init_launch_theta_angled,
        init_max_height,
        init_characteristic_distance,
        init_asymptotic_poloidal_velocity,
        poloidal_launch_velocity=init_poloidal_launch_velocity,
        height_step=init_height_step,
    )

    test_torus_streamline_angled = Streamline(
        init_launch_radius * 10,
        init_launch_theta_angled,
        init_max_height,
        init_characteristic_distance,
        init_asymptotic_poloidal_velocity,
        poloidal_launch_velocity=init_poloidal_launch_velocity,
        height_step=init_height_step,
    )

    my_blr_kwargs = {
        "smbh_mass_exp": init_smbh_mass_exp,
        "max_height": init_max_height,
        "rest_frame_wavelength_in_nm": init_rest_frame_wavelength_in_nm,
        "redshift_source": init_redshift_source,
        "height_step": init_height_step,
    }

    blr = BroadLineRegion(**my_blr_kwargs)

    blr.add_streamline_bounded_region(
        test_blr_streamline,
        test_blr_streamline_angled,
    )

    my_torus_kwargs = {
        "smbh_mass_exp": init_smbh_mass_exp,
        "max_height": init_max_height,
        "redshift_source": init_redshift_source,
        "height_step": init_height_step,
    }

    my_agn = Agn(
        agn_name="Wow, what an AGN.",
        **my_accretion_disk_kwargs,
    )

    my_blr_streamline_kwargs = {
        "InnerStreamline": test_blr_streamline,
        "OuterStreamline": test_blr_streamline_angled,
    }

    my_populated_agn = Agn(
        agn_name="Amazing AGN",
        **my_accretion_disk_kwargs,
    )
    my_populated_agn.add_default_accretion_disk()
    my_populated_agn.add_diffuse_continuum(**my_dc_kwargs)
    my_populated_agn.add_blr(**my_blr_kwargs)
    my_populated_agn.add_streamline_bounded_region_to_blr(**my_blr_streamline_kwargs)

    my_populated_agn.add_torus(**my_torus_kwargs)

    assert not my_populated_agn.intrinsic_signal_propagation_pipeline()
    assert not my_populated_agn.intrinsic_signal_propagation_pipeline(
        observer_frame_wavelengths_in_nm=500, speclite_filter="lsst2023-u"
    )

    assert not my_populated_agn.intrinsic_signal_propagation_pipeline(
        speclite_filter=["yellow"]
    )

    assert not my_populated_agn.intrinsic_signal_propagation_pipeline(
        observer_frame_wavelengths_in_nm=["cyan"]
    )

    assert not my_agn.intrinsic_signal_propagation_pipeline(
        speclite_filter="lsst2023-i"
    )

    assert not my_populated_agn.intrinsic_signal_propagation_pipeline(
        speclite_filter="lsst2023-*"
    )

    frequencies = np.linspace(1 / (2000), 1 / (2), 1000)

    power_spectrum = (1 / frequencies) ** 2

    intrinsic_signal_kwargs = {
        "power_spectrum": power_spectrum,
        "frequencies": frequencies,
    }

    assert my_populated_agn.add_intrinsic_signal_parameters(**intrinsic_signal_kwargs)

    all_my_signals = intrinsic_signal_propagation_pipeline_for_agn(
        my_populated_agn,
        speclite_filter=["lsst2023-u", "lsst2023-z"],
        return_components=True,
    )

    assert len(all_my_signals) == 3
    for jj in range(len(all_my_signals)):
        if isinstance(all_my_signals[jj], list):
            assert len(all_my_signals[jj]) == 2
        else:
            for key in all_my_signals[jj].keys():
                for band in range(len(all_my_signals[jj][key])):
                    assert len(all_my_signals[jj][key][band]) == 3
    for jj in range(len(all_my_signals)):
        for kk in range(len(all_my_signals[jj])):
            if isinstance(all_my_signals[jj], list):
                assert len(all_my_signals[jj][kk]) == 2
                assert len(all_my_signals[jj][kk][0]) == len(all_my_signals[jj][kk][1])
            else:
                for key in all_my_signals[jj].keys():
                    assert len(all_my_signals[jj][key][0]) == len(
                        all_my_signals[jj][key][1]
                    )

    all_my_signals_2 = intrinsic_signal_propagation_pipeline_for_agn(
        my_populated_agn,
        observer_frame_wavelengths_in_nm=[[100, 500], [500, 10000]],
        return_components=True,
    )

    assert len(all_my_signals_2) == 3
    for jj in range(len(all_my_signals_2)):
        if isinstance(all_my_signals_2[jj], list):
            assert len(all_my_signals_2[jj]) == 2
        else:
            for key in all_my_signals_2[jj].keys():
                for band in range(len(all_my_signals_2[jj][key])):
                    assert len(all_my_signals_2[jj][key][band]) == 3
    for jj in range(len(all_my_signals_2)):
        for kk in range(len(all_my_signals_2[jj])):
            if isinstance(all_my_signals_2[jj], list):
                assert len(all_my_signals_2[jj][kk]) == 2
                assert len(all_my_signals_2[jj][kk][0]) == len(
                    all_my_signals_2[jj][kk][1]
                )
            else:
                for key in all_my_signals_2[jj].keys():
                    assert len(all_my_signals_2[jj][key][0]) == len(
                        all_my_signals_2[jj][key][1]
                    )

    all_my_signals_3 = intrinsic_signal_propagation_pipeline_for_agn(
        my_populated_agn,
        observer_frame_wavelengths_in_nm=[100, 500, 10000, 750, 850],
        return_components=False,
    )

    assert len(all_my_signals_3) == 5
    for jj in range(len(all_my_signals_3)):
        assert len(all_my_signals_3[jj]) == 2
    for jj in range(len(all_my_signals_3)):
        for kk in range(len(all_my_signals_3[jj])):
            assert len(all_my_signals_3[jj][0]) == len(all_my_signals_3[jj][1])

    my_only_signal = intrinsic_signal_propagation_pipeline_for_agn(
        my_populated_agn, observer_frame_wavelengths_in_nm=[[300, 700]]
    )

    assert len(my_only_signal) == 1
    assert len(my_only_signal[0]) == 2
    assert len(my_only_signal[0][0]) == len(my_only_signal[0][1])

    my_other_only_signal = intrinsic_signal_propagation_pipeline_for_agn(
        my_populated_agn, speclite_filter="lsst2023-u"
    )

    assert len(my_other_only_signal) == 1
    assert len(my_other_only_signal[0]) == 2
    assert len(my_other_only_signal[0][0]) == len(my_other_only_signal[0][1])

    previous_filter = load_filter("lsst2023-u")
    assert isinstance(previous_filter, FilterResponse)
    alt_my_only_other_signal = intrinsic_signal_propagation_pipeline_for_agn(
        my_populated_agn, speclite_filter=previous_filter
    )

    assert len(alt_my_only_other_signal) == len(my_other_only_signal)
    assert np.sum(abs(my_other_only_signal[0][0] - alt_my_only_other_signal[0][0])) == 0
    assert np.sum(abs(my_other_only_signal[0][1] - alt_my_only_other_signal[0][1])) == 0

    smol_list_filter_responses = [
        load_filter("lsst2023-r"),
        load_filter("lsst2023-z"),
    ]

    smol_list_of_responses = intrinsic_signal_propagation_pipeline_for_agn(
        my_populated_agn, speclite_filter=smol_list_filter_responses
    )

    assert len(smol_list_of_responses) == len(smol_list_filter_responses)
    assert len(smol_list_of_responses[0]) == 2
    assert len(smol_list_of_responses[1]) == 2
    assert len(smol_list_of_responses[0][0]) == len(smol_list_of_responses[0][1])
    assert len(smol_list_of_responses[1][0]) == len(smol_list_of_responses[1][1])

    smol_list_filter_responses.append("something that isn't a filter")
    smol_list_filter_responses.append(np.ones(15))

    smol_list_of_responses = intrinsic_signal_propagation_pipeline_for_agn(
        my_populated_agn, speclite_filter=smol_list_filter_responses
    )

    assert len(smol_list_of_responses) < len(smol_list_filter_responses)
    assert len(smol_list_of_responses[0]) == 2
    assert len(smol_list_of_responses[1]) == 2
    assert len(smol_list_of_responses[0][0]) == len(smol_list_of_responses[0][1])
    assert len(smol_list_of_responses[1][0]) == len(smol_list_of_responses[1][1])

    just_a_lonely_response = intrinsic_signal_propagation_pipeline_for_agn(
        my_populated_agn, observer_frame_wavelengths_in_nm=500
    )
    assert len(just_a_lonely_response) == 1
    assert len(just_a_lonely_response[0]) == 2
    assert len(just_a_lonely_response[0][0]) == len(just_a_lonely_response[0][1])


def test_visualization_pipeline():

    init_smbh_mass_exp = 8.0
    init_eddingtion_ratio = 0.0001
    init_redshift_source = 0.0
    init_inclination_angle = 0.0
    init_corona_height = 10
    init_number_grav_radii = 1000
    init_resolution = 1000
    init_spin = 0
    init_OmM = 0.3
    init_H0 = 70

    my_accretion_disk_kwargs = create_maps(
        init_smbh_mass_exp,
        init_redshift_source,
        init_number_grav_radii,
        init_inclination_angle,
        init_resolution,
        eddington_ratio=init_eddingtion_ratio,
        spin=init_spin,
        corona_height=init_corona_height,
    )
    accretion_disk = AccretionDisk(**my_accretion_disk_kwargs)

    x_vals = np.linspace(-2000, 2000, 100)
    X, Y = np.meshgrid(x_vals, x_vals / np.cos(init_inclination_angle * np.pi / 180))
    R, Phi = convert_cartesian_to_polar(X, Y)

    radii_array = R
    phi_array = Phi
    cloud_density_radial_dependence = 0
    cloud_density_array = None
    r_in_in_gravitational_radii = 800
    r_out_in_gravitational_radii = 1000
    name = "my diffuse continuum"

    my_dc_kwargs = {
        "smbh_mass_exp": init_smbh_mass_exp,
        "redshift_source": init_redshift_source,
        "inclination_angle": init_inclination_angle,
        "radii_array": radii_array,
        "phi_array": phi_array,
        "cloud_density_radial_dependence": cloud_density_radial_dependence,
        "cloud_density_array": cloud_density_array,
        "OmM": init_OmM,
        "H0": init_H0,
        "r_in_in_gravitational_radii": r_in_in_gravitational_radii,
        "r_out_in_gravitational_radii": r_out_in_gravitational_radii,
        "name": name,
    }

    my_continuum = DiffuseContinuum(**my_dc_kwargs)

    init_launch_radius = 500  # Rg
    init_launch_theta = 0  # degrees
    init_max_height = 1000  # Rg
    init_height_step = 200
    init_rest_frame_wavelength_in_nm = 400
    init_characteristic_distance = init_max_height // 5
    init_asymptotic_poloidal_velocity = 0.2
    init_poloidal_launch_velocity = 10**-5

    test_blr_streamline = Streamline(
        init_launch_radius,
        init_launch_theta,
        init_max_height,
        init_characteristic_distance,
        init_asymptotic_poloidal_velocity,
        poloidal_launch_velocity=init_poloidal_launch_velocity,
        height_step=init_height_step,
    )

    init_launch_theta_angled = 45
    test_blr_streamline_angled = Streamline(
        init_launch_radius,
        init_launch_theta_angled,
        init_max_height,
        init_characteristic_distance,
        init_asymptotic_poloidal_velocity,
        poloidal_launch_velocity=init_poloidal_launch_velocity,
        height_step=init_height_step,
    )

    test_torus_streamline_angled = Streamline(
        init_launch_radius * 10,
        init_launch_theta_angled,
        init_max_height,
        init_characteristic_distance,
        init_asymptotic_poloidal_velocity,
        poloidal_launch_velocity=init_poloidal_launch_velocity,
        height_step=init_height_step,
    )

    my_blr_kwargs = {
        "smbh_mass_exp": init_smbh_mass_exp,
        "max_height": init_max_height,
        "rest_frame_wavelength_in_nm": init_rest_frame_wavelength_in_nm,
        "redshift_source": init_redshift_source,
        "height_step": init_height_step,
    }

    blr = BroadLineRegion(**my_blr_kwargs)

    blr.add_streamline_bounded_region(
        test_blr_streamline,
        test_blr_streamline_angled,
    )

    my_torus_kwargs = {
        "smbh_mass_exp": init_smbh_mass_exp,
        "max_height": init_max_height,
        "redshift_source": init_redshift_source,
        "height_step": init_height_step,
    }

    my_agn = Agn(
        agn_name="Wow, what an AGN.",
        **my_accretion_disk_kwargs,
    )

    my_non_updatable_agn = Agn(
        agn_name="broken.",
        **my_accretion_disk_kwargs,
    )
    my_non_updatable_agn.disk_is_updatable = False

    my_blr_streamline_kwargs = {
        "InnerStreamline": test_blr_streamline,
        "OuterStreamline": test_blr_streamline_angled,
    }

    my_populated_agn = Agn(
        agn_name="Amazing AGN",
        **my_accretion_disk_kwargs,
    )

    my_populated_agn.add_default_accretion_disk()
    my_populated_agn.add_diffuse_continuum(**my_dc_kwargs)
    my_populated_agn.add_blr(**my_blr_kwargs)
    my_populated_agn.add_blr(blr_index=1, **my_blr_kwargs)
    my_populated_agn.add_streamline_bounded_region_to_blr(**my_blr_streamline_kwargs)
    my_populated_agn.add_streamline_bounded_region_to_blr(
        blr_index=1, **my_blr_streamline_kwargs
    )
    my_populated_agn.update_line_strength(0, 0.1)

    first_wavelength = [[100, 1000]]
    inclination = 23
    assert not visualization_pipeline(
        my_populated_agn,
        inclination_angle=inclination,
    )
    assert not visualization_pipeline(
        my_non_updatable_agn,
        inclination_angle=inclination,
    )
    return_components = False
    current_output = visualization_pipeline(
        my_populated_agn,
        inclination,
        observer_frame_wavelengths_in_nm=first_wavelength,
        return_components=return_components,
    )
    assert isinstance(current_output, list)
    assert isinstance(current_output[0], FluxProjection)

    my_flux = current_output[0].flux_array.copy()
    my_total = current_output[0].total_flux.copy()

    inclination = 30
    current_output = visualization_pipeline(
        my_populated_agn,
        inclination_angle=inclination,
        observer_frame_wavelengths_in_nm=first_wavelength,
        return_components=return_components,
    )

    assert isinstance(current_output, list)
    assert isinstance(current_output[0], FluxProjection)

    assert abs(np.sum(my_flux) - np.sum(current_output[0].flux_array)) > 0

    return_components = True

    weighted_blr_total_output = visualization_pipeline(
        my_populated_agn,
        inclination_angle=inclination,
        observer_frame_wavelengths_in_nm=first_wavelength,
        return_components=return_components,
    )

    assert isinstance(weighted_blr_total_output, list)
    assert len(weighted_blr_total_output[0]) == 3
    for jj in range(3):
        assert isinstance(weighted_blr_total_output[0][jj], FluxProjection)

    my_populated_agn.update_line_strength(0, 0.2)

    weighted_blr_total_output_2 = visualization_pipeline(
        my_populated_agn,
        inclination_angle=inclination,
        observer_frame_wavelengths_in_nm=first_wavelength,
        return_components=return_components,
    )

    # check the blr weighted by a factor of 0.2 is brighter than the one weighted by 0.1
    assert (
        weighted_blr_total_output[0][2].total_flux
        < weighted_blr_total_output_2[0][2].total_flux
    )

    return_components = False
    weighted_blr_total_output = visualization_pipeline(
        my_populated_agn,
        inclination_angle=inclination,
        observer_frame_wavelengths_in_nm=first_wavelength,
        return_components=return_components,
    )

    # check that weighting the blr increases the total flux when all joined together
    assert current_output[0].total_flux < weighted_blr_total_output[0].total_flux

    # check that we cannot use both wavelength range and speclite filter
    assert not visualization_pipeline(
        my_populated_agn,
        inclination_angle=inclination,
        observer_frame_wavelengths_in_nm=first_wavelength,
        speclite_filter="lsst2023-u",
    )

    # try [list of] speclite filters as strings
    output_one = visualization_pipeline(
        my_populated_agn, inclination_angle=inclination, speclite_filter="lsst2023-u"
    )

    output_multiple = visualization_pipeline(
        my_populated_agn,
        inclination_angle=inclination,
        speclite_filter=["lsst2023-u", "lsst2023-i"],
    )

    my_filter = load_filter("lsst2023-g")

    output_one_filter_response = visualization_pipeline(
        my_populated_agn, inclination_angle=inclination, speclite_filter=my_filter
    )

    my_filters = [
        load_filter("lsst2023-z"),
        my_filter,
        "but this one is not a filter",
        np.ones(12),
    ]

    output_multiple_filter_response = visualization_pipeline(
        my_populated_agn, inclination_angle=inclination, speclite_filter=my_filters
    )

    alt_output_all_filter_response = visualization_pipeline(
        my_populated_agn, speclite_filter="lsst2023-*"
    )

    output_one_wavelength = visualization_pipeline(
        my_populated_agn, observer_frame_wavelengths_in_nm=550
    )

    list_of_wavelengths = [550, 650, 800]

    output_list_of_single_wavelengths = visualization_pipeline(
        my_populated_agn, observer_frame_wavelengths_in_nm=list_of_wavelengths
    )

    assert len(output_one) == 1
    assert len(output_multiple) == 2
    assert len(output_one_filter_response) == 1
    assert len(output_multiple_filter_response) < len(my_filters)
    assert len(output_multiple_filter_response) > len(output_one_filter_response)
    assert len(output_one_wavelength) == 1
    assert len(output_list_of_single_wavelengths) == len(list_of_wavelengths)

    # we should have 6 projections for each u, g, r, i, z, y filter
    assert len(alt_output_all_filter_response) == 6

    assert not visualization_pipeline(
        my_populated_agn, speclite_filter=["this one has no filters"]
    )
