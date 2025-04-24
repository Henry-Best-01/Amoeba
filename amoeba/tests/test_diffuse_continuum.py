import pytest
from amoeba.Classes.diffuse_continuum import DiffuseContinuum
from amoeba.Classes.flux_projection import FluxProjection
import numpy as np
import astropy.units as u
import astropy.constants as const
from amoeba.Util.util import convert_cartesian_to_polar


def test_initialization():

    inclination_angle = 30

    x_vals = np.linspace(-2000, 2000, 100)
    X, Y = np.meshgrid(x_vals, x_vals / np.cos(inclination_angle * np.pi / 180))
    R, Phi = convert_cartesian_to_polar(X, Y)

    smbh_mass_exp = 5
    redshift_source = 1
    radii_array = R
    phi_array = Phi
    cloud_density_radial_dependence = 0
    cloud_density_array = None
    OmM = 0.4
    H0 = 80
    r_in_in_gravitational_radii = 20
    r_out_in_gravitational_radii = 1000
    name = "my first diffuse continuum"

    my_kwargs = {
        "smbh_mass_exp": smbh_mass_exp,
        "redshift_source": redshift_source,
        "inclination_angle": inclination_angle,
        "radii_array": radii_array,
        "phi_array": phi_array,
        "cloud_density_radial_dependence": cloud_density_radial_dependence,
        "cloud_density_array": cloud_density_array,
        "OmM": OmM,
        "H0": H0,
        "r_in_in_gravitational_radii": r_in_in_gravitational_radii,
        "r_out_in_gravitational_radii": r_out_in_gravitational_radii,
        "name": name,
    }

    my_continuum = DiffuseContinuum(**my_kwargs)

    assert my_continuum.name == name
    assert my_continuum.smbh_mass_exp == smbh_mass_exp
    assert my_continuum.redshift_source == redshift_source
    assert my_continuum.inclination_angle == inclination_angle
    assert np.shape(my_continuum.radii_array) == np.shape(radii_array)
    assert (
        my_continuum.cloud_density_radial_dependence == cloud_density_radial_dependence
    )
    assert np.shape(my_continuum.cloud_density_array) == np.shape(radii_array)
    assert my_continuum.OmM == OmM
    assert my_continuum.H0 == H0
    assert my_continuum.r_in_in_gravitational_radii == r_in_in_gravitational_radii
    assert my_continuum.r_out_in_gravitational_radii == r_out_in_gravitational_radii

    my_basic_continuum = DiffuseContinuum(
        smbh_mass_exp=smbh_mass_exp,
        redshift_source=redshift_source,
        inclination_angle=inclination_angle,
        radii_array=radii_array,
        phi_array=phi_array,
        cloud_density_radial_dependence=cloud_density_radial_dependence,
        r_in_in_gravitational_radii=20,
        r_out_in_gravitational_radii=1000,
    )

    assert type(my_basic_continuum) == type(my_continuum)
    assert my_basic_continuum.smbh_mass_exp == smbh_mass_exp
    assert my_basic_continuum.redshift_source == redshift_source
    assert my_basic_continuum.inclination_angle == inclination_angle

    boosted_kwargs = my_kwargs.copy()

    lams = np.asarray([100, 120, 130, 300, 320, 700, 720, 1000])
    etas = np.asarray([0.1, 0.3, 0.15, 0.4, 0.18, 0.33, 0.22, 0.26])
    const_a = 0.3

    boosted_kwargs["emissivity_etas"] = etas
    boosted_kwargs["rest_frame_wavelengths"] = lams
    boosted_kwargs["responsivity_constant"] = const_a
    boosted_kwargs["name"] = "b00sted boi"

    my_boosted_continuum = DiffuseContinuum(**boosted_kwargs)

    assert my_boosted_continuum.name is not my_continuum.name
    assert type(my_boosted_continuum) is type(my_continuum)
    assert type(my_boosted_continuum.rest_frame_wavelengths) is not None
    assert type(my_boosted_continuum.emissivity_etas) is not None
    assert my_boosted_continuum.responsivity_constant is const_a

    with pytest.raises(ValueError):
        my_bad_continuum = DiffuseContinuum(
            smbh_mass_exp=smbh_mass_exp,
            redshift_source=redshift_source,
            inclination_angle=inclination_angle,
            radii_array=radii_array,
            phi_array=phi_array,
            cloud_density_radial_dependence=None,
            r_in_in_gravitational_radii=20,
            r_out_in_gravitational_radii=1000,
        )

    undefined_bounds_kwargs = my_kwargs.copy()
    del undefined_bounds_kwargs["r_out_in_gravitational_radii"]
    del undefined_bounds_kwargs["r_in_in_gravitational_radii"]
    my_unbound_diffuse_continuum = DiffuseContinuum(**undefined_bounds_kwargs)


def test_set_emissivity():

    inclination_angle = 30

    x_vals = np.linspace(-2000, 2000, 100)
    X, Y = np.meshgrid(x_vals, x_vals / np.cos(inclination_angle * np.pi / 180))
    R, Phi = convert_cartesian_to_polar(X, Y)

    smbh_mass_exp = 5
    redshift_source = 1
    radii_array = R
    phi_array = Phi
    cloud_density_radial_dependence = 0
    cloud_density_array = None
    OmM = 0.4
    H0 = 80
    r_in_in_gravitational_radii = 20
    r_out_in_gravitational_radii = 1000

    my_kwargs = {
        "smbh_mass_exp": smbh_mass_exp,
        "redshift_source": redshift_source,
        "inclination_angle": inclination_angle,
        "radii_array": radii_array,
        "phi_array": phi_array,
        "cloud_density_radial_dependence": cloud_density_radial_dependence,
        "OmM": OmM,
        "H0": H0,
        "r_in_in_gravitational_radii": r_in_in_gravitational_radii,
        "r_out_in_gravitational_radii": r_out_in_gravitational_radii,
    }

    my_continuum = DiffuseContinuum(**my_kwargs)

    lams = np.asarray([100, 120, 130, 300, 320, 700, 720, 1000])
    etas = np.asarray([0.1, 0.3, 0.15, 0.4, 0.18, 0.33, 0.22, 0.26])

    my_continuum.set_emissivity(rest_frame_wavelengths=lams, emissivity_etas=etas)

    assert np.shape(my_continuum.rest_frame_wavelengths) == np.shape(lams)
    assert np.shape(my_continuum.emissivity_etas) == np.shape(etas)


def test_set_responsivity_constant():

    inclination_angle = 30

    x_vals = np.linspace(-2000, 2000, 100)
    X, Y = np.meshgrid(x_vals, x_vals / np.cos(inclination_angle * np.pi / 180))
    R, Phi = convert_cartesian_to_polar(X, Y)

    smbh_mass_exp = 5
    redshift_source = 1
    radii_array = R
    phi_array = Phi
    cloud_density_radial_dependence = 0
    cloud_density_array = None
    OmM = 0.4
    H0 = 80
    r_in_in_gravitational_radii = 20
    r_out_in_gravitational_radii = 1000

    my_kwargs = {
        "smbh_mass_exp": smbh_mass_exp,
        "redshift_source": redshift_source,
        "inclination_angle": inclination_angle,
        "radii_array": radii_array,
        "phi_array": phi_array,
        "cloud_density_radial_dependence": cloud_density_radial_dependence,
        "OmM": OmM,
        "H0": H0,
        "r_in_in_gravitational_radii": r_in_in_gravitational_radii,
        "r_out_in_gravitational_radii": r_out_in_gravitational_radii,
    }

    my_continuum = DiffuseContinuum(**my_kwargs)

    const_a = 0.2

    my_continuum.set_responsivity_constant(responsivity_constant=const_a)

    assert my_continuum.responsivity_constant == const_a

    with pytest.raises(AssertionError):
        my_continuum.set_responsivity_constant(responsivity_constant=-1)
    with pytest.raises(AssertionError):
        my_continuum.set_responsivity_constant(responsivity_constant=33)


def test_interpolate_spectrum_to_wavelength():

    inclination_angle = 30

    x_vals = np.linspace(-2000, 2000, 100)
    X, Y = np.meshgrid(x_vals, x_vals / np.cos(inclination_angle * np.pi / 180))
    R, Phi = convert_cartesian_to_polar(X, Y)

    smbh_mass_exp = 5
    redshift_source = 1
    radii_array = R
    phi_array = Phi
    cloud_density_radial_dependence = 0
    cloud_density_array = None
    OmM = 0.4
    H0 = 80
    r_in_in_gravitational_radii = 20
    r_out_in_gravitational_radii = 1000

    lams = [0, 100]
    etas = [0, 1]

    my_kwargs = {
        "smbh_mass_exp": smbh_mass_exp,
        "redshift_source": redshift_source,
        "inclination_angle": inclination_angle,
        "radii_array": radii_array,
        "phi_array": phi_array,
        "cloud_density_radial_dependence": cloud_density_radial_dependence,
        "cloud_density_array": cloud_density_array,
        "OmM": OmM,
        "H0": H0,
        "r_in_in_gravitational_radii": r_in_in_gravitational_radii,
        "r_out_in_gravitational_radii": r_out_in_gravitational_radii,
        "emissivity_etas": etas,
        "rest_frame_wavelengths": lams,
    }

    my_continuum = DiffuseContinuum(**my_kwargs)

    zero_percentile = 0 * (1 + redshift_source)
    fifty_percentile = 50 * (1 + redshift_source)
    hundred_percentile = 100 * (1 + redshift_source)

    val_0 = my_continuum.interpolate_spectrum_to_wavelength(
        zero_percentile,
    )

    val_50 = my_continuum.interpolate_spectrum_to_wavelength(fifty_percentile)

    val_100 = my_continuum.interpolate_spectrum_to_wavelength(hundred_percentile)

    assert val_0 == 0
    assert val_50 == 0.5
    assert val_100 == 1.0

    lams = [0, 100]
    etas = [2, 5]

    my_continuum.set_emissivity(rest_frame_wavelengths=lams, emissivity_etas=etas)

    assert val_0 != my_continuum.interpolate_spectrum_to_wavelength(
        zero_percentile * (1 + redshift_source)
    )

    assert my_continuum.interpolate_spectrum_to_wavelength(
        100 * (1 + redshift_source)
    ) == my_continuum.interpolate_spectrum_to_wavelength(150 * (1 + redshift_source))

    assert my_continuum.interpolate_spectrum_to_wavelength(
        0
    ) == my_continuum.interpolate_spectrum_to_wavelength(-20)


def test_get_diffuse_continuum_emission():

    inclination_angle = 30

    x_vals = np.linspace(-2000, 2000, 100)
    X, Y = np.meshgrid(x_vals, x_vals / np.cos(inclination_angle * np.pi / 180))
    R, Phi = convert_cartesian_to_polar(X, Y)

    smbh_mass_exp = 5
    redshift_source = 1
    radii_array = R
    phi_array = Phi
    cloud_density_radial_dependence = 0
    cloud_density_array = None
    OmM = 0.4
    H0 = 80
    r_in_in_gravitational_radii = 20
    r_out_in_gravitational_radii = 1000

    lams = [0, 100]
    etas = [0, 1]

    my_kwargs = {
        "smbh_mass_exp": smbh_mass_exp,
        "redshift_source": redshift_source,
        "inclination_angle": inclination_angle,
        "radii_array": radii_array,
        "phi_array": phi_array,
        "cloud_density_radial_dependence": cloud_density_radial_dependence,
        "cloud_density_array": cloud_density_array,
        "OmM": OmM,
        "H0": H0,
        "r_in_in_gravitational_radii": r_in_in_gravitational_radii,
        "r_out_in_gravitational_radii": r_out_in_gravitational_radii,
        "emissivity_etas": etas,
        "rest_frame_wavelengths": lams,
    }

    my_continuum = DiffuseContinuum(**my_kwargs)

    wavelength_obs = 40

    my_projection = my_continuum.get_diffuse_continuum_emission(wavelength_obs)

    assert isinstance(my_projection, FluxProjection)

    assert my_projection.redshift_source == redshift_source
    assert my_projection.rest_frame_wavelength_in_nm == wavelength_obs / (
        1 + redshift_source
    )
    assert my_projection.smbh_mass_exp == smbh_mass_exp
    assert my_projection.mass == my_continuum.mass
    assert my_projection.r_out_in_gravitational_radii == r_out_in_gravitational_radii
    assert my_projection.inclination_angle == inclination_angle
    assert my_projection.rg == my_continuum.rg
    assert my_projection.OmM == OmM
    assert my_projection.H0 == H0
    assert my_projection.lum_dist == my_continuum.lum_dist


def test_get_diffuse_continuum_mean_lag():

    inclination_angle = 0

    x_vals = np.linspace(-100, 100, 1000)
    X, Y = np.meshgrid(x_vals, x_vals / np.cos(inclination_angle * np.pi / 180))
    R, Phi = convert_cartesian_to_polar(X, Y)

    smbh_mass_exp = 8
    redshift_source = 2
    radii_array = R
    phi_array = Phi
    cloud_density_radial_dependence = 0
    cloud_density_array = None
    OmM = 0.4
    H0 = 80
    r_in_in_gravitational_radii = 30
    r_out_in_gravitational_radii = 100

    first_mask = R >= r_in_in_gravitational_radii
    second_mask = R <= r_out_in_gravitational_radii
    r_mask = np.logical_and(first_mask, second_mask)
    R *= r_mask

    lams = [100, 300, 1000]
    etas = [0.2, 0.45, 0.3]

    my_kwargs = {
        "smbh_mass_exp": smbh_mass_exp,
        "redshift_source": redshift_source,
        "inclination_angle": inclination_angle,
        "radii_array": radii_array,
        "phi_array": phi_array,
        "cloud_density_radial_dependence": cloud_density_radial_dependence,
        "cloud_density_array": cloud_density_array,
        "OmM": OmM,
        "H0": H0,
        "r_in_in_gravitational_radii": r_in_in_gravitational_radii,
        "r_out_in_gravitational_radii": r_out_in_gravitational_radii,
        "emissivity_etas": etas,
        "rest_frame_wavelengths": lams,
    }

    my_continuum = DiffuseContinuum(**my_kwargs)

    wavelength_obs = 400

    mean_lag = my_continuum.get_diffuse_continuum_mean_lag(wavelength_obs)

    assert isinstance(mean_lag, float)
    assert mean_lag > 0

    my_continuum.cloud_density_radial_dependence = -1

    new_mean_lag = my_continuum.get_diffuse_continuum_mean_lag(wavelength_obs)

    assert isinstance(new_mean_lag, float)
    assert new_mean_lag > 0
    assert new_mean_lag < mean_lag

    my_continuum.cloud_density_radial_dependence = 1

    large_mean_lag = my_continuum.get_diffuse_continuum_mean_lag(wavelength_obs)

    assert isinstance(large_mean_lag, float)
    assert large_mean_lag > mean_lag

    # test the direct method of numerical integration
    cloud_density_radial_dependence = None
    cloud_density_array = np.ones(np.shape(R))
    my_kwargs["cloud_density_radial_dependence"] = cloud_density_radial_dependence
    my_kwargs["cloud_density_array"] = cloud_density_array
    my_kwargs["r_out_in_gravitational_radii"] = 50

    brute_continuum = DiffuseContinuum(**my_kwargs)

    brute_force_continuum_lag = brute_continuum.get_diffuse_continuum_mean_lag(
        wavelength_obs
    )

    assert isinstance(brute_force_continuum_lag, float)
    # assure better than 5% convergence for numerical integration
    percent_diff = abs(brute_force_continuum_lag - mean_lag) / mean_lag
    assert percent_diff <= 0.05


def test_get_diffuse_continuum_lag_contribution():

    inclination_angle = 0

    x_vals = np.linspace(-100, 100, 1000)
    X, Y = np.meshgrid(x_vals, x_vals / np.cos(inclination_angle * np.pi / 180))
    R, Phi = convert_cartesian_to_polar(X, Y)

    smbh_mass_exp = 8
    redshift_source = 2
    radii_array = R
    phi_array = Phi
    cloud_density_radial_dependence = 0
    cloud_density_array = None
    OmM = 0.4
    H0 = 80
    r_in_in_gravitational_radii = 30
    r_out_in_gravitational_radii = 100

    first_mask = R >= r_in_in_gravitational_radii
    second_mask = R <= r_out_in_gravitational_radii
    r_mask = np.logical_and(first_mask, second_mask)
    R *= r_mask

    lams = [100, 300, 1000]
    etas = [0.2, 0.45, 0.3]

    constant_a = 0

    my_kwargs = {
        "smbh_mass_exp": smbh_mass_exp,
        "redshift_source": redshift_source,
        "inclination_angle": inclination_angle,
        "radii_array": radii_array,
        "phi_array": phi_array,
        "cloud_density_radial_dependence": cloud_density_radial_dependence,
        "cloud_density_array": cloud_density_array,
        "OmM": OmM,
        "H0": H0,
        "r_in_in_gravitational_radii": r_in_in_gravitational_radii,
        "r_out_in_gravitational_radii": r_out_in_gravitational_radii,
        "emissivity_etas": etas,
        "rest_frame_wavelengths": lams,
        "responsivity_constant": constant_a,
    }

    my_continuum = DiffuseContinuum(**my_kwargs)

    wavelength_obs = 400

    diffuse_continuum_additional_lag = (
        my_continuum.get_diffuse_continuum_lag_contribution(wavelength_obs)
    )

    diffuse_tau = my_continuum.get_diffuse_continuum_mean_lag(wavelength_obs)

    assert isinstance(diffuse_continuum_additional_lag, float)
    assert diffuse_continuum_additional_lag > 0
    assert diffuse_continuum_additional_lag / diffuse_tau < 1

    # delay_peak_wavelength = lams[argmax[etas]]. These are rest frame.
    longest_delay_wavelength = lams[np.argmax(np.asarray(etas))]
    minimum_delay_wavelength = lams[np.argmin(np.asarray(etas))]
    middle_wavelength = (
        minimum_delay_wavelength
        + (longest_delay_wavelength - minimum_delay_wavelength) / 2
    )
    quarter_wavelenth = (
        minimum_delay_wavelength + (middle_wavelength - minimum_delay_wavelength) / 2
    )

    longest_delay_wavelength *= 1 + redshift_source
    minimum_delay_wavelength *= 1 + redshift_source
    middle_wavelength *= 1 + redshift_source
    quarter_wavelenth *= 1 + redshift_source

    longest_delay = my_continuum.get_diffuse_continuum_lag_contribution(
        longest_delay_wavelength
    )
    minimum_delay = my_continuum.get_diffuse_continuum_lag_contribution(
        minimum_delay_wavelength
    )
    middle_delay = my_continuum.get_diffuse_continuum_lag_contribution(
        middle_wavelength
    )
    quarter_delay = my_continuum.get_diffuse_continuum_lag_contribution(
        quarter_wavelenth
    )

    assert minimum_delay < longest_delay
    assert minimum_delay < middle_delay
    assert minimum_delay < quarter_delay
    assert middle_delay < longest_delay
    assert quarter_delay < middle_delay
