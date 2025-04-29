import numpy as np
from astropy import units as u
from astropy import constants as const
from scipy.integrate import quad
from scipy import fft
import scipy
from astropy.io import fits
from numpy.random import rand
import astropy
import warnings
from scipy.ndimage import rotate
from skimage.transform import rescale
from scipy.interpolate import interp1d
from scipy.signal import convolve
from speclite.filters import load_filter, load_filters, FilterResponse

warnings.filterwarnings("ignore")
np.seterr(divide="ignore", invalid="ignore")


def create_maps(
    smbh_mass_exp,
    redshift_source=0,
    number_grav_radii=500,
    inclination_angle=0,
    resolution=500,
    spin=0,
    eddington_ratio=0.1,
    temp_beta=0,
    corona_height=6,
    albedo=0,
    eta_x=0.0,
    generic_beta=False,
    disk_acc=None,
    height_array=None,
    OmM=0.3,
    H0=70,
    efficiency=0.1,
    visc_temp_prof="SS",
    name="",
):
    """This function sets up a dictionary used to create the AccretionDisk object in
    Amoeba. The dictionary may be modified afterwards as long as particular items are
    kept in the same shape (e.g. the set of arrays corresponding to radii, azimuths,
    redshifting, temperature, heights, and albedos). Note that the mass of the
    supermassive black hole (SMBH) both affects the size scale and the physical
    temperature profile of the accretion disk, so a new accretion disk should be
    produced if the mass is changed. The temperature profile generated follows that
    found in Best et al. 2025. The viscous temperature profile may be modeled as either
    the Shakura-Sunyaev or Novikov-Thorne (SS or NT, respectively) temperature profile.
    This may be further modified by the disk-wind model of Sun et al. 2018 and the
    irradiated disk model of Cackett et al. 2007.

    :param smbh_mass_exp: the solution of log10(m_smbh / m_sun)
    :param redshift_source: the redshift of the accretion disk
    :param number_grav_radii: the max radius of the accretion disk in gravitational
        radii (R_g). Typical optical accretion disks require between 500 and 2000 R_g to
        capture most of the flux.
    :param inclination_angle: the inclination of the accretion disk w.r.t. the observer,
        in degrees
    :param resolution: the number of pixels along one axis the images are resolved to.
        All images are created square.
    :param spin: the dimensionless spin parameter of the black hole, ranging from [-1,
        1]. Negative values represent retrograde accretion orbits w.r.t. the angular
        momentum of the black hole.
    :param eddington_ratio: the eddington ratio which the accretion disk is emitting
        radiation at. Thin disks typically range from 0.01 to 0.2, where lower and
        higher eddington ratios require different accretion modes (e.g. slim disk,
        ADAF).
    :param temp_beta: a wind parameter which serves to adjust the temperature profile
        (see Sun et al. 2018). Note that using the parameter "generic_beta==True" will
        force r^-beta dependence instead.
    :param corona_height: number of R_g above the accretion disk the assumed
        lamppost is. Determines the geometric factor in the irradiated accretion disk
        model (see Cackett et al. 2007)
    :param albedo: reflectivity of disk. Setting to 0 will make the disk absorb
        emission, heating it up. Experimental: may be defined as a 2-dimensional array.
    :param eta_x: lamppost source luminosity coefficient. Defined as Lx = eta * M_dot *
        c^2. Note that this is similar but not the same as the eta associated with the
        efficiency of energy conversion to calculate Bolometric flux. Ideally, the sum of
        this eta and the eta associated with the spin of the black hole (typically ~0.10,
        which peaks around 0.42, and actually defined in code as "efficiency") should
        not exceed 1 due to conservation of energy.
    :param generic_beta: boolean toggle to r^{-beta} dependence of the accretion disk.
        If true, the beta value of the equation in Sun et al. 2018 will be calculated from
        the beta parameter.
    :param disk_acc: optional amount of mass accreted by the accretion disk per time. If a
        number is given, units of solar_masses/year are assumed. However, using the
        eddingtion ratio parameter is more typical.
    :param height_array: a 2 dimensional array representing the height of the accretion disk
        at each coordinate. Note that this is an experimental feature.
    :param OmM: mass contribution to the energy budget of the universe
    :param H0: Hubble constant in units km/s/Mpc
    :param efficiency: efficiency of the conversion of gravitational potential energy to
        thermal energy. Typially taken to be ~0.1, may be spin dependent with a maximum
        efficiency of ~0.42, and minimum ~0.02.
    :param visc_temp_prof: string representing viscous temperature profile to use. Currently
        implemented are "SS" for Shakura-Sunyaev and "NT" for Novikov-Thorne. Future
        implimentation: allow for a 1d array representing any radial temperature profile.
    :param name: string representing a name or identifier for the accretion disk. Cosmetic.
    :return: a dictionary designed to be passed into the AccretionDisk object with all
        relevant metadata.
    """
    try:
        import sim5

        sim5_installed = True  # pragma: no cover
    except ModuleNotFoundError:
        sim5_installed = False

    assert redshift_source >= 0
    assert inclination_angle >= 0
    assert inclination_angle <= 90
    if inclination_angle == 90:
        inclination_angle -= 0.1
    assert abs(spin) <= 1
    assert temp_beta >= 0
    bh_mass_in_solar_masses = 10**smbh_mass_exp
    bh_mass_in_kg = bh_mass_in_solar_masses * const.M_sun.to(u.kg)
    grav_rad = calculate_gravitational_radius(bh_mass_in_solar_masses)
    temp_array = np.zeros((resolution, resolution))
    g_array = temp_array.copy()
    r_array = temp_array.copy()
    phi_array = temp_array.copy()
    if sim5_installed == True:  # pragma: no cover
        if inclination_angle == 0:
            inclination_angle += 0.1
        bh_rms = sim5.r_ms(spin)
        for yy in range(resolution):
            for xx in range(resolution):
                alpha = ((xx + 0.5) / resolution - 0.5) * 4.0 * number_grav_radii
                beta = ((yy + 0.5) / resolution - 0.5) * 4.0 * number_grav_radii
                gd = sim5.geodesic()
                error = sim5.intp()
                sim5.geodesic_init_inf(
                    inclination_angle * np.pi / 180, abs(spin), alpha, beta, gd, error
                )
                if error.value():
                    continue
                P = sim5.geodesic_find_midplane_crossing(gd, 0)
                if isnan(P):
                    continue
                r = sim5.geodesic_position_rad(gd, P)
                pol = sim5.geodesic_position_pol(gd, P)
                if isnan(r):
                    continue
                if r >= convert_spin_to_isco_radius(spin):
                    phi = (
                        sim5.geodesic_position_azm(gd, r, pol, P) + 5 / 2 * np.pi
                    ) % (2 * np.pi)
                    g_array[xx, yy] = sim5.gfactorK(r, abs(spin), gd.l)
                    phi_array[xx, yy] = phi
                    r_array[xx, yy] = r
    else:
        x_vals = np.linspace(-number_grav_radii, number_grav_radii, resolution)
        y_vals = x_vals.copy() / np.cos(np.pi * inclination_angle / 180)
        X, Y = np.meshgrid(x_vals, y_vals)
        r_array, phi_array = convert_cartesian_to_polar(X, Y)
        g_array = np.ones(np.shape(r_array))
    r_in = convert_spin_to_isco_radius(spin)
    temp_array = accretion_disk_temperature(
        r_array * grav_rad,
        r_in * grav_rad,
        bh_mass_in_solar_masses,
        eddington_ratio,
        beta=temp_beta,
        corona_height=corona_height,
        albedo=albedo,
        eta_x_rays=eta_x,
        generic_beta=generic_beta,
        disk_acc=disk_acc,
        efficiency=efficiency,
        spin=spin,
        visc_temp_prof=visc_temp_prof,
    )
    if isinstance(albedo, (int, float)):
        albedo_array = np.ones(np.shape(temp_array)) * albedo
    else:
        albedo_array = albedo
    disk_params = {
        "smbh_mass_exp": smbh_mass_exp,
        "redshift_source": redshift_source,
        "inclination_angle": inclination_angle,
        "corona_height": corona_height,
        "temp_array": temp_array,
        "phi_array": phi_array,
        "g_array": g_array,
        "radii_array": r_array,
        "r_out_in_gravitational_radii": number_grav_radii,
        "height_array": height_array,
        "albedo_array": albedo_array,
        "spin": spin,
        "OmM": OmM,
        "H0": H0,
        "name": name,
    }

    return disk_params


def calculate_keplerian_velocity(radius_in_meters, mass_in_solar_masses):
    """Helper function to calculate the magnitude of Keplerian velocity of a circular
    orbit around a massive object.

    v_orbit = sqrt(GM/r)

    :param radius_in_meters: radius in units meters or an astropy quantity
    :param mass_in_solar_masses: mass in units solar masses or an astropy quantity
    :return: keplerian velocity represented as a fraction of the speed of light
    """
    if type(radius_in_meters) != u.Quantity:
        radius_in_meters *= u.m
    if type(mass_in_solar_masses) != u.Quantity:
        mass_in_solar_masses *= const.M_sun.to(u.kg)
    return (
        ((const.G * mass_in_solar_masses.to(u.kg) / radius_in_meters) ** (0.5))
        / const.c
    ).value


def convert_spin_to_isco_radius(spin):
    """This helper function converts the dimensionless spin parameter into the ISCO size
    in units R_g.

    :param spin: dimensionless spin of the SMBH on range (-1, 1). Note that spin may
        approach +/- 1, but should not naturally exceeed 0.998 (Thorne 1974).
    :return: The size of the innermost stable circular orbit in units R_g
    """
    if abs(spin) > 1:
        raise ValueError("Spin out of range. Must satisfy -1 <= spin <= 1.")
    z1 = 1 + (1 - spin**2) ** (1 / 3) * ((1 + spin) ** (1 / 3) + (1 - spin) ** (1 / 3))
    z2 = (3 * spin**2 + z1**2) ** (1 / 2)
    return 3 + z2 - np.sign(spin) * ((3 - z1) * (3 + z1 + 2 * z2)) ** (1 / 2)


def convert_eddington_ratio_to_accreted_mass(
    mass_in_solar_masses, eddington_ratio, efficiency=0.1
):
    """This function converts an Eddington Ratio (i.e. 0.15) into the corresponding
    accretion rate in physical units assuming bol_lum = eddington_ratio * edd_lum.

    The following equations hold:

        edd_lum = 4 * pi * G * M * M_proton * c / (sigma_T)
        bol_lum = M_dot * c^2 * efficiency

    therefore:
        M_dot = edd_lum / (efficiency * c^2)

    where:

        edd_lum = Eddingtion luminosity (the maximum luminosity allowed by Bondi
            accretion where the gravitational force is balanced by the radiation pressure
        pi = 3.14...
        G = Gravitational constant
        M = mass of the accreting body
        M_proton = mass of the proton
        c = speed of light
        sigma_T = Thompson cross section of the electron
        bol_lum = bolometric luminosity of the accretion disk assuming energy is released
            proportional to the mass of the accreted material with some efficiency factor
        efficiency = defines the efficiency of conversion of gravitational potential energy
            to radiation energy
        M_dot = accreted matter required for the disk to radiate at the given Eddington ratio

    :param mass_in_solar_masses: mass of SMBH in solar masses or astropy quantity. Note this is NOT smbh_mass_exp!
    :param eddington_ratio: percentage of theoretical Bondi limit of accretion rate
    :param efficiency: conversion efficiency between gravitational potential energy and
        thermal energy
    :return: accreted mass as astropy units
    """
    if type(mass_in_solar_masses) != u.Quantity:
        mass_in_solar_masses *= const.M_sun.to(u.kg)
    edd_lum = (
        4 * np.pi * const.G * mass_in_solar_masses * const.m_p * const.c / const.sigma_T
    )
    bol_lum = edd_lum * eddington_ratio
    return bol_lum / (efficiency * const.c**2)


def accretion_disk_temperature(
    radius_in_meters,
    min_radius_in_meters,
    mass_in_solar_masses,
    eddington_ratio,
    disk_acc=None,
    spin=0,
    visc_temp_prof="SS",
    efficiency=0.1,
    corona_height=6,
    albedo=1,
    eta_x_rays=0.1,
    beta=0,
    generic_beta=False,
):
    """Defines the radial temperature profile of the accretion disk according to the
    function given in Best et al. 2024, which combines a viscous temperature profile
    with the irradiated disk profile and the disk + wind profile. The viscous
    temperature profile may be defined as a Shakura-Sunyaev or Novikov-Thorne profile.
    Future: update to allow any user-defined radial temperature profile.

    Note that parameters are categorized by thermal profile type (viscous, irradiated, wind).

    ----- Thin disk ------
    :param radius_in_meters: radius or list of radii in meters
    :param min_radius_in_meters: inner radius in meters. Typically taken to be the innermost stable
        circular orbit, but may be greater than this if the truncated accretion disk model is used
    :param mass_in_solar_masses: mass of the supermassive black hole (SMBH) in solar masses or an
        astropy unit. Note this is NOT smbh_mass_exp.
    :param eddington_ratio: percent of eddington limit the SMBH is accreting at
    :param disk_acc: Override for accretion rate at inner radius in solar masses per year. This is
        typically not favored over the Eddingtion ratio, but can be useful in some studies.
    :param spin: dimensionless spin of the SMBH w.r.t. the accretion disk flow on the range (-1, 1).
        Negative values represent accretion flows with angular momentum opposing the SMBH, leading to
        larger innermost stable circular orbits. Required for Novikov-Thorne profile.
    :param visc_temp_prof: str representing the Shakura-Sunyaev or Novikov-Thorne thermal profile
        as "SS" or "NT", respectively. Thanks @ Joshua Fagin for coding the Novikov-Thorne profile.
    :param efficiency: efficiency of converting gravitational potential energy into radiation. Typical
        values are ~0.1, and may be as large as ~0.42 for maximally spinning black holes.

    ----- Irradiated disk ------
    :param corona_height: The height of the irradiating source in gravitational radii in the lamppost
        geometry (Cacket et al. 2007). The default value of 6 represents the Schwarzschild ISCO case.
    :param albedo: reflection coefficent of the accretion disk such that 0
        causes perfect absorption and 1 causing perfect reflection of X-ray energy.
        Default is 1, meaning no thermal contribution from the lamppost term. Experimental: a
        2 dimensional array may be used in addition to simulate a changing albedo with radius.
    :param eta_x_rays: efficiency coefficient of lamppost source, defined as Lx = eta_x_rays * L_bol.
        Due to conservation of energy, the sum of eta_x_rays and efficiency should NOT exceed 1,
        since the total energy must come from some physical source.

    ----- Disk + wind ------
    :param beta: wind strength providing the following accretion rate relationship
        m_dot = m0_dot * (r / r_in)^beta from Sun et al. 2018. Note that this can greatly increase
        the total radiated energy and the Eddington ratio is no longer conserved.
    :param generic_beta: boolean toggle to force a thermal profile of the form r^(-beta). This is
        done by computing the beta required to make this dependence occur.

    :return: temperature in Kelvins
    """
    if generic_beta == True:
        dummy = 3 - 4 * beta
        beta = dummy

    if type(radius_in_meters) == u.Quantity:
        radius_in_meters = radius_in_meters.to(u.m).value
    if type(min_radius_in_meters) == u.Quantity:
        min_radius_in_meters = min_radius_in_meters.to(u.m).value
    if disk_acc is None:
        disk_acc = convert_eddington_ratio_to_accreted_mass(
            mass_in_solar_masses, eddington_ratio, efficiency=efficiency
        ).value
    else:
        if type(disk_acc) == u.Quantity:
            disk_acc = disk_acc.to(u.kg / u.s).value
        else:
            disk_acc *= const.M_sun.to(u.kg) / u.yr.to(u.s)
            disk_acc = disk_acc.value

    if type(mass_in_solar_masses) == u.Quantity:
        mass_in_kg = mass_in_solar_masses.to(u.kg).value
        mass_in_solar_masses = mass_in_kg / const.M_sun.to(u.kg).value
    else:
        mass_in_kg = mass_in_solar_masses * const.M_sun.to(u.kg).value
    grav_rad_in_meters = calculate_gravitational_radius(mass_in_solar_masses)
    schwarz_rad_in_meters = 2 * grav_rad_in_meters

    radius_in_grav_rad = radius_in_meters / grav_rad_in_meters
    inner_rad_in_grav_rad = min_radius_in_meters / grav_rad_in_meters

    m0_dot = disk_acc / (inner_rad_in_grav_rad) ** (beta)
    corona_height += 0.5
    corona_height *= calculate_gravitational_radius(mass_in_solar_masses)

    zeroes = radius_in_meters > min_radius_in_meters
    if visc_temp_prof == "SS":
        temp_map = (
            (
                (
                    3.0
                    * const.G
                    * mass_in_kg
                    * m0_dot
                    * (1.0 - (min_radius_in_meters / radius_in_meters) ** (0.5))
                )
                / (8.0 * np.pi * const.sigma_sb * schwarz_rad_in_meters**3)
            )
            ** (0.25)
        ).decompose().value * (
            (radius_in_meters / schwarz_rad_in_meters) ** ((beta - 3) / 4)
        )
    elif visc_temp_prof == "NT":

        x = np.sqrt(radius_in_grav_rad)
        x0 = np.sqrt(inner_rad_in_grav_rad)
        x1 = 2 * np.cos((np.arccos(spin) - np.pi) / 3)
        x2 = 2 * np.cos((np.arccos(spin) + np.pi) / 3)
        x3 = -2 * np.cos(np.arccos(spin) / 3)

        """
        F_NT = (
            1.0
            / (x**7 - 3 * x**5 + 2 * spin * x**4)
            * (
                x
                - x0
                - (3.0 / 2.0) * spin * np.log(x / x0)
                - 3
                * (x1 - spin) ** 2
                / (x1 * (x1 - x2) * (x1 - x3))
                * np.log((x - x1) / (x0 - x1))
                - 3
                * (x2 - spin) ** 2
                / (x2 * (x2 - x1) * (x2 - x3))
                * np.log((x - x2) / (x0 - x2))
                - 3
                * (x3 - spin) ** 2
                / (x3 * (x3 - x1) * (x3 - x2))
                * np.log((x - x3) / (x0 - x3))
            )
        )
        
        temp_map = (
            (
                (
                    3
                    * disk_acc
                    * const.c**6
                    / (8 * np.pi * const.G**2 * mass_in_solar_masses**2)
                )
                * F_NT
                / const.sigma_sb
            )
            ** (0.25)
        ).value
        """
        F_NT = (
            (1 + spin * x**-3) / (x * (1 - 3 * x**-2 + 2 * spin * x**-3) ** (1 / 2))
        ) * (
            x
            - x0
            - 3 * spin / 2 * np.log(x / x0)
            - (3 * (x1 - spin) ** 2 / (x1 * (x1 - x2) * (x1 - x3)))
            * np.log((x - x1) / (x0 - x1))
            - (3 * (x2 - spin) ** 2 / (x2 * (x2 - x3) * (x2 - x1)))
            * np.log((x - x2) / (x0 - x2))
            - (3 * (x3 - spin) ** 2 / (x3 * (x3 - x1) * (x3 - x2)))
            * np.log((x - x3) / (x0 - x3))
        )

        temp_map = (
            (
                F_NT
                * 3
                * const.G
                * mass_in_kg
                * m0_dot
                / (8 * np.pi * const.sigma_sb * schwarz_rad_in_meters**3)
            )
            ** (0.25)
        ).decompose().value * (
            (radius_in_meters / schwarz_rad_in_meters) ** ((beta - 3) / 4)
        )
    else:
        print(
            "Please use visc_temp_prof = 'SS' or 'NT', other values are not supported at this time. \n Revering to SS disk."
        )
        temp_map = (
            (
                (
                    3.0
                    * const.G
                    * mass_in_kg
                    * m0_dot
                    * (1.0 - (min_radius_in_meters / radius_in_meters) ** (0.5))
                )
                / (8.0 * np.pi * const.sigma_sb * schwarz_rad_in_meters**3)
            )
            ** (0.25)
        ).decompose().value * (
            (radius_in_meters / schwarz_rad_in_meters) ** ((beta - 3) / 4)
        )
    visc_temp = temp_map

    geometric_term = (
        (
            (1 - albedo)
            * corona_height
            / (
                4
                * np.pi
                * const.sigma_sb
                * (radius_in_meters**2 + corona_height**2) ** (3 / 2)
            )
        )
        .decompose()
        .value
    )
    x_ray_lum = (eta_x_rays * disk_acc * const.c**2).decompose().value

    temperature = (visc_temp**4 + geometric_term * x_ray_lum) ** (1 / 4) * zeroes

    return np.nan_to_num(temperature)


def planck_law(temperature, rest_wavelength_in_nm):
    """Calculates the spectral radiance of a black body in [W m^-2 m^-1]. Keep in mind
    this is in mks units, making this represent a very long wavelength range! This is
    not in [W m^-2 nm^-1] or [W m^-2 Hz]. To effectively use this to calculate flux, you
    should integrate over some wavelength range. This is efficiently done by converting
    the spectral radiance to some small wavelength range and then approximating the
    integration as a sum over wavelengths.

    :param temperature: int or array of temperature in Kelvins
    :param rest_wavelength_in_nm: int/float representing the rest frame wavelength in
        nanometers or as an astropy unit
    :return: Spectral radiance of a black body in [W/m^2/m].
    """

    if type(rest_wavelength_in_nm) == u.Quantity:
        dummyval = rest_wavelength_in_nm.to(u.m)
        rest_wavelength_in_m = dummyval.value
    elif type(rest_wavelength_in_nm) != u.Quantity:
        dummyval = rest_wavelength_in_nm * u.nm.to(u.m)
        rest_wavelength_in_m = dummyval

    return np.nan_to_num(
        2.0
        * const.h.value
        * const.c.value**2
        * (rest_wavelength_in_m) ** (-5.0)
        * (
            (
                np.e
                ** (
                    const.h.value
                    * const.c.value
                    / (rest_wavelength_in_m * const.k_B.value * temperature)
                )
                - 1.0
            )
            ** (-1.0)
        )
    )


def planck_law_derivative(temperature, rest_wavelength_in_nm):
    """Numerical approximation of the temperature derivative of the Planck law
    calculated through limit definition of the derivative. For all typical temperatures
    associated with black body radiation, one Kelvin is a small change and effective at
    computing the derivative. We note that there is a closed form of this integral, but
    the simplicity of this method avoids potential overflows.

    delta_B \approx \frac{B(T + delta_T, lam) - B(T, lam)}{delta_T}

    with delta_T = 1 Kelvin

    :param temperature: int or array of temperature in Kelvins
    :param rest_wavelength_in_nm: rest frame wavelength in nanometers or astropy unit
    :return: dervative of the spectral radiance w.r.t. temperature, in units [W/m^2/m/K]
    """
    PlanckA = planck_law(temperature, rest_wavelength_in_nm)
    PlanckB = planck_law(temperature + 1, rest_wavelength_in_nm)
    return PlanckB - PlanckA


def calculate_gravitational_radius(mass_in_solar_masses):
    """Calculate the gravitational radius of a massive object following
    gravitational_radius = R_g = G M / c^2.

    :param mass_in_solar_masses: mass of the object in units of solar masses or as an
        astropy quantity
    :return: length of one gravitational radius in meters.
    """
    if isinstance(mass_in_solar_masses, u.Quantity):
        mass_in_kg = mass_in_solar_masses.to(u.kg)
    else:
        mass_in_kg = mass_in_solar_masses * const.M_sun.to(u.kg)

    return (const.G * mass_in_kg / const.c**2).decompose().value


def calculate_angular_diameter_distance(redshift, OmM=0.3, H0=70):
    """This funciton takes in a redshift value, and calculates the angular diameter
    distance. This is given as the output. This assumes LCDM model. Follows Distance
    measures in cosmology (Hogg 1999)

    :param redshift: redshift of the object
    :param OmM: total fraction of the universe's energy budget is in mass.
    :param H0: Hubble constant in units km/s/Mpc
    :return: angular diameter distance in units meters, assuming a flat lambda-CDM
        universe
    """
    OmL = 1 - OmM
    multiplier = (9.26 * 10**25) * (H0 / 100) ** (-1) * (1 / (1 + redshift))
    integrand = lambda z_p: (OmM * (1 + z_p) ** (3.0) + OmL) ** (-0.5)
    integral, _ = quad(integrand, 0, redshift)
    angular_diameter_distance = multiplier * integral

    return angular_diameter_distance


def calculate_angular_diameter_distance_difference(
    redshift_lens, redshift_source, OmM=0.3, H0=70
):
    """This function takes in 2 redshifts, designed to represent z1 = redshift (lens)
    and z2 = redshift (source). This assumes LCDM model. Follows Distance measures in
    cosmology (Hogg 1999)

    :param redshift_lens: redshift the gravitational lens
    :param redshift_source: redshift the source
    :param OmM: total fraction of the universe's energy budget is in mass
    :param H0: Hubble constant in units km/s/Mpc
    :return: angular diameter distance difference in units meters
    """
    if redshift_lens > redshift_source:
        dummy_var = redshift_source
        redshift_source = redshift_lens
        redshift_lens = dummy_var

    OmL = 1 - OmM
    multiplier = (9.26 * 10**25) * (H0 / 100) ** (-1) * (1 / (1 + redshift_source))
    integrand = lambda z_p: (OmM * (1 + z_p) ** (3.0) + OmL) ** (-0.5)
    integral1, _ = quad(integrand, 0, redshift_lens)
    integral2, _ = quad(integrand, 0, redshift_source)
    angular_diameter_distance_difference = multiplier * (integral2 - integral1)

    return angular_diameter_distance_difference


def calculate_luminosity_distance(redshift, OmM=0.3, H0=70):
    """This calculates the luminosity distance using the
    calculate_angular_diameter_distance formula for flat lambda-CDM model. Follows
    Distance measures in cosmology (Hogg 1999)

    :param redshift: redshift of the object
    :param OmM: mass fraction of the universe's energy budget
    :param H0: Hubble constant in units km/s/Mpc
    :return: luminosity distance of the object in meters
    """
    luminosity_distance = (1 + redshift) ** 2 * calculate_angular_diameter_distance(
        redshift, OmM=OmM, H0=H0
    )
    return luminosity_distance


def calculate_angular_einstein_radius(
    redshift_lens=None,
    redshift_source=None,
    mean_microlens_mass_in_kg=1 * const.M_sun.to(u.kg),
    OmM=0.3,
    H0=70,
    D_lens=None,
    D_source=None,
    D_LS=None,
):
    """This function calculates the Einstein radius of the microlens in radians assuming
    the LCDM model.

    :param redshift_lens: redshift of the lens. Required if D_lens is None.
    :param redshift_source: redshift of the source object. Required if D_source is None.
    :param mean_microlens_mass_in_kg: average mass of microlenses in the lensing galaxy.
        This is typically modeled between 0.1 and 1.0 solar masses. Note that this
        formula is used for any compact object which may be approximated by a point
        source, and is valid for relatively compact strongly lensing galaxies.
    :param OmM: energy budget of the universe in mass
    :param H0: Hubble constant in units km/s/Mpc
    :param D_lens: angular diameter distance of the lens. Will be computed if None.
    :param D_source: angular diameter distance of the source. Will be computed if None.
    :param D_LS: angular diameter distance difference between the lens and source. Will
        be computed if None.
    :return: Einstein radius of the lens in radians
    """
    assert redshift_lens is not None or D_lens is not None
    assert redshift_source is not None or D_source is not None

    if D_lens is None:
        D_lens = calculate_angular_diameter_distance(redshift_lens, OmM=OmM, H0=H0)
    if D_source is None:
        D_source = calculate_angular_diameter_distance(redshift_source, OmM=OmM, H0=H0)
    if D_lens > D_source:
        dummy = D_source
        D_source = D_lens
        D_lens = dummy
    if D_LS is None:
        if redshift_lens is not None and redshift_source is not None:
            D_LS = calculate_angular_diameter_distance_difference(
                redshift_lens, redshift_source, OmM=OmM, H0=H0
            )
        else:
            D_LS = abs(D_lens - D_source)

    angular_einstein_radius = (
        (
            (4 * const.G * mean_microlens_mass_in_kg / const.c**2)
            * D_LS
            / (D_lens * D_source)
        )
        ** (0.5)
    ).value

    return angular_einstein_radius


def calculate_einstein_radius_in_meters(
    redshift_lens=None,
    redshift_source=None,
    mean_microlens_mass_in_kg=1 * const.M_sun.to(u.kg),
    OmM=0.3,
    H0=70,
    D_lens=None,
    D_source=None,
    D_LS=None,
):
    """This function determines the einstein radius of the lensing object in meters.

    :param redshift_lens: redshift of the lens. Required if D_lens is None.
    :param redshift_source: redshift of the source object. Required if D_source is None.
    :param mean_microlens_mass_in_kg: average mass of microlenses in the lensing galaxy.
        This is typically modeled between 0.1 and 1.0 solar masses. Note that this
        formula is used for any compact object which may be approximated by a point
        source, and is valid for relatively compact strongly lensing galaxies.
    :param OmM: energy budget of the universe in mass
    :param H0: Hubble constant in units km/s/Mpc
    :param D_lens: angular diameter distance of the lens. Will be computed if None.
    :param D_source: angular diameter distance of the source. Will be computed if None.
    :param D_LS: angular diameter distance difference between the lens and source. Will
        be computed if None.
    :return: Einstein radius of the lens in meters
    """
    if D_source is not None and D_lens is not None:
        if D_source < D_lens:
            dummy = D_source
            D_source = D_lens
            D_lens = dummy
    elif redshift_lens is not None and redshift_source is not None:
        if redshift_source < redshift_lens:
            dummy = redshift_source
            redshift_source = redshift_lens
            redshift_lens = dummy

    if D_source is None:
        ang_diam_dist_source_plane = calculate_angular_diameter_distance(
            redshift_source,
            OmM=OmM,
            H0=H0,
        )
    else:
        ang_diam_dist_source_plane = D_source

    ein_rad_in_radians = calculate_angular_einstein_radius(
        redshift_lens=redshift_lens,
        redshift_source=redshift_source,
        mean_microlens_mass_in_kg=mean_microlens_mass_in_kg,
        OmM=OmM,
        H0=H0,
        D_lens=D_lens,
        D_source=D_source,
        D_LS=D_LS,
    )
    einstein_radius_in_meters = ang_diam_dist_source_plane * ein_rad_in_radians

    return einstein_radius_in_meters


def pull_value_from_grid(array_2d, x_position, y_position):
    """This approximates the point (x_position, y_position) in a 2d array of values.
    x_position and y_position may be decimals, and are assumed to be measured in pixels.
    This uses bilinear interpolation (or linear interpolation if one value is an
    integer).

    :param array_2d: 2 dimensional array of values.
    :param x_position: x coordinate in array_2d in pixels
    :param y_position: y coordinate in array_2d in pixels
    :return: approximation of array_2d at point (x_position, y_position)
    """

    array_2d = np.pad(array_2d, (0, 1), mode="edge")

    if isinstance(x_position, (int, float)) and isinstance(y_position, (int, float)):
        assert x_position >= 0 and y_position >= 0
        assert (
            x_position <= np.size(array_2d, 0) - 1
            and y_position <= np.size(array_2d, 1) - 1
        )

        x_int = x_position // 1
        y_int = y_position // 1
        dx = x_position % 1
        dy = y_position % 1

        base_value = array_2d[int(x_int), int(y_int)]
        base_plus_x = array_2d[int(x_int) + 1, int(y_int)]
        base_plus_y = array_2d[int(x_int), int(y_int) + 1]
        base_plus_x_plus_y = array_2d[int(x_int) + 1, int(y_int) + 1]

        value = (
            base_value * (1 - dx) * (1 - dy)
            + base_plus_x * (1 - dx) * dy
            + base_plus_y * dx * (1 - dy)
            + base_plus_x_plus_y * dx * dy
        )

        array_2d = array_2d[:-2, :-2]

        return value

    else:
        assert min(x_position) >= 0 and min(y_position) >= 0
        assert (
            max(x_position) <= np.size(array_2d, 0) - 1
            and max(y_position) <= np.size(array_2d, 1) - 1
        )

        x_int = x_position // 1
        y_int = y_position // 1
        dx = x_position % 1
        dy = y_position % 1

        base_value = array_2d[(x_int.astype(int)), (y_int.astype(int))]
        base_plus_x = array_2d[(x_int.astype(int) + 1), (y_int.astype(int))]
        base_plus_y = array_2d[(x_int.astype(int)), (y_int.astype(int) + 1)]
        base_plus_x_plus_y = array_2d[(x_int.astype(int) + 1), (y_int.astype(int) + 1)]

        value = (
            base_value * (1 - dx) * (1 - dy)
            + base_plus_x * (1 - dx) * dy
            + base_plus_y * dx * (1 - dy)
            + base_plus_x_plus_y * dx * dy
        )

        array_2d = array_2d[:-2, :-2]

        return value


def convert_1d_array_to_2d_array(array_1d):
    """Converts a 1 dimensional list of rays into its 2 dimensional representation.
    Gerlumph maps are stored as a binary file, as a single list of values. This requires
    a square magnification map.

    :param array_1d: A 1d array (or list) of values which must be restacked into a 2d
        array. Note this array must correspond to a square output array.
    :return: A 2d numpy array representation of the input
    """
    resolution = int(np.size(array_1d) ** 0.5)
    array_2d = np.reshape(np.asarray(array_1d), newshape=(resolution, resolution))

    return array_2d


def convert_cartesian_to_polar(x, y):
    """Converts coordinate pair (x, y) into (r, phi) coordinates. Rotates the phi
    direction such that phi=0 points in the negative y direction (towards the observer
    in our model).

    :param: x value or coordinate
    :param: y value or coordinate in same dimensions as x value
    :return: tuple representation of radius and azimuth coordinates
    """
    r = (x**2 + y**2) ** 0.5
    phi = np.arctan2(y, x)

    phi = (5 / 2 * np.pi + phi) % (2 * np.pi)
    return (r, phi)


def convert_polar_to_cartesian(r, phi):
    """Converts coordinate pair (r, phi) into (x, y) coordinates.

    :param r: radius in polar coordinates
    :param phi: azimuth angle in radians within our azimuth convention
    :return: tuple representation of x and y coordinates
    """

    x = r * np.sin(phi)
    y = -r * np.cos(phi)

    return (x, y)


def perform_microlensing_convolution(
    magnification_array,
    flux_array,
    redshift_lens,
    redshift_source,
    smbh_mass_exp=8.0,
    mean_microlens_mass_in_kg=1.0 * const.M_sun.to(u.kg),
    number_of_microlens_einstein_radii=25,
    number_of_smbh_gravitational_radii=1000,
    relative_orientation=0,
    OmM=0.3,
    H0=70,
    return_preconvolution_information=False,
    random_seed=None,
):
    """This takes a magnification map and convolves it with a 2 dimensional array
    (usually associated with a FluxProjection object).

    :param magnification_array: a 2d array representation of the magnifications due to
        microlenses
    :param flux_array: a 2d array representation of the flux distribution
    :param redshift_lens: an int/float representing the cosmological redshift of the
        lens
    :param redshift_source: an int/float representing the cosmological redshift of the
        source
    :param smbh_mass_exp: a float representing the solution to log_{10} (M_{smbh} /
        M_sun)
    :param mean_microlens_mass_in_kg: the mean mass in kg of the microlenses, typically
        between 0.1 and 1.0 M_sun
    :param number_of_microlens_einstein_radii: size of the magnification map in Einstein
        radii (R_e)
    :param number_of_smbh_gravitational_radii: radial size of the flux_array (e.g. half
        of one square side)
    :param relative_orientation: angular rotation of flux distribution w.r.t.
        microlensing magnification distribution. If int or float, this defines the
        specific orientation. Any other input will be assigned a random value.
    :param OmM: mass fraction of the Universe in the lambda-CDM model
    :param H0: Hubble constant in units km/s/Mpc
    :param return_preconvolution_information: return the rescaled flux_array instead of
        the convolution
    :return: a 2d array representing the real valued convolution between the
        magnification_array and flux_array and the pixel_shift to locate the smbh
        location
    """
    np.random.seed(random_seed)

    if not isinstance(relative_orientation, (int, float)):
        relative_orientation = np.random.rand() * 360
    flux_array = rotate(flux_array, relative_orientation, axes=(0, 1), reshape=False)
    original_total_flux = np.sum(flux_array)

    gravitational_radius_of_smbh = calculate_gravitational_radius(10**smbh_mass_exp)
    pixel_size_flux_array = (
        2
        * (number_of_smbh_gravitational_radii * gravitational_radius_of_smbh)
        / np.size(flux_array, 0)
    )

    pixel_size_magnification_array = (
        number_of_microlens_einstein_radii
        * calculate_einstein_radius_in_meters(
            redshift_lens,
            redshift_source,
            mean_microlens_mass_in_kg=mean_microlens_mass_in_kg,
            OmM=OmM,
            H0=H0,
        )
    ) / np.size(magnification_array, 0)

    pixel_ratio = pixel_size_flux_array / pixel_size_magnification_array

    flux_array_rescaled = rescale(flux_array, pixel_ratio)
    new_total_flux = np.sum(flux_array_rescaled)

    flux_array_rescaled *= original_total_flux / new_total_flux

    if return_preconvolution_information:
        return flux_array_rescaled

    dummy_map = np.zeros(np.shape(magnification_array))
    dummy_map[: np.size(flux_array_rescaled, 0), : np.size(flux_array_rescaled, 1)] = (
        flux_array_rescaled
    )
    convolution = fft.irfft2(fft.rfft2(dummy_map) * fft.rfft2(magnification_array))

    pixel_shift = np.size(flux_array_rescaled, 0) // 2

    output = convolution.real

    return output, pixel_shift


def extract_light_curve(
    convolution_array,
    pixel_size,
    effective_transverse_velocity,
    light_curve_time_in_years,
    pixel_shift=0,
    x_start_position=None,
    y_start_position=None,
    phi_travel_direction=None,
    return_track_coords=False,
    random_seed=None,
):
    """Extracts a light curve from the convolution between two arrays by selecting a
    trajectory and calling pull_value_from_grid at each relevant point. If the light
    curve is too long, or the size of the object is too large, a "light curve"
    representing a constant magnification is returned.

    :param convolution_array: The convolution between a flux distribtion and the
        magnification array due to microlensing. Note coordinates on arrays have (y, x)
        signature.
    :param pixel_size: Physical size of a pixel in the source plane, in meters
    :param effective_transverse_velocity: effective transverse velocity in the source
        plane, in km / s
    :param light_curve_time_in_years: duration of the light curve to generate, in years
    :param pixel_shift: offset of the SMBH with respect to the convolved map, in pixels
    :param x_start_position: None or the x coordinate to start pulling a light curve
        from, in pixels
    :param y_start_position: None or the y coordinate to start pulling a light curve
        from, in pixels
    :param phi_travel_direction: None or the angular direction of travel along the
        convolution, in degrees
    :param return_track_coords: boolean toggle to return the x and y coordinates of the
        track in pixels
    :return: list representing the microlensing light curve
    """
    rng = np.random.default_rng(seed=random_seed)

    if type(effective_transverse_velocity) == u.Quantity:
        effective_transverse_velocity = effective_transverse_velocity.to(
            u.m / u.s
        ).value
    else:
        effective_transverse_velocity *= u.km.to(u.m)
    if type(light_curve_time_in_years) == u.Quantity:
        light_curve_time_in_years = light_curve_time_in_years.to(u.s).value
    else:
        light_curve_time_in_years *= u.yr.to(u.s)

    if pixel_shift >= np.size(convolution_array, 0) / 2:
        print(
            "warning, flux projection too large for this magnification map. Returning average flux."
        )
        return np.sum(convolution_array) / np.size(convolution_array)

    pixels_traversed = (
        effective_transverse_velocity * light_curve_time_in_years / pixel_size
    )

    n_points = (
        effective_transverse_velocity * light_curve_time_in_years / pixel_size
    ) + 2

    if pixel_shift > 0:
        safe_convolution_array = convolution_array[
            pixel_shift : -pixel_shift - 1, pixel_shift : -pixel_shift - 1
        ]
    else:
        safe_convolution_array = convolution_array

    if pixels_traversed >= np.size(safe_convolution_array, 0):
        print(
            "warning, light curve is too long for this magnification map. Returning average flux."
        )
        return np.sum(convolution_array) / np.size(convolution_array)

    if x_start_position is not None:
        if x_start_position < 0:
            print(
                "Warning, chosen position lays in the convolution artifact region. Returning average flux."
            )
            return np.sum(convolution_array) / np.size(convolution_array)
    else:
        x_start_position = rng.integers(0, np.size(safe_convolution_array, 0))

    if y_start_position is not None:
        if y_start_position < 0:
            print(
                "Warning, chosen position lays in the convolution artifact region. Returning average flux."
            )
            return np.sum(convolution_array) / np.size(convolution_array)
    else:
        y_start_position = rng.integers(0, np.size(safe_convolution_array, 1))

    if phi_travel_direction is not None:
        angle = phi_travel_direction * np.pi / 180
        delta_x = pixels_traversed * np.cos(angle)
        delta_y = pixels_traversed * np.sin(angle)

        if (
            x_start_position + delta_x >= np.size(safe_convolution_array, 0)
            or y_start_position + delta_y >= np.size(safe_convolution_array, 1)
            or x_start_position + delta_x < 0
            or y_start_position + delta_y < 0
        ):
            print(
                "Warning, chosen track leaves the convolution array. Returning average flux."
            )
            return np.sum(convolution_array) / np.size(convolution_array)
    else:
        success = None
        angle = rng.random() * 360 * np.pi / 180
        while success is None:
            angle += np.pi / 2
            delta_x = pixels_traversed * np.cos(angle)
            delta_y = pixels_traversed * np.sin(angle)
            if (
                x_start_position + delta_x < np.size(safe_convolution_array, 0)
                and y_start_position + delta_y < np.size(safe_convolution_array, 1)
                and x_start_position + delta_x >= 0
                and y_start_position + delta_y >= 0
            ):
                success = True

    x_positions = np.linspace(
        x_start_position, x_start_position + delta_x, int(n_points)
    )
    y_positions = np.linspace(
        y_start_position, y_start_position + delta_y, int(n_points)
    )

    light_curve = pull_value_from_grid(safe_convolution_array, x_positions, y_positions)

    if return_track_coords:
        return (
            np.asarray(light_curve),
            x_positions + pixel_shift,
            y_positions + pixel_shift,
        )

    return np.asarray(light_curve)


def calculate_time_lag_array(
    radii_array,
    phi_array,
    inclination_angle,
    corona_height,
    axis_offset_in_gravitational_radii=0,
    angle_offset_in_degrees=0,
    height_array=None,
):
    """Calculate the time lag between a lamppost source and every position on the
    accretion disk's plane in units of R_g / c.

    :param radii_array: a 2d array of radial values in units of gravitational radii
    :param phi_array: a 2d array of azimuth values in radians
    :param inclination_angle: the inclination of the object w.r.t. the observer in
        degrees
    :param corona_height: the height of the source in gravitational radii
    :param axis_offset_in_gravitational_radii: the cylindrical radial distance from the
        SMBH axis of symmetry to be used as the source position
    :param angle_offset_in_degrees: azimuth angle in degrees of the offset lamppost.
        Note that 0 degrees is nearest to the observer and 180 degrees is furthest away.
    :param height_array: an optional 2d array of the height values in gravitational
        radii. Note that height array is experimental!
    :return: a 2d array of time lags in units R_g / c
    """
    inclination_angle *= np.pi / 180
    angle_offset_in_radians = angle_offset_in_degrees * np.pi / 180

    x_axis_offset = axis_offset_in_gravitational_radii * np.cos(angle_offset_in_radians)
    y_axis_offset = axis_offset_in_gravitational_radii * np.sin(angle_offset_in_radians)

    if height_array is not None:
        assert np.shape(height_array) == np.shape(radii_array)
        height_array = np.asarray(height_array).copy()
    else:
        height_array = np.zeros(np.shape(radii_array)).copy()

    height_array -= corona_height

    x_array, y_array = convert_polar_to_cartesian(radii_array, phi_array)
    x_array -= x_axis_offset
    y_array -= y_axis_offset

    new_radii, new_azimuths = convert_cartesian_to_polar(x_array, y_array)

    time_lag_array = (
        (new_radii**2 + height_array**2) ** 0.5
        - height_array * np.cos(inclination_angle)
        - new_radii * np.cos(new_azimuths) * np.sin(inclination_angle)
    )

    return time_lag_array


def calculate_geometric_disk_factor(
    temp_array,
    radii_array,
    phi_array,
    smbh_mass_exp,
    corona_height,
    axis_offset_in_gravitational_radii=0,
    angle_offset_in_degrees=0,
    height_array=None,
    albedo_array=None,
):
    """Calculate the geometric factor of the accretion disk due to lamppost heating
    according to.

    f_geo = (1 - A) cos(theta_x) / (4 * pi * sigma_sb * r_{*}^{2}}

    where:
    A is the albedo of the material
    theta_x is the angle of incidence of a ray of radiation
    pi = 3.14...
    sigma_sb = the Stefan-Boltzman constant
    r_{*} = the distance of any 3 dimensional position to the source

    This gets weighted by the lamppost X-ray flux L_{x} (eq. 2 in Cackett+ 2007)

    :param temp_array: a 2d array of effective temperatures in Kelvin
    :param radii_array: a 2d array of radii in gravitational radii
    :param phi_array: a 2d array of azimuth angles in radians
    :param smbh_mass_exp: the solution of log10(m_smbh / m_sun)
    :param corona_height: the height of the source in gravitational radii
    :param axis_offset_in_gravitational_radii: the cylindrical axis offset of the lamppost
        w.r.t. axis of symmetry
    :param angle_offset_in_degrees: azimuth position of the offset lamppost
    :param height_array: array of heights to calculate the disk at. Allows for greater flexability in
        disk model. Note this is experimental!
    :param albedo_array: int, float, or array of albedos (reflectivities) to use for the disk
    :return: a 2d array of geometric disk factors which determine the flux reprocessing of the
        lamppost by the accretion disk
    """

    assert np.shape(temp_array) == np.shape(radii_array)
    if height_array is not None:
        assert np.shape(height_array) == np.shape(radii_array)
        height_array = np.asarray(height_array).copy()
    else:
        height_array = np.zeros(np.shape(radii_array))
    height_array -= corona_height

    angle_offset_in_degrees *= np.pi / 180

    x_axis_offset = axis_offset_in_gravitational_radii * np.cos(angle_offset_in_degrees)
    y_axis_offset = axis_offset_in_gravitational_radii * np.sin(angle_offset_in_degrees)
    x_array, y_array = convert_polar_to_cartesian(radii_array, phi_array)
    x_array -= x_axis_offset
    y_array -= y_axis_offset

    new_radii, new_azimuths = convert_cartesian_to_polar(x_array, y_array)

    if isinstance(albedo_array, (int, float)):
        albedo_array *= np.ones(np.shape(new_radii))
    elif isinstance(albedo_array, (np.ndarray)):
        assert np.shape(albedo_array) == np.shape(new_radii)
    else:
        albedo_array = np.zeros(np.shape(new_radii))

    gravitational_radius = calculate_gravitational_radius(10**smbh_mass_exp)

    height_gradient_x, height_gradient_y = np.gradient(height_array)
    radii_gradient_x, radii_gradient_y = np.gradient(new_radii)

    dh_dr = (
        (height_gradient_x / radii_gradient_x) ** 2
        + (height_gradient_y / radii_gradient_y) ** 2
    ) ** 0.5

    theta_star = np.pi - np.arctan(dh_dr) - np.arctan2(height_array, new_radii)

    theta_star = abs(theta_star % (np.pi))

    cos_theta_star = np.cos(theta_star)

    radii_star = (new_radii**2 + height_array**2) ** 0.5 * gravitational_radius

    return np.nan_to_num(
        (1 - albedo_array)
        * cos_theta_star
        / (4 * np.pi * const.sigma_sb * radii_star**2)
    )


def calculate_dt_dlx(
    temp_array,
    radii_array,
    phi_array,
    smbh_mass_exp,
    corona_height,
    axis_offset_in_gravitational_radii=0,
    angle_offset_in_degrees=0,
    height_array=None,
    albedo_array=None,
):
    """Approximates the change in temperature due to the change in lamppost flux
    assuming the irradiated disk model, following the Taylor expansion.

    delta_t / delta_lx ~ geometric_disk_factor / (4 * disk_temp**3)

    As such, this primarily uses calculate_geometric_disk_factor() and weights it by the
    temperature at each coordinate.

    :param temp_array: a 2d array representing the effective temperature of the
        accretion disk
    :param radii_array: a 2d array representing the radii from the smbh in gravitational
        radii
    :param phi_array: a 2d array representing the azimuths on the accretion disk in
        radians
    :param smbh_mass_exp: the solution of log10(m_smbh / m_sun)
    :param corona_height: the lamppost height in gravitational radii
    :param axis_offset_in_gravitational_radii: the cylindrical offset of the lamppost in
        gravitational radii with respect to the black hole
    :param angle_offset_in_degrees: the azimuth of the offset of the lamppost in degrees
    :param height_array: array of heights to calculate the disk at. Allows for greater
        flexability in disk model. Note, this is experimental!
    :param albedo_array: int, float, or array of albedos (reflectivities) to use for the
        disk
    :return: a 2d array representing the change in effective temperature with respect to
        the luminosity of the x-ray source
    """
    geometric_weighting_array = calculate_geometric_disk_factor(
        temp_array,
        radii_array,
        phi_array,
        smbh_mass_exp,
        corona_height,
        axis_offset_in_gravitational_radii=axis_offset_in_gravitational_radii,
        angle_offset_in_degrees=angle_offset_in_degrees,
        height_array=height_array,
        albedo_array=albedo_array,
    )

    mask = temp_array > 0
    geometric_weighting_array = np.nan_to_num(mask * geometric_weighting_array.value)

    return np.nan_to_num(geometric_weighting_array / (4 * temp_array**3))


def construct_accretion_disk_transfer_function(
    rest_wavelength_in_nm,
    temp_array,
    radii_array,
    phi_array,
    g_array,
    inclination_angle,
    smbh_mass_exp,
    corona_height,
    axis_offset_in_gravitational_radii=0,
    angle_offset_in_degrees=0,
    height_array=None,
    albedo_array=None,
    return_response_array_and_lags=False,
):
    """This calculates the accretion disk's transfer function in the lamppost geometry
    for some given effective temperature mapping. Does not rely on a particular
    temperature profile or disk geometry, but it does assume black-body radiation.

    :param rest_wavelength_in_nm: rest wavelength to calculate the transfer function at,
        in nm
    :param temp_array: a 2d array of effective temperatures of the accretion disk
    :param radii_array: a 2d array of radii across the accretion disk in gravitational
        radii
    :param phi_array: a 2d array of azimuths on the accretion disk in radians
    :param g_array: a 2d array representing the relativistic redshifts on the accretion
        disk
    :param inclination_angle: the inclination of the accretion disk w.r.t. to the
        observer, in degrees
    :param smbh_mass_exp: the solution to log10(m_smbh / m_sun)
    :param corona_height: height of the lamppost in gravitational radii
    :param axis_offset_in_gravitational_radii: the cylindrical radial distance of the
        source from the smbh
    :param angle_offset_in_degrees: azimuth angle in degrees of the offset lamppost.
        Note that 0 degrees is nearest to the observer and 180 degrees is furthest away.
    :param height_array: an optional 2d array of the height values in gravitational
        radii
    :param albedo_array: float, int, or array of albedo (reflectivity) values to use
    :param return_response_array_and_lags: boolean toggle to return the response map and
        time lags instead of the transfer function
    :return: a normalized 1d representation of the transfer function of the accretion
        disk with time lags represented in units R_g / c.
    """
    time_lag_array = calculate_time_lag_array(
        radii_array,
        phi_array,
        inclination_angle,
        corona_height,
        axis_offset_in_gravitational_radii=axis_offset_in_gravitational_radii,
        angle_offset_in_degrees=angle_offset_in_degrees,
        height_array=height_array,
    )

    db_dt_array = planck_law_derivative(
        temp_array,
        rest_wavelength_in_nm,
    )

    dt_dlx_array = calculate_dt_dlx(
        temp_array,
        radii_array,
        phi_array,
        smbh_mass_exp,
        corona_height,
        axis_offset_in_gravitational_radii=axis_offset_in_gravitational_radii,
        angle_offset_in_degrees=angle_offset_in_degrees,
        height_array=height_array,
        albedo_array=albedo_array,
    )

    response_factors = db_dt_array * dt_dlx_array * g_array**4

    if return_response_array_and_lags:
        return response_factors, time_lag_array

    transfer_function = np.histogram(
        rescale(time_lag_array, 10),
        range=(0, np.max(time_lag_array) + 1),
        bins=int(np.max(time_lag_array) + 1),
        weights=np.nan_to_num(rescale(response_factors, 10)),
        density=True,
    )[0]

    return transfer_function / np.sum(transfer_function)


def calculate_microlensed_transfer_function(
    magnification_array,
    redshift_lens,
    redshift_source,
    rest_wavelength_in_nm,
    temp_array,
    radii_array,
    phi_array,
    g_array,
    inclination_angle,
    smbh_mass_exp,
    corona_height,
    mean_microlens_mass_in_kg=1.0 * const.M_sun.to(u.kg),
    number_of_microlens_einstein_radii=25,
    number_of_smbh_gravitational_radii=1000,
    relative_orientation=0,
    OmM=0.3,
    H0=70,
    axis_offset_in_gravitational_radii=0,
    angle_offset_in_degrees=0,
    height_array=None,
    albedo_array=None,
    x_position=None,
    y_position=None,
    return_response_array_and_lags=False,
    return_descaled_response_array_and_lags=False,
    return_magnification_map_crop=False,
    random_seed=None,
):
    """Calculate the transfer function assuming the response of the disk can be
    amplified by microlensing. Essentially this is done by calculating the response and
    time lag maps of the accretion disk, determining the scale ratio between sizes in
    the source plane, rescaling the accretion disk's arrays to the resolution of the
    magnification map, weighting each pixel by its corresponding magnification, then
    computing the transfer function.

    ----- microlensing params -----
    :param magnification_array: a 2d array of magnifications in the source plane
    :param redshift_lens: int/float representing the redshift of the lens
    :param redshift_source: int/float representing the redshift of the source
    :param mean_microlens_mass_in_kg: average mass of the microlensing objects in
        kg. Typical values range from 0.1 to 1.0 M_sun.
    :param number_of_microlens_einstein_radii: number of R_e the magnification map
        covers along one edge.
    :param relative_orientation: orientation of the accretion disk w.r.t. the
        magnification map
    :param OmM: mass contribution to the energy budget of the universe
    :param H0: Hubble constant in units km/s/Mpc
    :param x_position: an optional x coordinate location to use on the magnification
        map. Otherwise, will be chosen randomly
    :param y_position: an optional y coordinate location to use on the magnification
        map. Otherwise, will be chosen randomly
    :param return_response_array_and_lags: boolean toggle to return a representation of the
        amplified response and time lags before the caluclation of the transfer function.
        Also returns x and y positions of where the microlensing was assumed to take place.
    :param return_descaled_response_array_and_lags: boolean toggle to return a representation
        of the amplified response and time lags at the resolution of the magnification map.
        Also returns x and y positions of where the microlensing was assumed to take place.
    :param return_magnification_map_crop: boolean toggle to return the section of the
        magnification map which amplifies the response function.
    :param random_seed: random seed to use for reproducibility

    ----- accretion disk params ------
    :param rest_wavelength_in_nm: rest frame wavelength in nanometers to calculate the
        transfer function at
    :param temp_array: a 2d array representing the effective temperatures of the
        accretion disk
    :param radii_array: a 2d array representing the radii of each pixel in the source
        plane with units of gravitational radii
    :param phi_array: a 2d array representing the azimuths of each pixel in the source
        plane in radians
    :param g_array: a 2d array representing the redshift factors due to relativistic
        effects.
    :param inclination_angle: inclination of the accretion disk w.r.t. the observer in
        degrees
    :param smbh_mass_exp: the solution of log10(m_smbh / m_sun)
    :param corona_height: height of the lamppost in gravitational radii
    :param number_of_smbh_gravitational_radii: maximum radius of the accretion disk in R_g
    :param axis_offset_in_gravitational_radii: the cylindrical radial offset of the
        irradiation source in gravitational radii
    :param angle_offset_in_degrees: the azimuth of the offset of the lamppost in degrees
    :param height_array: array of heights to calculate the disk at. Allows for greater
        flexability in disk model. Note that this is experimental!
    :param albedo_array: int, float, or array of albedos (reflectivities) to use for the
        disk

    :return: transfer function calculated assuming the response of the disk is amplified
        by the magnification_array
    """
    rng = np.random.default_rng(seed=random_seed)

    assert redshift_lens != redshift_source

    disk_response_array, time_lag_array = construct_accretion_disk_transfer_function(
        rest_wavelength_in_nm,
        temp_array,
        radii_array,
        phi_array,
        g_array,
        inclination_angle,
        smbh_mass_exp,
        corona_height,
        axis_offset_in_gravitational_radii=axis_offset_in_gravitational_radii,
        angle_offset_in_degrees=angle_offset_in_degrees,
        height_array=height_array,
        albedo_array=albedo_array,
        return_response_array_and_lags=True,
    )

    rescaled_response_array = perform_microlensing_convolution(
        magnification_array,
        disk_response_array,
        redshift_lens,
        redshift_source,
        smbh_mass_exp=smbh_mass_exp,
        mean_microlens_mass_in_kg=mean_microlens_mass_in_kg,
        number_of_microlens_einstein_radii=number_of_microlens_einstein_radii,
        number_of_smbh_gravitational_radii=number_of_smbh_gravitational_radii,
        relative_orientation=relative_orientation,
        OmM=OmM,
        H0=H0,
        return_preconvolution_information=True,
    )

    scale_ratio = np.size(rescaled_response_array, 0) / np.size(disk_response_array, 0)

    rescaled_time_lag_array = rescale(time_lag_array, scale_ratio)
    assert np.shape(rescaled_time_lag_array) == np.shape(rescaled_response_array)

    pixel_shift = np.size(rescaled_time_lag_array, 0) // 2

    magnification_array_padded = np.pad(
        magnification_array, pixel_shift, mode='edge'
    )

    if x_position is None:
        x_position = int(
            rng.random() * (
                np.size(magnification_array, 0) - np.size(rescaled_response_array, 0)
            ) + pixel_shift
        )
        
    if y_position is None:
        y_position = int(
            rng.random() * (
                np.size(magnification_array, 1) - np.size(rescaled_response_array, 1)
            ) + pixel_shift
        )

    magnification_crop = magnification_array_padded[
        x_position :
        x_position + np.size(rescaled_response_array, 0),
        y_position :
        y_position + np.size(rescaled_response_array, 1),
    ]

    if return_magnification_map_crop:
        return magnification_crop

    magnified_response_array = rescaled_response_array * magnification_crop

    if return_response_array_and_lags:
        return magnified_response_array, rescaled_time_lag_array, x_position, y_position

    unscaled_magnified_response_array = rescale(
        magnified_response_array, 1 / scale_ratio
    )

    descaling_factor = np.sum(rescaled_response_array) / np.sum(
        unscaled_magnified_response_array
    )
    unscaled_magnified_response_array *= descaling_factor

    unscaled_magnified_response_array *= np.sum(magnified_response_array) / np.sum(
        unscaled_magnified_response_array
    )
    unscaled_time_lag_array = rescale(rescaled_time_lag_array, 1 / scale_ratio)

    if return_descaled_response_array_and_lags:
        return (
            unscaled_magnified_response_array,
            unscaled_time_lag_array,
            x_position,
            y_position,
        )

    microlensed_transfer_function = np.histogram(
        rescale(rescaled_time_lag_array, 10),
        range=(0, np.max(rescaled_time_lag_array) + 1),
        bins=int(np.max(rescaled_time_lag_array) + 1),
        weights=np.nan_to_num(rescale(magnified_response_array, 10)),
        density=True,
    )[0]

    return np.nan_to_num(
        microlensed_transfer_function / np.sum(microlensed_transfer_function)
    )


def generate_drw_signal(
    length_of_light_curve, time_step, sf_infinity, tau_drw, random_seed=None
):
    """Generate a damped random walk using typical parameters as defined in Kelly+ 2009.
    Uses recursion, so this is not as fast as generating directly from the psd.

    :param length_of_light_curve: the length of the light curve in arbitrary units
    :param time_step: the spacing of the light curve, in identical units to maximum_time
    :param sf_infinity: the asymptotic structure function of the damped random walk
    :param tau_drw: the characteristic time scale of the variability in units equivalent
        to length_of_light_curve and time_step
    :param random_seed: random seed to use for reproducibility
    :return: an array representing the damped random walk
    """
    rng = np.random.default_rng(seed=random_seed)

    number_of_points = 2 * int(length_of_light_curve / time_step) + 1

    output_drw = np.zeros(number_of_points)

    for point in range(number_of_points - 1):
        output_drw[point + 1] = output_drw[point] * np.exp(
            -abs(time_step / tau_drw)
        ) + (sf_infinity / np.sqrt(1 / 2)) * rng.random() * (
            1 - (np.exp(-2 * abs(time_step / tau_drw)))
        ) ** (
            1 / 2
        )

    output_drw = output_drw[int(number_of_points // 2) :]

    output_drw -= np.mean(output_drw)
    output_drw /= np.std(output_drw)

    return output_drw


def generate_signal_from_psd(
    length_of_light_curve,
    power_spectrum,
    frequencies,
    random_seed=None,
):
    """Generate a signal from any power spectrum using the methods of Timmer+.

    length_of_light_curve and frequencies must be recipricol units. the output light
    curve will be normalized to have mean 0, standard deviation 1. Thanks @ Joshua Fagin
    for assistance on writing this function. Thanks at James H.H. Chan and the rest of
    the FutureLens group for discussions.

    :param length_of_light_curve: maximum length of the light curve to generate. Note that this maximum
        value is dependent on the input frequencies, since the frequencies can only generate a light
        curve ranging from values between the Nyquist frequency [1/(2 * max(frequency))] and 1/min(frequency)
    :param power_spectrum: array representing the input power spectrum of the stochastic signal at each
        fourier frequency defined in the frequencies parameter.
    :param frequencies: the input fourier frequencies associated with the power spectrum. Note these should be
        defined in linear space as:
        np.linspace(1/length_of_light_curve, 1/(2 * desired_time_resolution), int(length_of_light_curve)+1)
    :param random_seed: random seed to use for reproducibility
    :return: signal generated from the power spectrum ith length defined by length_of_light_curve.
    """
    rng = np.random.default_rng(seed=random_seed)

    observations_per_day = 2 * np.max(frequencies)

    random_phases = 2 * np.pi * rng.random(size=len(frequencies))

    positive_fourier_plus_phases = np.sqrt(power_spectrum) * np.exp(1j * random_phases)

    fourier_transform_of_output = np.concatenate(
        (
            positive_fourier_plus_phases,
            positive_fourier_plus_phases[-2:0:-1].conjugate(),
        )
    )

    light_curve = np.fft.ifft(fourier_transform_of_output)[
        : int(length_of_light_curve * observations_per_day)
    ]

    light_curve -= np.mean(light_curve)

    time_axis = np.linspace(0, length_of_light_curve - 1, len(light_curve))

    if np.std(light_curve) > 0:
        light_curve /= np.std(light_curve)

    return time_axis, light_curve.real


def generate_snapshots_of_radiation_pattern(
    rest_wavelength_in_nm,
    time_stamps,
    temp_array,
    radii_array,
    phi_array,
    g_array,
    smbh_mass_exp,
    driving_signal,
    driving_signal_fractional_strength,
    corona_height,
    inclination_angle,
    axis_offset_in_gravitational_radii=0,
    angle_offset_in_degrees=0,
    height_array=None,
    albedo_array=None,
):
    """Generate the radiation pattern at particular time steps labeled in time_stamps.

    :param rest_wavelength_in_nm: rest frame wavelength in nm
    :param time_stamps: list of dates to extract the radiation pattern at, in days
    :param temp_array: a 2d array of the effective temperatures of the accretion disk in
        Kelvins
    :param radii_array: a 2d array of radii of the accretion disk, in gravitational
        radii
    :param phi_array: a 2d array of azimuth values on the accretion disk, in radians
    :param g_array: a 2d array representing the relativistic redshift factor
    :param smbh_mass_exp: the solution to log10(m_smbh / m_sun)
    :param driving_signal: a list representing the underlying driving signal which
        produces the radiation pattern on the accretion disk. Must be evenly spaced in
        days.
    :param driving_signal_fractional_strength: relative strength of the total flux due
        to the reprocessing, on a scale of (0, 1). 0 represents no contribution while 1
        represnts no static flux contribution.
    :param corona_height: height of the lamppost corona in gravitational radii
    :param inclination_angle: the inclination of the accretion disk w.r.t. to the
        observer, in degrees
    :param axis_offset_in_gravitational_radii: the cylindrical radial offset of the
        lamppost in gravitational radii
    :param angle_offset_in_degrees: the azimuth of the offset of the lamppost in degrees
    :param height_array: array of heights to calculate the disk at. Note that this is an
        experimental feature!
    :param albedo_array: int, float, or array of albedos (reflectivities) to use for the
        disk
    :return: a series of 2d arrays representing the radiation pattern at each value of
        time_stamps
    """
    assert driving_signal_fractional_strength >= 0
    assert driving_signal_fractional_strength <= 1
    static_flux = planck_law(temp_array, rest_wavelength_in_nm) * g_array**4

    total_static_flux = np.sum(static_flux)

    response_array, time_lag_array = construct_accretion_disk_transfer_function(
        rest_wavelength_in_nm,
        temp_array,
        radii_array,
        phi_array,
        g_array,
        inclination_angle,
        smbh_mass_exp,
        corona_height,
        axis_offset_in_gravitational_radii=axis_offset_in_gravitational_radii,
        angle_offset_in_degrees=angle_offset_in_degrees,
        height_array=height_array,
        albedo_array=albedo_array,
        return_response_array_and_lags=True,
    )

    gravitational_radius = calculate_gravitational_radius(10**smbh_mass_exp)
    gr_per_day = gravitational_radius / const.c.to(u.m / u.day).value

    time_lag_array *= gr_per_day
    maximum_time_lag_in_days = np.max(time_lag_array)

    response_array *= total_static_flux / np.sum(response_array)

    if len(driving_signal) < np.max(time_stamps + maximum_time_lag_in_days):
        print(
            "warning, driving signal is not long enough to support all snapshots. looping signal"
        )
        while len(driving_signal) < np.max(time_stamps + maximum_time_lag_in_days):
            driving_signal = np.concatenate((driving_signal, driving_signal))

    burn_in_time = maximum_time_lag_in_days
    accretion_disk_mask = temp_array > 1000

    list_of_snapshots = []
    for time in time_stamps:
        array_of_time_stamps = (
            int(burn_in_time) + int(time) - time_lag_array.astype(int)
        )
        list_of_snapshots.append(
            (1 - driving_signal_fractional_strength) * static_flux * accretion_disk_mask
            + driving_signal_fractional_strength
            * np.take(driving_signal, array_of_time_stamps)
            * response_array
            * accretion_disk_mask
        )
    return list_of_snapshots


def project_blr_to_source_plane(
    blr_density_rz_grid,
    blr_vertical_velocity_grid,
    blr_radial_velocity_grid,
    inclination_angle,
    smbh_mass_exp,
    velocity_range=[-1, 1],
    weighting_grid=None,
    radial_resolution=1,
    vertical_resolution=1,
):
    """Takes an axi-symmetric grid of density values and weighting grid with (R, Z)
    coordinates and projects it to the source plane. It also can select particular
    velocity ranges to isolate projections into particular filters.

    :param blr_density_rz_grid: a 2d array of values representing the density of the BLR
        at each point in (R, Z) coords.
    :param blr_vertical_velocity_grid: a 2d array of v_{z} values, normalized by the
        speed of light.
    :param blr_radial_velocity_grid: a 2d array of v_{r} values, normalized by the speed
        of light
    :param inclination_angle: the inclination of the agn w.r.t. the observer in degrees.
    :param smbh_mass_exp: the solution of log10(m_smbh/m_sun)
    :param velocity_range: the range of line-of-sight velocities which are accepted, in
        units of speed of light. We take the convention of positive values are aimed
        towards the observer. Note that the BLR object will compute this for a given
        speclite filter.
    :param weighting_grid: a 2d array of values which correspond to weighting factors in
        the blr_density_rz_grid to represent additional weighting by the local optimally
        emitting cloud model
    :param radial_resolution: the spacing between radial coordinates in gravitational
        radii
    :param vertical_resolution: the spacing between vertical coordinates in
        gravitational radii
    :return: a 2d array representing the projected BLR in the source plane with pixel
        resolution equal to radial_resolution and the maximum radius required for
        creating a FluxProjection object.
    """
    assert inclination_angle >= 0
    assert inclination_angle < 90
    assert np.shape(blr_density_rz_grid) == np.shape(blr_vertical_velocity_grid)
    assert np.shape(blr_vertical_velocity_grid) == np.shape(blr_radial_velocity_grid)
    inclination_angle *= np.pi / 180

    if weighting_grid is None:
        weighting_grid = np.ones(np.shape(blr_density_rz_grid))
    assert np.shape(weighting_grid) == np.shape(blr_density_rz_grid)

    gravitational_radius = calculate_gravitational_radius(10**smbh_mass_exp)

    source_plane_resolution = radial_resolution

    max_r = np.size(blr_density_rz_grid, 0) * radial_resolution
    max_z = np.size(blr_density_rz_grid, 1) * vertical_resolution

    max_projected_size_in_source_plane = np.max(
        [max_z * np.sin(inclination_angle) + max_r * np.cos(inclination_angle), max_r]
    )

    new_max_r_required = int(
        2 * max_projected_size_in_source_plane / source_plane_resolution
    )

    source_plane_projection = np.zeros((new_max_r_required, new_max_r_required))

    for height in range(np.size(blr_density_rz_grid, 1)):
        current_y_offset = height * vertical_resolution * np.tan(inclination_angle)

        x_coordinates = np.linspace(
            -max_projected_size_in_source_plane,
            max_projected_size_in_source_plane,
            int(2 * max_projected_size_in_source_plane / source_plane_resolution),
        )
        y_coordinates = np.linspace(
            -max_projected_size_in_source_plane / np.cos(inclination_angle)
            - current_y_offset,
            max_projected_size_in_source_plane / np.cos(inclination_angle)
            - current_y_offset,
            int(2 * max_projected_size_in_source_plane / source_plane_resolution),
        )
        X, Y = np.meshgrid(x_coordinates, y_coordinates)
        R, Phi = convert_cartesian_to_polar(
            X,
            Y,
        )

        index_grid = R // radial_resolution

        keplerian_velocities = calculate_keplerian_velocity(
            index_grid * radial_resolution * gravitational_radius,
            10**smbh_mass_exp,
        )

        index_mask = np.logical_and(
            (index_grid < np.size(blr_density_rz_grid, 0)), (index_grid > 0)
        )

        index_grid *= index_mask

        if np.sum(index_grid) == 0:
            continue

        line_of_sight_velocities = (
            np.cos(inclination_angle)
            * blr_vertical_velocity_grid[index_grid.astype(int), height]
            + np.sin(inclination_angle)
            * np.cos(Phi)
            * blr_radial_velocity_grid[index_grid.astype(int), height]
            - np.sin(inclination_angle)
            * np.sin(Phi)
            * keplerian_velocities[index_grid.astype(int), height]
        )

        velocity_selected_mask = np.logical_and(
            (line_of_sight_velocities >= velocity_range[0]),
            (line_of_sight_velocities < velocity_range[1]),
        )

        current_density = (
            blr_density_rz_grid[index_grid.astype(int), height]
            * velocity_selected_mask
            * index_mask
            * weighting_grid[index_grid.astype(int), height]
        )

        source_plane_projection += current_density

    return source_plane_projection, new_max_r_required


def calculate_blr_transfer_function(
    blr_density_rz_grid,
    blr_vertical_velocity_grid,
    blr_radial_velocity_grid,
    inclination_angle,
    smbh_mass_exp,
    velocity_range=[-1, 1],
    weighting_grid=None,
    radial_resolution=1,
    vertical_resolution=1,
):
    """Calculate the response function of the BLR by assuming weighting factors for some
    given wavelength range. The BLR emission is assumed to be proportional to the
    particle density and the weighting factor.

    Todo: this is a very slow function. If there's a way to project the BLR into the
    3-dimensional cylindrical grid faster then compute the time lags as a function of
    (R, Z, phi) and take a single histogram over the whole space, that would probably
    speed it up significantly. Figure out how to do this sometime!

    :param blr_density_rz_grid: a 2d array of values representing the density of the blr
        at each point in (R, Z) coords.
    :param blr_vertical_velocity_grid: a 2d array of v_{z} values, normalized by the
        speed of light.
    :param blr_radial_velocity_grid: a 2d array of v_{r} values, normalized by the speed
        of light
    :param inclination_angle: the inclination of the agn w.r.t. the observer in degrees.
    :param smbh_mass_exp: the solution of log_{10} (M_{bh} / M_{sun})
    :param velocity_range: the range of line-of-sight velocities which are accepted, in
        units of speed of light. We take the convention of positive values are aimed
        towards the observer, and are therefore blueshifted.
    :param weighting_grid: a 2d array of values which correspond to weighting factors in
        the blr_density_rz_grid
    :param radial_resolution: the spacing between radial coordinates in gravitational
        radii
    :param vertical_resolution: the spacing between vertical coordinates in
        gravitational radii
    :return: a 1d array representing the normalized response function of the BLR w.r.t.
        the optical accretion disk approximated as emitting from the SMBH in units of
        gravitational radii
    """
    assert inclination_angle >= 0
    assert inclination_angle < 90
    assert np.shape(blr_density_rz_grid) == np.shape(blr_vertical_velocity_grid)
    assert np.shape(blr_vertical_velocity_grid) == np.shape(blr_radial_velocity_grid)
    inclination_angle *= np.pi / 180

    if weighting_grid is None:
        weighting_grid = np.ones(np.shape(blr_density_rz_grid))
    assert np.shape(weighting_grid) == np.shape(blr_density_rz_grid)

    x_coordinates = np.linspace(
        -np.size(blr_density_rz_grid, 0) * radial_resolution,
        np.size(blr_density_rz_grid, 0) * radial_resolution,
        int(2 * np.size(blr_density_rz_grid, 0)),
    )

    y_coordinates = np.linspace(
        -np.size(blr_density_rz_grid, 0)
        * radial_resolution
        / np.cos(inclination_angle),
        np.size(blr_density_rz_grid, 0) * radial_resolution / np.cos(inclination_angle),
        int(2 * np.size(blr_density_rz_grid, 0)),
    )

    X, Y = np.meshgrid(x_coordinates, y_coordinates)
    R, Phi = convert_cartesian_to_polar(
        X,
        Y,
    )

    index_grid = R // radial_resolution

    index_mask = np.logical_and(
        (index_grid < np.size(blr_density_rz_grid, 0)), (index_grid > 0)
    )

    gravitational_radius = calculate_gravitational_radius(10**smbh_mass_exp)

    keplerian_velocities = calculate_keplerian_velocity(
        index_grid * radial_resolution * gravitational_radius, 10**smbh_mass_exp
    )

    index_grid *= index_mask

    h_0_time_delays = calculate_time_lag_array(
        R,
        Phi,
        inclination_angle * 180 / np.pi,
        0,
    )

    h_z_time_delays = calculate_time_lag_array(
        R,
        Phi,
        inclination_angle * 180 / np.pi,
        0,
        height_array=np.ones(np.shape(R * radial_resolution))
        * np.size(blr_density_rz_grid, 1)
        * vertical_resolution,
    )

    output_transfer_function = np.zeros(
        int(np.max((np.max(h_0_time_delays), np.max(h_z_time_delays))) + 1)
    )

    for height in range(np.size(blr_density_rz_grid, 1)):

        line_of_sight_velocities = (
            np.cos(inclination_angle)
            * blr_vertical_velocity_grid[index_grid.astype(int), height]
            + np.sin(inclination_angle)
            * np.cos(Phi)
            * blr_radial_velocity_grid[index_grid.astype(int), height]
            - np.sin(inclination_angle)
            * np.sin(Phi)
            * keplerian_velocities[index_grid.astype(int), height]
        )

        velocity_selected_mask = np.logical_and(
            (line_of_sight_velocities >= velocity_range[0]),
            (line_of_sight_velocities < velocity_range[1]),
        )

        response_of_current_slab = (
            blr_density_rz_grid[index_grid.astype(int), height]
            * velocity_selected_mask
            * index_mask
            * weighting_grid[index_grid.astype(int), height]
        )

        time_delays_of_current_slab = calculate_time_lag_array(
            R,
            Phi,
            inclination_angle * 180 / np.pi,
            0,
            height_array=np.ones(np.shape(R * radial_resolution))
            * height
            * vertical_resolution,
        )

        transfer_function_of_slab = np.histogram(
            time_delays_of_current_slab,
            range=(0, np.max(time_delays_of_current_slab) + 1),
            bins=int((np.max(time_delays_of_current_slab) + 1) / radial_resolution),
            weights=np.nan_to_num(response_of_current_slab),
            density=True,
        )[0]

        rescaled_transfer_function_as_values = np.repeat(
            transfer_function_of_slab, int(radial_resolution)
        )

        output_transfer_function[
            : np.size(rescaled_transfer_function_as_values)
        ] += np.nan_to_num(rescaled_transfer_function_as_values)

    if np.sum(output_transfer_function) > 0:
        output_transfer_function /= np.sum(output_transfer_function)

    return output_transfer_function


def determine_emission_line_velocities(
    rest_frame_emitted_wavelength,
    minimum_admitted_wavelength,
    maximum_admitted_wavelength,
    redshift,
):
    """Helper function to define the velocity range of an emission line due to Doppler
    broadening which is required to shift the emission line into the desired filter.

    :param rest_frame_emitted_wavelength: emission line wavelength in rest frame. Units
        must match passband units, but may be arbitrary.
    :param minimum_admitted_wavelength: minimum wavelength of the passband (filter).
        Units may be arbitrary, but must match rest_frame_emitted_wavelength.
    :param maximum_admitted_wavelength: maximum wavelength of the passband (filter).
        Units may be arbitrary, but must match rest_frame_emitted_wavelength.
    :param redshift: cosmological redshift factor
    :return: list of [minimum, maximum] velocities which will shift the emission line
        into the desired filter at redshift.
    """
    assert redshift >= 0
    rest_frame_minimum_admitted_wavelength = minimum_admitted_wavelength / (
        1 + redshift
    )
    rest_frame_maximum_admitted_wavelength = maximum_admitted_wavelength / (
        1 + redshift
    )

    required_velocity_minimum = (
        1
        - (rest_frame_maximum_admitted_wavelength / rest_frame_emitted_wavelength) ** 2
    ) / (
        (rest_frame_maximum_admitted_wavelength / rest_frame_emitted_wavelength) ** 2
        + 1
    )

    required_velocity_maximum = (
        1
        - (rest_frame_minimum_admitted_wavelength / rest_frame_emitted_wavelength) ** 2
    ) / (
        (rest_frame_minimum_admitted_wavelength / rest_frame_emitted_wavelength) ** 2
        + 1
    )

    return [required_velocity_minimum, required_velocity_maximum]


def convolve_signal_with_transfer_function(
    smbh_mass_exp=None,
    driving_signal=None,
    initial_time_axis=None,
    transfer_function=None,
    redshift_source=0,
    desired_cadence_in_days=1,
):
    """Helper function to convolve a signal with a transfer function which has spacing
    in gravitational radii. If the initial time axis is not given, the function assumes
    that the spacing in the signal is in days in the source frame. Hypersampling is used
    to enforce that we have the desired observational cadence after redshifting.

    :param smbh_mass_exp: solution to log10(m_smbh/m_sun).
    :param driving_signal: driving signal to convolve with the transfer function.
        Assumed to be in units of days unless the initial time axis is given.
    :param initial_time_axis: time axis of the driving signal in units days, but may
        have interday values.
    :param transfer_function: transfer function which represents the response of an AGN
        component to an impulse, assumed to be normalized and with spacing of
        gravitational radii.
    :param redshift_source: redshift of the system
    :param desired_cadence_in_days: desired sampling of the output signal in units days.
        This will be enforced via linear interpolation.
    :return: array representing the time axis and an array representing the reprocessed
        signal
    """
    assert redshift_source >= 0
    gravitational_radius = calculate_gravitational_radius(10**smbh_mass_exp)

    light_travel_time_for_grav_rad = (
        gravitational_radius / const.c.to(u.m / u.day).value
    )

    required_hyper_resolution = (1 + redshift_source) / min(desired_cadence_in_days, 1)

    if initial_time_axis is None:
        initial_time_axis = np.linspace(0, len(driving_signal) - 1, len(driving_signal))

    driving_signal_interpolation = interp1d(
        initial_time_axis, driving_signal, bounds_error=False, fill_value="extrapolate"
    )

    desired_time_axis = np.linspace(
        0,
        max(initial_time_axis),
        int(max(initial_time_axis) * required_hyper_resolution),
    )

    tau_axis = np.linspace(
        0,
        (len(transfer_function) - 1) * light_travel_time_for_grav_rad,
        len(transfer_function),
    )

    interpolated_transfer_function = interp1d(
        tau_axis, transfer_function, bounds_error=False, fill_value="extrapolate"
    )

    desired_tau_axis = np.linspace(
        0,
        (len(transfer_function) - 1) * light_travel_time_for_grav_rad,
        int(
            (len(transfer_function) - 1)
            * light_travel_time_for_grav_rad
            * required_hyper_resolution
        ),
    )

    hypersampled_signal = driving_signal_interpolation(desired_time_axis)

    if len(desired_tau_axis) <= 1:
        print("warning: unresolvable transfer function")
        hypersample_times = np.linspace(
            0, max(initial_time_axis), len(hypersampled_signal)
        ) * (1 + redshift_source)
        return hypersample_times, hypersampled_signal

    hypersampled_transfer_function = interpolated_transfer_function(desired_tau_axis)

    convolution = convolve(hypersampled_signal, hypersampled_transfer_function)[
        : len(hypersampled_signal)
    ]
    hypersample_times = np.linspace(0, max(initial_time_axis), len(convolution)) * (
        1 + redshift_source
    )

    return hypersample_times, convolution


def convert_speclite_filter_to_wavelength_range(filter_string, min_threshold=0.01):
    """This function takes a speclite filter object or a string associated with speclite
    filters and outputs a wavelength range corresponding to the passband greater than
    minimum threshold.

    :param filter_string: speclite object or string associated with a speclite object.
    :param min_threshold: value to use in order to define the minimum and maximum
        wavelengths.
    :return: list of wavelength ranges in nm
    """

    if isinstance(filter_string, FilterResponse):
        working_filters = [filter_string]
    elif isinstance(filter_string, str):
        try:
            working_filters = [load_filter(filter_string)]
        except:
            print("no found speclite filter")
            return False
    elif isinstance(filter_string, list):
        try:
            working_filters = load_filters(filter_string)
        except:
            working_filters = []
            for item in filter_string:
                try:
                    current_filter = load_filter(item)
                    working_filters.append(current_filter)
                except:
                    continue
            if len(working_filters) == 0:
                print("no found speclite filters")
                return False
    else:
        print("incompatible filter string")
        return False

    output_wavelength_ranges = []
    for current_filter in working_filters:
        current_wavelengths = current_filter.wavelength.copy()
        unit_conversion = current_filter.effective_wavelength.unit.to(u.nm)
        current_wavelengths = unit_conversion * current_wavelengths
        passband = current_filter.response

        peak_response_arg = np.argmax(passband)
        min_wavelength_arg = np.argmin(
            abs(passband[:peak_response_arg] - min_threshold)
        )
        max_wavelength_arg = peak_response_arg + np.argmin(
            abs(passband[peak_response_arg:] - min_threshold)
        )
        output_wavelength_ranges.append(
            [
                current_wavelengths[min_wavelength_arg],
                current_wavelengths[max_wavelength_arg],
            ]
        )
    return output_wavelength_ranges
