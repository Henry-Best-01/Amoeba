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

warnings.filterwarnings("ignore")
np.seterr(divide="ignore", invalid="ignore")


def create_maps(
    mass_exp,
    redshift,
    number_grav_radii,
    inc_ang,
    resolution,
    spin=0,
    eddington_ratio=0.1,
    temp_beta=0,
    corona_height=6,
    albedo=1,
    eta=0.1,
    generic_beta=False,
    disk_acc=None,
    height_array=None,
    albedo_array=None,
    Om0=0.3,
    H0=70,
    efficiency=1.0,
    visc_temp_prof="SS",
    name="",
):
    """This function sets up maps required for the AccretionDisk class in Amoeba.

    :param mass_exp: the mass exponent of the smbh. mass_bh = 10**mass_exp * M_sun
    :param redshift: the redshift of the AGN
    :param number_grav_radii: the max radius of the accretion disk in gravitational
        radii
    :param inc_ang: the inclination of the accretion disk w.r.t. the observer, in
        degrees
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
        (see Sun et al, 2018). Note that using the parameter "generic_beta==True" will
        force r^-beta dependence instead.
    :param disk_acc: the amount of mass accreted by the accretion disk per time. If a
        number is given, units of solar_masses/year are assumed.
    :param corona_height: number of grav. radii above the accretion disk the assumed
        lamppost is
    :param albedo: reflectivity of disk. Setting to 0 will make the disk absorb
        emission, heating it up
    :param eta: lamppost source luminosity coefficient. Defined as Lx = eta * M_dot *
        c^2
    :param efficiency: efficiency of the conversion of gravitational potential energy to
        thermal energy.
    :return: a list representing 6 values (mass_exp, redshift, number_grav_radii,
        inc_ang, corona_height, spin) and 4 arrays (temp_array, r_array, g_array,
        phi_array) These are all recorded for conveninence, as they all get put into the
        AccretionDisk constructor in order.
    """
    try:
        import sim5

        sim5_installed = True
    except ModuleNotFoundError:
        sim5_installed = False

    assert redshift >= 0
    assert inc_ang >= 0
    assert inc_ang <= 90
    if inc_ang == 90:
        inc_ang -= 0.001
    assert abs(spin) <= 1
    assert temp_beta >= 0
    bh_mass_in_solar_masses = 10**mass_exp
    bh_mass_in_kg = bh_mass_in_solar_masses * const.M_sun.to(u.kg)
    grav_rad = calculate_gravitational_radius(bh_mass_in_solar_masses)
    temp_array = np.zeros((resolution, resolution))
    g_array = temp_array.copy()
    r_array = temp_array.copy()
    phi_array = temp_array.copy()
    if sim5_installed == True:
        if inc_ang == 0:
            inc_ang += 0.001
        bh_rms = sim5.r_ms(spin)
        for yy in range(resolution):
            for xx in range(resolution):
                # Note that Sim5 coordinates define Rg = 2GM/c^2, but we use Rg = GM/c^2
                alpha = ((xx + 0.5) / resolution - 0.5) * 4.0 * number_grav_radii
                beta = ((yy + 0.5) / resolution - 0.5) * 4.0 * number_grav_radii
                gd = sim5.geodesic()
                error = sim5.intp()
                sim5.geodesic_init_inf(
                    inc_ang * np.pi / 180, abs(spin), alpha, beta, gd, error
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
                    phi = sim5.geodesic_position_azm(gd, r, pol, P)
                    g_array[xx, yy] = sim5.gfactorK(r, abs(spin), gd.l)
                    phi_array[xx, yy] = phi
                    r_array[xx, yy] = r
    else:
        x_vals = np.linspace(-number_grav_radii, number_grav_radii, resolution)
        y_vals = x_vals.copy() / np.cos(np.pi * inc_ang / 180)
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
        eta_x_rays=eta,
        generic_beta=generic_beta,
        disk_acc=disk_acc,
        efficiency=efficiency,
        spin=spin,
        visc_temp_prof=visc_temp_prof,
    )
    disk_params = {
        "smbh_mass_exp": mass_exp,
        "redshift_source": redshift,
        "inclination_angle": inc_ang,
        "corona_height": corona_height,
        "temp_array": temp_array,
        "phi_array": phi_array,
        "g_array": g_array,
        "radii_array": r_array,
        "r_out_in_gravitational_radii": number_grav_radii,
        "height_array": height_array,
        "albedo_array": albedo_array,
        "spin": spin,
        "Om0": Om0,
        "H0": H0,
        "name": name,
    }

    return disk_params


def calculate_keplerian_velocity(radius_in_meters, mass_in_solar_masses):
    """Helper function to calculate the magnitude of Keplerian velocity of an orbit
    around a massive object :param radius_in_meters: radius in units meters or an
    astropy quantity :param mass_in_solar_masses: mass in units solar masses or an
    astropy quantity :return: keplerian velocity represented as a fraction of the speed
    of light."""
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

    :param spin: dimensionless spin of the SMBH on range (-1, 1)
    :return: ISCO size in units gravitational radii
    """
    if abs(spin) > 1:
        raise ValueError("Spin out of range. Must satisfy -1 <= spin <= 1.")
    z1 = 1 + (1 - spin**2) ** (1 / 3) * ((1 + spin) ** (1 / 3) + (1 - spin) ** (1 / 3))
    z2 = (3 * spin**2 + z1**2) ** (1 / 2)
    return 3 + z2 - np.sign(spin) * ((3 - z1) * (3 + z1 + 2 * z2)) ** (1 / 2)


def convert_eddington_ratio_to_accreted_mass(mass, eddington_ratio, efficiency=1.0):
    """This function converts an Eddington Ratio (i.e. 0.15) into the corresponding
    accretion rate in physical units assuming bol_lum = eddington_ratio * edd_lum.

    edd_lum = 4 pi G M M_proton c / (sigma_T) bol_lum = M_dot * c^2 * efficiency M_dot =
    edd_lum / (efficiency * c^2)

    :param mass: mass of SMBH in solar masses or astropy quantity
    :param eddington_ratio: percentage of theoretical Bondi limit of accretion rate
    :param efficiency: conversion efficiency between gravitational potential energy and
        thermal energy
    :return: accreted mass as astropy units
    """
    if type(mass) != u.Quantity:
        mass *= const.M_sun.to(u.kg)
    edd_lum = 4 * np.pi * const.G * mass * const.m_p * const.c / const.sigma_T
    bol_lum = edd_lum * eddington_ratio
    return bol_lum / (efficiency * const.c**2)


def accretion_disk_temperature(
    radius_in_meters,
    min_radius_in_meters,
    mass_in_solar_masses,
    eddington_ratio,
    beta=0,
    corona_height=6,
    albedo=1,
    eta_x_rays=0.1,
    generic_beta=False,
    disk_acc=None,
    efficiency=1.0,
    spin=0,
    visc_temp_prof="SS",
):
    """
    This function aims to take the viscous Thin disk and allows multiple additional modifications.
    Base Thin disk requires:
            :param radius_in_meters: radius or radii in meters
            :param min_radius_in_meters: inner radius in meters
            :param mass_in_solar_masses: mass of SMBH in solar masses
            :param eddington_ratio: percent of eddington limit the SMBH is accreting at
            :param disk_acc: = Override for accretion rate at inner radius in solar masses per year
    All other default inputs will return the temperature profile of the thin disk model.

    A wind effect which acts to remove accreting material and adjusts the slope (Sun+ 2018) may be modeled by:
            :param beta: wind strength providing the following accretion rate relationship
                m_dot = m0_dot * (r / r_in)^beta

    A corona heating effect due to lamppost geometry (Cacket+ 2007) may be modeled by:
            :param corona_height: lamppost height in gravitational radii.
                Default is 6, the Schwarzschild ISCO case.
            :param albedo: reflection coefficent of the accretion disk such that 0
                causes perfect absorption and 1 causing perfect reflection of X-ray energy.
                Default is 1, meaning no thermal contribution from the lamppost term.
            :param eta_x_rays: efficiency coefficient of lamppost source, defined as Lx = eta_x_rays * L_bol

    Some further arguments are included as convenience:
            :param generic_beta: bool for a thermal profile of the form r^(-beta).
            :param efficiency: efficiency of conversion of gravitational energy to thermal energy

    One additional argument is required for creating a Novikov-Thorne profile (Thanks: Josh Fagin)
            :param spin: dimensionless spin parameter of black hole

    Param to switch between thermal profiles
            :param visc_temp_prof: string defined by
                "SS" for Shakura-Sunyaev thin disk
                "NT" for Novikov-Thorne thin disk

    :return: temperature in Kelvins
    """
    if generic_beta == True:
        dummy = 3 - 4 * beta
        beta = dummy

    if type(radius_in_meters) == u.Quantity:
        radius_in_meters = radius_in_meters.to(u.m)
    if type(min_radius_in_meters) == u.Quantity:
        min_radius_in_meters = min_radius_in_meters.to(u.m)
    if disk_acc is None:
        disk_acc = convert_eddington_ratio_to_accreted_mass(
            mass_in_solar_masses, eddington_ratio, efficiency=efficiency
        )
    else:
        if type(disk_acc) == u.Quantity:
            disk_acc = disk_acc.to(u.kg / u.s)
        else:
            disk_acc *= const.M_sun.to(u.kg) / u.yr.to(
                u.s
            )  # Assumed was in M_sun / year
            disk_acc = disk_acc.value

    if type(mass_in_solar_masses) == u.Quantity:
        mass_in_kg = mass_in_solar_masses.to(u.kg).value
    else:
        mass_in_solar_masses *= const.M_sun.to(u.kg)  # Assumed was in M_sun
        mass_in_kg = mass_in_solar_masses.value
    grav_rad_in_meters = calculate_gravitational_radius(mass_in_solar_masses)
    schwarz_rad_in_meters = 2 * grav_rad_in_meters

    radius_in_grav_rad = radius_in_meters / grav_rad_in_meters

    inner_rad_in_grav_rad = min_radius_in_meters / grav_rad_in_meters

    # m
    m0_dot = disk_acc / (inner_rad_in_grav_rad**beta)
    corona_height += 0.5  # Avoid singularities
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
        x1 = 2 * np.cos(1.0 / 3.0 * np.arccos(spin) - np.pi / 3)
        x2 = 2 * np.cos(1.0 / 3.0 * np.arccos(spin) + np.pi / 3)
        x3 = -2 * np.cos(1.0 / 3.0 * np.arccos(spin))
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


def planck_law(temperature, rest_wavelength):
    """Calculates the spectral radiance of a black body. Keep in mind mks units makes
    this represents a very long wavelength range! This is not in [W m^-2 nm^-1] or [W
    m^-2 Hz].

    :param temperature: temperature in Kelvins
    :param rest_wavelength: rest frame wavelength in nanometers or astropy unit
    :return: Spectral radiance of a black body in [W/m^3].
    """

    if type(rest_wavelength) == u.Quantity:
        dummyval = rest_wavelength.to(u.m)
        rest_wavelength = dummyval.value
    elif type(rest_wavelength) != u.Quantity:
        dummyval = rest_wavelength * u.nm.to(u.m)
        rest_wavelength = dummyval

    return np.nan_to_num(
        2.0
        * const.h.value
        * const.c.value**2
        * (rest_wavelength) ** (-5.0)
        * (
            (
                np.e
                ** (
                    const.h.value
                    * const.c.value
                    / (rest_wavelength * const.k_B.value * temperature)
                )
                - 1.0
            )
            ** (-1.0)
        )
    )


def planck_law_derivative(temperature, rest_wavelength_in_nm):
    """Numerical approximation of the temperature derivative of the Planck law
    calculated through secant method of finding a derivative.

    :param temperature: temperature in Kelvins
    :param rest_wavelength_in_nm: rest frame wavelength in nanometers or astropy unit
    :return: dervative of the spectral radiance w.r.t. temperature, in units [W/m^3/K]
    """
    PlanckA = planck_law(temperature, rest_wavelength_in_nm)
    PlanckB = planck_law(temperature + 1, rest_wavelength_in_nm)
    return PlanckB - PlanckA


def calculate_gravitational_radius(mass_in_solar_masses):
    """Calculates the gravitational radius of a massive object following
    gravitational_radius = G m / c^2 :param mass_in_solar_masses: mass of the object in
    units of solar masses :return: length of one gravitational radius in meters."""
    if isinstance(mass_in_solar_masses, u.Quantity):
        mass_in_kg = mass_in_solar_masses.to(u.kg)
    else:
        mass_in_kg = mass_in_solar_masses * const.M_sun.to(u.kg)

    return (const.G * mass_in_kg / const.c**2).decompose().value


def calculate_angular_diameter_distance(redshift, Om0=0.3, little_h=0.7):
    """This funciton takes in a redshift value of z, and calculates the angular diameter
    distance. This is given as the output. This assumes LCDM model. Follows Distance
    measures in cosmology (Hogg 1999) :param redshift: redshift the object of interest
    is at :param Om0: total fraction of the Universe's energy budget is in mass.

    :param little_h: reduced Hubble constant defined by H_0 = little_h * 100 [km s^-1
        Mpc^-1]
    :return: angular diameter distance in units meters, assuming a flat lambda-CDM
        universe
    """
    OmL = 1 - Om0
    multiplier = (
        (9.26 * 10**25) * (little_h) ** (-1) * (1 / (1 + redshift))
    )  # This does not need to be integrated over
    integrand = lambda z_p: (Om0 * (1 + z_p) ** (3.0) + OmL) ** (-0.5)
    integral, err = quad(integrand, 0, redshift)
    value = multiplier * integral
    return value


def calculate_angular_diameter_distance_difference(
    redshift_lens, redshift_source, Om0=0.3, little_h=0.7
):
    """This function takes in 2 redshifts, designed to represent z1 = redshift (lens)
    and z2 = redshift (source). This assumes LCDM model. Follows Distance measures in
    cosmology (Hogg 1999)

    :param redshift_lens: redshift the gravitational lens
    :param redshift_source: redshift the source
    :param Om0: total fraction of the Universe's energy budget is in mass
    :param little_h: reduced Hubble constant defined by H_0 = little_h * 100 [km s^-1
        Mpc^-1]
    :return: angular diameter distance difference in units meters
    """
    if redshift_lens > redshift_source:
        # assume the redshifts were inserted in wrong order
        dummy_var = redshift_source
        redshift_source = redshift_lens
        redshift_lens = dummy_var

    OmL = 1 - Om0
    multiplier = (9.26 * 10**25) * (little_h) ** (-1) * (1 / (1 + redshift_source))
    integrand = lambda z_p: (Om0 * (1 + z_p) ** (3.0) + OmL) ** (-0.5)
    integral1, err1 = quad(integrand, 0, redshift_lens)
    integral2, err2 = quad(integrand, 0, redshift_source)
    return multiplier * (integral2 - integral1)


def calculate_luminosity_distance(redshift, Om0=0.3, little_h=0.7):
    """This calculates the luminosity distance using the
    calculate_angular_diameter_distance formula for flat lambda-CDM model. Follows
    Distance measures in cosmology (Hogg 1999). :param redshift: redshift of the object
    :param Om0: total fraction of the Universe's energy budget is in mass.

    :param little_h: reduced Hubble constant defined by H_0 = little_h * 100 [km s^-1
        Mpc^-1]
    :return: luminosity distance of the object
    """
    return (1 + redshift) ** 2 * calculate_angular_diameter_distance(
        redshift, Om0=Om0, little_h=little_h
    )


def calculate_angular_einstein_radius(
    redshift_lens,
    redshift_source,
    mean_microlens_mass_in_kg=1 * const.M_sun.to(u.kg),
    Om0=0.3,
    little_h=0.7,
):
    """This function calculates the Einstein radius of the microlens in radians.

    This assumes LCDM model.
    :param redshift_lens: redshift of the lensing galaxy
    :param redshift_source: redshift of the source
    :param mean_microlens_mass_in_kg: average mass of microlenses in the lensing galaxy.
        This is typically modeled between 0.1 and 1.0 solar masses
    :param Om0: energy budget of the Universe in mass
    :param little_h: reduced Hubble constant
    :return: average Einstein radius in radians
    """
    D_lens = calculate_angular_diameter_distance(
        redshift_lens, Om0=Om0, little_h=little_h
    )
    D_source = calculate_angular_diameter_distance(
        redshift_source, Om0=Om0, little_h=little_h
    )
    D_LS = calculate_angular_diameter_distance_difference(
        redshift_lens, redshift_source, Om0=Om0, little_h=little_h
    )
    value = (
        (
            (4 * const.G * mean_microlens_mass_in_kg / const.c**2)
            * D_LS
            / (D_lens * D_source)
        )
        ** (0.5)
    ).value
    return value


def calculate_einstein_radius_in_meters(
    redshift_lens,
    redshift_source,
    mean_microlens_mass_in_kg=1 * const.M_sun.to(u.kg),
    Om0=0.3,
    little_h=0.7,
):
    """This function determines the einstein radius of the microlenses in physical
    lengths :param redshift_lens: redshift of the lensing galaxy :param redshift_source:
    redshift of the source :param mean_microlens_mass_in_kg: average mass of microlenses
    in the lensing galaxy.

    This is typically     modeled between 0.1 and 1.0 solar masses
    :param Om0: energy budget of the Universe in mass
    :param little_h: reduced Hubble constant
    :return: average Einstein radius of the microlenses in meters
    """
    ang_diam_dist_source_plane = calculate_angular_diameter_distance(
        redshift_source, Om0=Om0, little_h=little_h
    )
    ein_rad_in_radians = calculate_angular_einstein_radius(
        redshift_lens,
        redshift_source,
        mean_microlens_mass_in_kg=mean_microlens_mass_in_kg,
        Om0=0.3,
        little_h=0.7,
    )
    value = ang_diam_dist_source_plane * ein_rad_in_radians
    return value


def pull_value_from_grid(array_2d, x_position, y_position):
    """This approximates the point (x_position, y_position) in a 2d array of values.
    x_position and y_position may be decimals, and are assumed to be measured in pixels.

    :param array_2d: 2 dimensional array of values.
    :param x_position: x coordinate in array_2d
    :param y_position: y coordinate in array_2d
    :return: approximation of array_2d at point (x_position, y_position)
    """
    assert x_position >= 0 and y_position >= 0
    assert x_position < np.size(array_2d, 0) and y_position < np.size(array_2d, 1)
    x_int = x_position // 1
    y_int = y_position // 1
    decx = x_position % 1
    decy = y_position % 1
    baseval = array_2d[int(x_int), int(y_int)]
    # Calculate 1d gradients, allow for edge values
    if int(x_int) + 1 == np.size(array_2d, 0):
        dx = (
            (-1)
            * (array_2d[int(x_int) - 1, int(y_int)] - array_2d[int(x_int), int(y_int)])
            * decx
        )
    else:
        dx = (
            array_2d[int(x_int) + 1, int(y_int)] - array_2d[int(x_int), int(y_int)]
        ) * decx
    if int(y_int) + 1 == np.size(array_2d, 1):
        dy = (
            (-1)
            * (array_2d[int(x_int), int(y_int) - 1] - array_2d[int(x_int), int(y_int)])
            * decy
        )
    else:
        dy = (
            array_2d[int(x_int), int(y_int) + 1] - array_2d[int(x_int), int(y_int)]
        ) * decy
    return array_2d[int(x_int), int(y_int)] + dx + dy


def convert_1d_array_to_2d_array(array_1d):
    """Converts a 1 dimensional list of rays into its 2 dimensional representation.
    Gerlumph maps are stored as a binary file, as a single list of values.

    :param array_1d: A 1d array (or list) of values which must be restacked into a 2d
        array Note this array must correspond to a square output array.
    :return: A 2d numpy array representation of the input
    """
    resolution = int(np.size(array_1d) ** 0.5)
    array_2d = np.reshape(array_1d, newshape=(resolution, resolution))

    return array_2d


def convert_cartesian_to_polar(x, y):
    """Converts coordinate pair (x, y) into (r, phi) coordinates.

    :param: x value in any dimensions
    :param: y value in same dimensions as x value
    :return: tuple representation of radius and azimuth coordinates
    """
    r = (x**2 + y**2) ** 0.5
    # Note numpy uses (y, x)
    phi = np.arctan2(y, x)
    return (r, phi)


def convert_polar_to_cartesian(r, phi):
    """Converts coordinate pair (r, phi) into (x, y) coordinates.

    :param r: radius in polar coordinates
    :param phi: azimuth angle in radians
    :return: tuple representation of x and y coordinates
    """
    x = r * np.sin(phi)
    y = r * np.cos(phi)
    # Note numpy uses (y, x)
    return (y, x)


def perform_microlensing_convolution(
    magnification_array,
    flux_array,
    redshift_lens,
    redshift_source,
    smbh_mass_exponent=8.0,
    mean_microlens_mass_in_kg=1.0 * const.M_sun.to(u.kg),
    number_of_microlens_einstein_radii=25,
    number_of_smbh_gravitational_radii=1000,
    relative_orientation=0,
    Om0=0.3,
    little_h=0.7,
    return_preconvolution_info=False,
    random_seed=None,
):
    """This takes a magnification array and convolves it with the FluxProjection object
    associated with some component of the agn.

    :param magnification_array: a 2d array representation of the magnifications due to
        microlenses
    :param flux_array: a 2d array representation of the flux distribution from the AGN
    :param redshift_lens: an int/float representing the cosmological redshift of the
        lens
    :param redshift_source: an int/float representing the cosmological redshift of the
        source
    :param smbh_mass_exponent: a float representing log_{10} (M_{smbh} / M_{sun})
    :param mean_microlens_mass_in_kg: the mean mass in kg of the microlenses (to
        determine R_{ein})
    :param number_of_microlens_einstein_radii: size of the magnification map in R_{ein}
    :param number_of_smbh_gravitational_radii: radial size of the flux_array (e.g. half
        of one square side)
    :param relative_orientation: angular rotation of flux distribution w.r.t.
        microlensing magnification distribution. If int or float, this defines the
        specific orientation. Any other input will be assigned a random value.
    :param Om0: mass fraction of the Universe in the lambda-CDM model
    :param little_h: the reduced Hubble constant defined as H0 / 100 km / s / Mpc
    :param return_preconvolution_info: return the rescaled flux_array instead of the
        convolution
    :return: a 2d array representing the real valued convolution between the
        magnification_array and flux_array and the pixel_shift to locate the smbh
        location
    """
    np.random.seed(random_seed)

    # rotate the flux distribution by defined or random amount
    if not isinstance(relative_orientation, (int, float)):
        relative_orientation = np.random.rand() * 360
    flux_array = rotate(flux_array, relative_orientation, axes=(0, 1), reshape=False)
    original_total_flux = np.sum(flux_array)

    gravitational_radius_of_smbh = calculate_gravitational_radius(
        10**smbh_mass_exponent
    )
    # determine physical pixel sizes in source plane
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
            Om0=Om0,
            little_h=little_h,
        )
    ) / np.size(magnification_array, 0)

    pixel_ratio = pixel_size_flux_array / pixel_size_magnification_array

    flux_array_rescaled = rescale(flux_array, pixel_ratio)
    new_total_flux = np.sum(flux_array_rescaled)

    # rescale values for flux conservation
    flux_array_rescaled *= original_total_flux / new_total_flux

    if return_preconvolution_info:
        return flux_array_rescaled

    dummy_map = np.zeros(np.shape(magnification_array))
    dummy_map[: np.size(flux_array_rescaled, 0), : np.size(flux_array_rescaled, 1)] = (
        flux_array_rescaled
    )
    convolution = fft.irfft2(fft.rfft2(dummy_map) * fft.rfft2(magnification_array))

    # determine shift of coordinates relative to smbh position
    pixel_shift = np.size(flux_array_rescaled, 0) // 2

    return convolution.real, pixel_shift


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
    trajectory and calling pull_value_from_grid at each relevant point.

    :param convolution_array: The convolution between a flux distribtion and the
        magnification array due to microlensing. Note coordinates on arrays have (y, x)
        signature.
    :param pixel_size: Physical size of a pixel in the source plane, in meters
    :param effective_transverse_velocity: effective transverse velocity in the source
        plane, in km / s
    :param light_curve_time_in_years: duration of the light curve to generate, in years
    :param pixel_shift: offset of the SMBH with respect to the convolved map, in pixels
    :param x_start_position: the x coordinate to start pulling a light curve from, in
        pixels
    :param y_start_position: the y coordinate to start pulling a light curve from, in
        pixels
    :param phi_travel_direction: the angular direction of travel along the convolution,
        in degrees
    :param return_track_coords: bool switch allowing a list of relevant positions to be
        returned
    :return: list representing the microlensing light curve
    """
    rng = np.random.default_rng(seed=random_seed)

    if type(effective_transverse_velocity) == u.Quantity:
        effective_transverse_velocity = effective_transverse_velocity.to(u.m / u.s)
    else:
        effective_transverse_velocity *= u.km.to(u.m)
    if type(light_curve_time_in_years) == u.Quantity:
        light_curve_time_in_years = light_curve_time_in_years.to(u.s)
    else:
        light_curve_time_in_years *= u.yr.to(u.s)

    # check convolution if the map was large enough. Otherwise return original total flux.
    # Note the convolution should be weighted by the square of the pixel shift to conserve flux.
    if pixel_shift >= np.size(convolution_array, 0) / 2:
        print(
            "warning, flux projection too large for this magnification map. Returning average flux."
        )
        return np.sum(convolution_array) / np.size(convolution_array)

    # determine the path length of the light curve in the source plane and include endpoints
    pixels_traversed = (
        effective_transverse_velocity * light_curve_time_in_years / pixel_size
    )

    n_points = (
        effective_transverse_velocity * light_curve_time_in_years / pixel_size
    ) + 2

    # ignore convolution artifacts
    if pixel_shift > 0:
        safe_convolution_array = convolution_array[
            pixel_shift:-pixel_shift, pixel_shift:-pixel_shift
        ]
    else:
        safe_convolution_array = convolution_array

    # guarantee that we will be able to extract a light curve from the safe region for any random start point
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
        # One quadrant will have enough space to extract the light curve
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
                break

    # generate each (x, y) coordinate on the convolution
    x_positions = np.linspace(
        x_start_position, x_start_position + delta_x, int(n_points)
    )
    y_positions = np.linspace(
        y_start_position, y_start_position + delta_y, int(n_points)
    )

    light_curve = []
    for position in range(int(n_points)):
        light_curve.append(
            pull_value_from_grid(
                safe_convolution_array, x_positions[position], y_positions[position]
            )
        )
    if return_track_coords:
        return np.asarray(light_curve), x_positions, y_positions
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
    accretion disk's plane.

    :param radii_array: a 2d array of radial values in units of gravitational radii
    :param phi_array: a 2d array of azimuth values in radians
    :param inclination_angle: the inclination of the agn w.r.t. the observer in degrees
    :param corona_height: the height of the lamppost in gravitational radii (Z_{*}
        coord)
    :param axis_offset_in_gravitational_radii: radial distance from the agn axis of
        symmetry to be used as the lamppost position (R_{*} coord)
    :param angle_offset_in_degrees: azimuth angle in degrees of the offset lamppost
        (phi_{*} coord). Note that 0 degrees is nearest to the observer and 180 degrees
        is furthest away.
    :param height_array: an optional 2d array of the height values in gravitational
        radii
    :return: a 2d array of time lags in units [r_{g} / c]
    """

    inclination_angle *= np.pi / 180
    angle_offset_in_degrees *= np.pi / 180

    x_axis_offset = -axis_offset_in_gravitational_radii * np.cos(
        angle_offset_in_degrees
    )
    y_axis_offset = axis_offset_in_gravitational_radii * np.sin(angle_offset_in_degrees)

    if height_array is not None:
        assert np.shape(height_array) == np.shape(radii_array)
        height_array = np.asarray(height_array)
    else:
        height_array = np.zeros(np.shape(radii_array))

    # get height relative to lamppost source
    height_array -= corona_height

    # convert to cartesian to allow non axi-symmetric systems
    x_array, y_array = convert_polar_to_cartesian(radii_array, phi_array)
    x_array += x_axis_offset
    y_array += y_axis_offset

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
    smbh_mass_exponent,
    corona_height,
    axis_offset_in_gravitational_radii=0,
    angle_offset_in_degrees=0,
    height_array=None,
    albedo_array=None,
):
    """
    Calculate the geometric factor of the accretion disk due to lamppost heating according to
    (1 - A) cos(theta_x) / (4 * pi * sigma_sb * R_{*}^{2}}
    This gets weighted by the lamppost X-ray flux L_{x} (eq. 2 in Cackett+ 2007)

    :param temp_array: a 2d array of effective temperatures in Kelvin
    :param radii_array: a 2d array of radii in gravitational radii
    :param smbh_mass_exponent: the solution of log_{10} (M_{smbh} / M_{sun})
    :param corona_height: the height of the lamppost in gravitational radii
    :param axis_offset_in_gravitational_radii: axis offset of the lamppost w.r.t. axis of symmetry
    :param angle_offset_in_degrees: azimuth position of the offset lamppost
    :param height_array: array of heights to calculate the disk at. Allows for greater flexability in
        disk model (e.g. no more flat disks only!)
    :param albedo_array: int, float, or array of albedos (reflectivities) to use for the disk
    :return: a 2d array of geometric disk factors which determine the flux reprocessing of the
        lamppost by the accretion disk
    """

    assert np.shape(temp_array) == np.shape(radii_array)
    # check height array if it was input
    if height_array is not None:
        assert np.shape(height_array) == np.shape(radii_array)
    else:
        height_array = np.zeros(np.shape(radii_array))
    height_array -= corona_height

    angle_offset_in_degrees *= np.pi / 180

    x_axis_offset = -axis_offset_in_gravitational_radii * np.cos(
        angle_offset_in_degrees
    )
    y_axis_offset = axis_offset_in_gravitational_radii * np.sin(angle_offset_in_degrees)
    x_array, y_array = convert_polar_to_cartesian(radii_array, phi_array)
    x_array += x_axis_offset
    y_array += y_axis_offset

    new_radii, new_azimuths = convert_cartesian_to_polar(x_array, y_array)

    # check albedo array if it was input
    if isinstance(albedo_array, (int, float)):
        albedo_array *= np.ones(np.shape(new_radii))
    elif isinstance(albedo_array, (np.ndarray)):
        assert np.shape(albedo_array) == np.shape(new_radii)
    else:
        albedo_array = np.zeros(np.shape(new_radii))

    gravitational_radius = calculate_gravitational_radius(10**smbh_mass_exponent)

    # approximate the normal height vector to get angle of incidence
    # I really need dh/dr to do this.
    # both height array and radii array are calculated on the same field, so chain rule works
    # gradient function takes x and y directions individually.
    height_gradient_x, height_gradient_y = np.gradient(height_array)
    radii_gradient_x, radii_gradient_y = np.gradient(new_radii)

    # need to find a way to make negative dh/dr values, and assign them no reprocessing

    dh_dr = (
        (height_gradient_x / radii_gradient_x) ** 2
        + (height_gradient_y / radii_gradient_y) ** 2
    ) ** 0.5

    # use arctan for dh_dr since there is only one argument.
    # arctan2 is for angles relative to two arguments.
    theta_star = np.pi - np.arctan(dh_dr) - np.arctan2(height_array, new_radii)

    # fix the quadrant
    theta_star = abs(theta_star % (np.pi))

    # cos_theta_star = np.ones(np.shape(radii_array)) * corona_height * gravitational_radius / radii_array
    cos_theta_star = np.cos(theta_star)

    radii_star = (new_radii**2 + height_array**2) ** 0.5 * gravitational_radius

    return (
        (1 - albedo_array)
        * cos_theta_star
        / (4 * np.pi * const.sigma_sb * radii_star**2)
    )


def calculate_dt_dlx(
    temp_array,
    radii_array,
    phi_array,
    smbh_mass_exponent,
    corona_height,
    axis_offset_in_gravitational_radii=0,
    angle_offset_in_degrees=0,
    height_array=None,
    albedo_array=None,
):
    """Approximates the change in temperature due to the change in lamppost flux
    assuming the irradiated.

    disk model, following the Taylor expansion. delta_t / delta_lx ~
    geometric_disk_factor / (4 * disk_temp**3)

    As such, this primarily uses calculate_geometric_disk_factor() and weights it by the
    temperature.

    :param temp_array: a 2d array representing the effective temperature of the
        accretion disk
    :param radii_array: a 2d array representing the radii from the smbh in gravitational
        radii
    :param phi_array: a 2d array representing the azimuths on the accretion disk in
        radians
    :param smbh_mass_exponent: the solutkon of log_{10} (M_{smbh} / M_{sun})
    :param corona_height: the lamppost height in gravitational radii
    :param axis_offset_in_gravitational_radii: the offset of the lamppost in
        gravitational radii
    :param angle_offset_in_degrees: the azimuth of the offset of the lamppost in degrees
    :param height_array: array of heights to calculate the disk at. Allows for greater
        flexability in disk model (e.g. no more flat disks only!)
    :param albedo_array: int, float, or array of albedos (reflectivities) to use for the
        disk
    :return: a 2d array representing the change in effective temperature with respect to
        the luminosity of the x-ray source
    """

    geometric_weighting_array = calculate_geometric_disk_factor(
        temp_array,
        radii_array,
        phi_array,
        smbh_mass_exponent,
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
    inclination_angle,
    smbh_mass_exponent,
    corona_height,
    axis_offset_in_gravitational_radii=0,
    angle_offset_in_degrees=0,
    height_array=None,
    albedo_array=None,
    return_response_array_and_lags=False,
):
    """This calculates the accretion disk's transfer function in the lamppost geometry
    for some given effective temperature mapping. Does not rely on a particular
    temperature profile, but it does assume the radiation is black-body like.

    :param rest_wavelength_in_nm: rest wavelength to calculate the transfer function at,
        in nm
    :param temp_array: a 2d array of effective temperatures of the accretion disk
    :param radii_array: a 2d array of radii across the accretion disk in gravitational
        radii
    :param phi_array: a 2d array of azimuths on the accretion disk in radians
    :param inclination_angle: the inclination of the accretin disk w.r.t. to the
        observer, in degrees
    :param smbh_mass_exponent: the solution to log_{10} (M_{smbh} / M_{sun})
    :param corona_height: height of the lamppost in gravitational radii
    :param axis_offset_in_gravitational_radii: radial distance from the agn axis of
        symmetry to be used as the lamppost position (R_{*} coord)
    :param angle_offset_in_degrees: azimuth angle in degrees of the offset lamppost
        (phi_{*} coord). Note that 0 degrees is nearest to the observer and 180 degrees
        is furthest away.
    :param height_array: an optional 2d array of the height values in gravitational
        radii
    :param albedo_array: float, int, or array of albedo (reflectivity) values to use
    :param return_response_array_and_lags: bool used to return the response map instead
        of the transfer function
    :return: a normalized 1d representation of the transfer function of the accretion
        disk with time lags represented in units r_{g} / c.
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
        smbh_mass_exponent,
        corona_height,
        axis_offset_in_gravitational_radii=axis_offset_in_gravitational_radii,
        angle_offset_in_degrees=angle_offset_in_degrees,
        height_array=height_array,
        albedo_array=albedo_array,
    )

    response_factors = db_dt_array * dt_dlx_array

    if return_response_array_and_lags:
        return response_factors, time_lag_array

    transfer_function = np.histogram(
        rescale(time_lag_array, 10),
        range=(0, np.max(time_lag_array) + 1),
        bins=int(np.max(time_lag_array) + 1),
        weights=np.nan_to_num(rescale(response_factors, 10)),
        density=True,
    )[0]

    # return the normalized transfer function
    return transfer_function / np.sum(transfer_function)


def calculate_microlensed_transfer_function(
    magnification_array,
    redshift_lens,
    redshift_source,
    rest_wavelength_in_nm,
    temp_array,
    radii_array,
    phi_array,
    inclination_angle,
    smbh_mass_exponent,
    corona_height,
    mean_microlens_mass_in_kg=1.0 * const.M_sun.to(u.kg),
    number_of_microlens_einstein_radii=25,
    number_of_smbh_gravitational_radii=1000,
    relative_orientation=0,
    Om0=0.3,
    little_h=0.7,
    axis_offset_in_gravitational_radii=0,
    angle_offset_in_degrees=0,
    height_array=None,
    albedo_array=None,
    x_position=None,
    y_position=None,
    return_response_array_and_lags=False,
    return_descaled_response_array_and_lags=False,
    random_seed=None,
):
    """Calculate the transfer function assuming the response of the disk can be
    amplified by microlensing.

    :param magnification_array: a 2d array of magnifications in the source plane
    :param rest_wavelength_in_nm: rest frame wavelength in nanometers to calculate the
        transfer function at
    :param temp_array: a 2d array representing the effective temperatures of the
        accretion disk
    :param radii_array: a 2d array representing the radii of each pixel in the source
        plane with units of gravitational radii
    :param phi_array: a 2d array representing the azimuths of each pixel in the source
        plane in radians
    :param inclination_angle: inclination of the accretion disk w.r.t. the observer in
        degrees
    :param smbh_mass_exponent: the solution of log_{10} ( M_{smbh} / M_{sun} )
    :param corona_height: height of the lamppost in gravitational radii
    :param axis_offset_in_gravitational_radii: the offset of the lamppost in
        gravitational radii
    :param angle_offset_in_degrees: the azimuth of the offset of the lamppost in degrees
    :param height_array: array of heights to calculate the disk at. Allows for greater
        flexability in disk model (e.g. no more flat disks only!)
    :param albedo_array: int, float, or array of albedos (reflectivities) to use for the
        disk
    :param x_position: an optional x coordinate location to use on the magnification
        map. Otherwise, will be chosen with np.random
    :param y_position: an optional y coordinate location to use on the magnification
        map. Otherwise, will be chosen with np.random
    :param random_seed: random seed to use for reproducibility
    :param return_response_array_and_lags: bool to return a representation of the
        amplified response map
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
        inclination_angle,
        smbh_mass_exponent,
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
        smbh_mass_exponent=smbh_mass_exponent,
        mean_microlens_mass_in_kg=mean_microlens_mass_in_kg,
        number_of_microlens_einstein_radii=number_of_microlens_einstein_radii,
        number_of_smbh_gravitational_radii=number_of_smbh_gravitational_radii,
        relative_orientation=relative_orientation,
        Om0=Om0,
        little_h=little_h,
        return_preconvolution_info=True,
    )

    scale_ratio = np.size(rescaled_response_array, 0) / np.size(disk_response_array, 0)

    rescaled_time_lag_array = rescale(time_lag_array, scale_ratio)
    assert np.shape(rescaled_time_lag_array) == np.shape(rescaled_response_array)

    # rescaled arrays represent the images scaled to the size of pixels in the magnification array
    pixel_shift = np.size(rescaled_time_lag_array, 0) // 2

    # assure any position can be used, even if it falls off the edge of the map. Pad with ones.
    magnification_array_padded = np.pad(
        magnification_array, pixel_shift, constant_values=(1, 1)
    )

    # use random location if not provided
    if x_position is None:
        x_position = int(rng.random() * np.size(magnification_array, 0))
    if y_position is None:
        y_position = int(rng.random() * np.size(magnification_array, 1))

    # account for padding
    x_position += pixel_shift
    y_position += pixel_shift

    # get the relevant magnification region
    magnification_crop = magnification_array_padded[
        x_position
        - pixel_shift : x_position
        - pixel_shift
        + np.size(rescaled_response_array, 0),
        y_position
        - pixel_shift : y_position
        - pixel_shift
        + np.size(rescaled_response_array, 1),
    ]

    magnified_response_array = rescaled_response_array * magnification_crop

    if return_response_array_and_lags:
        return magnified_response_array, rescaled_time_lag_array

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
        return unscaled_magnified_response_array, unscaled_time_lag_array

    microlensed_transfer_function = np.histogram(
        rescaled_time_lag_array,
        range=(0, np.max(rescaled_time_lag_array) + 1),
        bins=int(np.max(rescaled_time_lag_array) + 1),
        weights=np.nan_to_num(magnified_response_array),
        density=True,
    )[0]

    # return the normalized transfer function
    return np.nan_to_num(
        microlensed_transfer_function / np.sum(microlensed_transfer_function)
    )


def generate_drw_signal(
    length_of_light_curve, time_step, sf_infinity, tau_drw, random_seed=None
):
    """Generate a damped random walk using typical parameters as defined in Kelly+ 2009.
    Uses recursion.

    :param length_of_light_curve: the length of the light curve
    :param time_step: the spacing of the light curve, in identical units to maximum_time
    :param sf_infinity: the asymptotic structure function of the damped random walk
    :param tau_drw: the characteristic time scale of the variability
    :param random_seed: random seed to use for reproducibility
    :return: an array representing the damped random walk
    """
    rng = np.random.default_rng(seed=random_seed)

    number_of_points = int(length_of_light_curve / time_step) + 1

    output_drw = np.zeros(number_of_points)

    for point in range(number_of_points - 1):
        output_drw[point + 1] = output_drw[point] * np.exp(
            -abs(time_step / tau_drw)
        ) + (sf_infinity / np.sqrt(1 / 2)) * rng.random() * (
            1 - (np.exp(-2 * abs(time_step / tau_drw)))
        ) ** (
            1 / 2
        )

    return output_drw


def generate_signal_from_psd(
    length_of_light_curve,
    power_spectrum,
    frequencies,
    random_seed=None,
):
    """Generate a signal from any power spectrum using the methods of Timmer+.
    length_of_light_curve and frequencies must be recipricol units. the output light
    curve will be normalized to have mean 0, std 1.

    :param length_of_light_curve: maximum length of the light curve to generate. Note that this maximum
        value is dependent on the input frequencies, since the frequencies can only generate a light
        curve ranging from values between the Nyquist frequency [1/(2 * max(frequency))] and 1/min(frequency)
    :param power_spectrum: the input power spectrum of the stochastic signal at each fourier frequency
        defined in the frequencies parameter.
    :param frequencies: the input fouer frequencies associated with the power spectrum. Note these should be
        defined in linear space as:
        np.linspace(1/length_of_light_curve, 1/(2 * desired_time_resolution), int(length_of_light_curve)+1)
    :param random_seed: random seed to use for reproducibility
    :return: signal generated from the power spectrum ith length defined by length_of_light_curve.
    """
    rng = np.random.default_rng(seed=random_seed)

    random_phases = 2 * np.pi * rng.random(size=len(frequencies))

    positive_fourier_plus_phases = np.sqrt(power_spectrum) * np.exp(1j * random_phases)

    fourier_transform_of_output = np.concatenate(
        (
            positive_fourier_plus_phases,
            positive_fourier_plus_phases[-2:0:-1].conjugate(),
        )
    )

    light_curve = np.fft.ifft(fourier_transform_of_output)[: int(length_of_light_curve)]

    light_curve -= np.mean(light_curve)

    if np.std(light_curve) > 0:
        light_curve /= np.std(light_curve)

    return light_curve


def generate_snapshots_of_radiation_pattern(
    rest_wavelength_in_nm,
    time_stamps,
    temp_array,
    radii_array,
    phi_array,
    smbh_mass_exponent,
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

    :param rest_wavelength_in_nm: rest frame wavelength of observation, in nm
    :param time_stamps: list of dates to extract the radiation pattern at, in days
    :param temp_array: a 2d array of the effective temperatures of the accretion disk
    :param radii_array: a 2d array of radii of the accretion disk, in gravitational
        radii
    :param phi_array: a 2d array of azimuth values on the accretion disk, in radians
    :param smbh_mass_exponent: the solution to log_{10} (M_{bh} / M_{sun})
    :param driving_signal: a list representing the underlying driving signal which
        produces the radiation pattern on the accretion disk
    :param driving_signal_fractional_strength: relative strength of the total flux due
        to the reprocessing, on a scale of (0, 1). 0 represents no contribution while 1
        represnts no static flux contribution.
    :param corona_height: height of the lamppost corona in gravitational radii
    :param inclination_angle: the inclination of the agn w.r.t. to the observer, in
        degrees
    :param axis_offset_in_gravitational_radii: the offset of the lamppost in
        gravitational radii
    :param angle_offset_in_degrees: the azimuth of the offset of the lamppost in degrees
    :param height_array: array of heights to calculate the disk at. Allows for greater
        flexability in disk model (e.g. no more flat disks only!)
    :param albedo_array: int, float, or array of albedos (reflectivities) to use for the
        disk
    :return: a series of 2d arrays representing the radiation pattern at each value of
        time_stamps
    """

    static_flux = planck_law(temp_array, rest_wavelength_in_nm)

    total_static_flux = np.sum(static_flux)

    response_array, time_lag_array = construct_accretion_disk_transfer_function(
        rest_wavelength_in_nm,
        temp_array,
        radii_array,
        phi_array,
        inclination_angle,
        smbh_mass_exponent,
        corona_height,
        axis_offset_in_gravitational_radii=axis_offset_in_gravitational_radii,
        angle_offset_in_degrees=angle_offset_in_degrees,
        height_array=height_array,
        albedo_array=albedo_array,
        return_response_array_and_lags=True,
    )

    gravitational_radius = calculate_gravitational_radius(10**smbh_mass_exponent)
    gr_per_day = gravitational_radius / const.c.to(u.m / u.day).value

    # convert time lags from R_g / c to units of days
    time_lag_array *= gr_per_day
    maximum_time_lag_in_days = np.max(time_lag_array)

    # normalize response_array because we want a fractional response w.r.t. the static_flux array
    response_array *= total_static_flux / np.sum(response_array)

    if len(driving_signal) < np.max(time_stamps + maximum_time_lag_in_days):
        print(
            "warning, driving signal is not long enough to support all snapshots. looping signal"
        )
        while len(driving_signal) < np.max(time_stamps + maximum_time_lag_in_days):
            driving_signal = np.concatenate((driving_signal, driving_signal))

    # now the signal is guaranteed to support all time stamps, so generate radiation patterns
    # define a burn in such that the whole disk is being driven at t=0
    burn_in_time = maximum_time_lag_in_days
    accretion_disk_mask = temp_array > 0

    list_of_snapshots = []
    # prepare snapshots
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
    smbh_mass_exponent,
    velocity_range=[-1, 1],
    weighting_grid=None,
    radial_resolution=1,
    vertical_resolution=1,
):
    """Takes an axi-symmetric grid of density values and weighting grid with coordinate
    signature (R, Z) and projects it to the source plane. It also can select particular
    velocity ranges.

    :param blr_density_rz_grid: a 2d array of values representing the density of the blr
        at each point in (R, Z) coords.
    :param blr_vertical_velocity_grid: a 2d array of v_{z} values, normalized by the
        speed of light.
    :param blr_radial_velocity_grid: a 2d array of v_{r} values, normalized by the speed
        of light
    :param inclination_angle: the inclination of the agn w.r.t. the observer in degrees.
    :param smbh_mass_exponent: the solution of log_{10} (M_{bh} / M_{sun})
    :param velocity_range: the range of line-of-sight velocities which are accepted, in
        units of speed of light. We take the convention of positive values are aimed
        towards the observer, and are therefore blueshifted.
    :param weighting_grid: a 2d array of values which correspond to weighting factors in
        the blr_density_rz_grid
    :param radial_resolution: the spacing between radial coordinates in gravitational
        radii
    :param vertical_resolution: the spacing between vertical coordinates in
        gravitational radii
    :return: a 2d array representing the projected blr in the source plane with pixel
        resolution equal to radial_resolution.
    """
    # check values and array sizes
    assert inclination_angle >= 0
    assert inclination_angle < 90
    assert np.shape(blr_density_rz_grid) == np.shape(blr_vertical_velocity_grid)
    assert np.shape(blr_vertical_velocity_grid) == np.shape(blr_radial_velocity_grid)
    if inclination_angle > 80:
        print("warning, source plane is nearly orthogonal to each constant height slab")
        print("each slab follows scaling O(tan(inc)^2)")
    inclination_angle *= np.pi / 180

    if weighting_grid is None:
        weighting_grid = np.ones(np.shape(blr_density_rz_grid))
    assert np.shape(weighting_grid) == np.shape(blr_density_rz_grid)

    gravitational_radius = calculate_gravitational_radius(10**smbh_mass_exponent)

    # do not over-resolve in source plane or gaps will form
    source_plane_resolution = radial_resolution

    max_r = np.size(blr_density_rz_grid, 0) * radial_resolution
    max_z = np.size(blr_density_rz_grid, 1) * vertical_resolution

    max_projected_size_in_source_plane = max_z * np.tan(inclination_angle) + max_r

    # initialize the projection in the source plane
    source_plane_projection = np.zeros(
        (
            int(2 * max_projected_size_in_source_plane / source_plane_resolution),
            int(2 * max_projected_size_in_source_plane / source_plane_resolution),
        )
    )

    # project each slab of the blr into the source plane
    for height in range(np.size(blr_density_rz_grid, 1)):
        current_y_offset = height * vertical_resolution * np.tan(inclination_angle)

        # get the projected x, y coords in the current slab
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

        # define the indexes to use
        index_grid = R // radial_resolution

        keplerian_velocities = calculate_keplerian_velocity(
            index_grid * radial_resolution * gravitational_radius,
            10**smbh_mass_exponent,
        )

        # mask out any radii not included in the blr_density_rz_grid
        index_mask = np.logical_and(
            (index_grid < np.size(blr_density_rz_grid, 0)), (index_grid > 0)
        )

        index_grid *= index_mask

        # non-relativistic approximation by addition of components
        line_of_sight_velocities = (
            np.cos(inclination_angle)
            * blr_vertical_velocity_grid[index_grid.astype(int), height]
            + np.sin(inclination_angle)
            * np.sin(Phi)
            * blr_radial_velocity_grid[index_grid.astype(int), height]
            - np.sin(inclination_angle)
            * np.cos(Phi)
            * keplerian_velocities[index_grid.astype(int), height]
        )

        # generate the velocity selected region
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

    return source_plane_projection


def calculate_blr_transfer_function(
    blr_density_rz_grid,
    blr_vertical_velocity_grid,
    blr_radial_velocity_grid,
    inclination_angle,
    smbh_mass_exponent,
    velocity_range=[-1, 1],
    weighting_grid=None,
    radial_resolution=1,
    vertical_resolution=1,
):
    """Calculate the response functino of the blr assuming some weighting factors for
    some wavelength range.

    :param blr_density_rz_grid: a 2d array of values representing the density of the blr
        at each point in (R, Z) coords.
    :param blr_vertical_velocity_grid: a 2d array of v_{z} values, normalized by the
        speed of light.
    :param blr_radial_velocity_grid: a 2d array of v_{r} values, normalized by the speed
        of light
    :param inclination_angle: the inclination of the agn w.r.t. the observer in degrees.
    :param smbh_mass_exponent: the solution of log_{10} (M_{bh} / M_{sun})
    :param velocity_range: the range of line-of-sight velocities which are accepted, in
        units of speed of light. We take the convention of positive values are aimed
        towards the observer, and are therefore blueshifted.
    :param weighting_grid: a 2d array of values which correspond to weighting factors in
        the blr_density_rz_grid
    :param radial_resolution: the spacing between radial coordinates in gravitational
        radii
    :param vertical_resolution: the spacing between vertical coordinates in
        gravitational radii
    :return: a 1d array representing the response function of the blr w.r.t. the optical
        accretion disk
    """

    # check values and array sizes
    assert inclination_angle >= 0
    assert inclination_angle < 90
    assert np.shape(blr_density_rz_grid) == np.shape(blr_vertical_velocity_grid)
    assert np.shape(blr_vertical_velocity_grid) == np.shape(blr_radial_velocity_grid)
    if inclination_angle > 80:
        print("warning, source plane is nearly orthogonal to each constant height slab")
        print("each slab follows scaling O(tan(inc)^2)")
    inclination_angle *= np.pi / 180

    if weighting_grid is None:
        weighting_grid = np.ones(np.shape(blr_density_rz_grid))
    assert np.shape(weighting_grid) == np.shape(blr_density_rz_grid)

    # unlike in a projection, we follow the axis of symmetry
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

    # define the indexes to use
    index_grid = R // radial_resolution

    index_mask = np.logical_and(
        (index_grid < np.size(blr_density_rz_grid, 0)), (index_grid > 0)
    )

    # get Keplerian velocitites (no z dependence)
    gravitational_radius = calculate_gravitational_radius(10**smbh_mass_exponent)

    keplerian_velocities = calculate_keplerian_velocity(
        index_grid * radial_resolution * gravitational_radius, 10**smbh_mass_exponent
    )

    # mask those which extend beyond max radius AFTER calculating Kep. velocities
    index_grid *= index_mask

    # initialize transfer function
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

    # cycle through each slab of the blr
    for height in range(np.size(blr_density_rz_grid, 1)):

        # non-relativistic approximation by addition of components
        line_of_sight_velocities = (
            np.cos(inclination_angle)
            * blr_vertical_velocity_grid[index_grid.astype(int), height]
            + np.sin(inclination_angle)
            * np.sin(Phi)
            * blr_radial_velocity_grid[index_grid.astype(int), height]
            - np.sin(inclination_angle)
            * np.cos(Phi)
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
            rescale(time_delays_of_current_slab, 2 * radial_resolution),
            range=(0, np.max(time_delays_of_current_slab) + 1),
            bins=int(np.max(time_delays_of_current_slab) + 1),
            weights=np.nan_to_num(
                rescale(response_of_current_slab, 2 * radial_resolution)
            ),
            density=True,
        )[0]

        # add this slab's contribution to the total transfer function
        output_transfer_function[: np.size(transfer_function_of_slab)] += np.nan_to_num(
            transfer_function_of_slab
        )

    # only normalize if it won't make nans
    if np.sum(output_transfer_function) > 0:
        output_transfer_function /= np.sum(output_transfer_function)

    return output_transfer_function


def determine_emission_line_velocities(
    rest_frame_emitted_wavelength, passband_minimum, passband_maximum, redshift
):
    """Helper function to define the velocity range of an emission line due to Doppler
    broadening which is required to shift the emission line into the desired filter.

    :param rest_frame_emitted_wavelength: emission line wavelength in rest frame. Units
        must match passband units, but may be arbitrary.
    :param passband_minimum: minimum wavelength of the passband (filter). Units may be
        arbitrary, but must match rest_frame_emitted_wavelength.
    :param passband_maximum: maximum wavelength of the passband (filter). Units may be
        arbitrary, but must match rest_frame_emitted_wavelength.
    :param redshift: cosmological redshift factor
    :return: list of [minimum, maximum] velocities which will shift the emission line
        into the desired filter at redshift.
    """
    assert redshift >= 0
    rest_frame_passband_minimum = passband_minimum / (1 + redshift)
    rest_frame_passband_maximum = passband_maximum / (1 + redshift)

    # since we define positive velocity as approaching the observer, these are switched
    required_velocity_minimum = (
        1 - (rest_frame_passband_maximum / rest_frame_emitted_wavelength) ** 2
    ) / ((rest_frame_passband_maximum / rest_frame_emitted_wavelength) ** 2 + 1)

    required_velocity_maximum = (
        1 - (rest_frame_passband_minimum / rest_frame_emitted_wavelength) ** 2
    ) / ((rest_frame_passband_minimum / rest_frame_emitted_wavelength) ** 2 + 1)

    return [required_velocity_minimum, required_velocity_maximum]
