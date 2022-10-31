"""
Functions and objects related to the calculation of atmospheric properties.
"""

# Imports ==========================================================================================
from math import sqrt, exp
from dataclasses import dataclass

import flyingcircus.constants as constants

# Objects ==========================================================================================


@dataclass
class AtmosState:

    geometric_altitude_m: float
    geopotential_altitude_m: float
    temperature_k: float
    pressure_pa: float
    density_kg_m3: float
    speed_of_sound_m_s: float
    dynamic_viscosity_ns_m2: float


# Functions ========================================================================================


def calc_standard_atmosphere_speed_of_sound(temperature_k: float) -> float:
    """Given a temperature in Kelvin returns the standard atmosphere speed of sound, for that
    temperature, in meters per second.

    References:
    - Phillips, Warren F. - "Mechanics of Flight", 2nd edition, page 12.

    Args:
        temperature_k (float): air temperature in Kelvin.

    Returns:
        float: standard atmosphere speed of sound for that temperature in meters per second.
    """

    gamma = constants.STANDARD_ATMOSPHERE_HEAT_CAPACITY_RATIO
    r = constants.STANDARD_ATMOSPHERE_IDEAL_GAS_CONSTANT_NM_KGK
    t = temperature_k

    return sqrt(gamma * r * t)


def calc_standard_atmosphere_density(temperature_k: float, pressure_pa: float) -> float:
    """Given a temperature in Kelvin and a pressure in Pascals returns the standard atmosphere
    density in kilograms per cubic meter.

    References:
    - Phillips, Warren F. - "Mechanics of Flight", 2nd edition, page 12.

    Args:
        temperature_k (float): air temperature in Kelvin.

    Returns:
        float: standard atmosphere density in kilograms per cubic meter.
    """

    r = constants.STANDARD_ATMOSPHERE_IDEAL_GAS_CONSTANT_NM_KGK
    t = temperature_k
    p = pressure_pa

    return p / (r * t)


def calc_geopotential_altitude(geometrical_altitude_m: float) -> float:
    """Given a geometric altitude in meters returns the respective geopotential altitude in meters.

    References:
    - Phillips, Warren F.; "Mechanics of Flight", 2nd edition, page 12.

    Args:
        geometrical_altitude_m (float): geometrical altitude in meters.

    Returns:
        float: geopotential altitude in meters.
    """

    r_e = constants.EARTH_RADIUS_M
    h = geometrical_altitude_m

    return (r_e * h) / (r_e + h)


def calc_standard_atmosphere_dynamic_viscosity(temperature_k: float) -> float:
    """Given a temperature in Kelvin returns the standard atmosphere dynamic viscosity for that
    temperature based on Sutherland's formula and empirical data.

    References:
    - NASA - "U.S. Standard Atmosphere 1976", NASA-TM-X-74335, pages 19; 115
    - Schlichting, Hermann; Gersten, Klaus - "Boundary-Layer Theory", 9th edition, page 243

    Args:
        temperature_k (float): air temperature in Kelvin

    Returns:
        float: air's dynamic viscosity in (Newton . second) / (meter^2)
    """

    u_r = constants.STANDARD_ATMOSPHERE_SEA_LEVEL_DYNAMIC_VISCOSITY_NS_M2
    t_r = constants.STANDARD_ATMOSPHERE_SEA_LEVEL_TEMPERATURE_K
    t = temperature_k
    s = constants.STANDARD_ATMOSPHERE_SUTHERLAND_CONSTANT_K

    return u_r * ((t_r + s) / (t + s)) * (t / t_r) ** (3 / 2)


def calc_standard_atmosphere_temperature(geopotential_altitude_m: float) -> float:
    """Given a geopotential altitude in meters returns the standard atmosphere temperature
    for that altitude.

    References:
    - Phillips, Warren F. - "Mechanics of Flight", 2nd edition, pages 10-14.
    - NASA - "U.S. Standard Atmosphere 1976", NASA-TM-X-74335

    Args:
        geopotential_altitude_m (float): geopotential altitude in meters

    Returns:
        float: standard atmosphere temperature in Kelvin.
    """

    altitude_ranges = sorted(constants.STANDARD_ATMOSPHERE_TEMPERATURE_GRADIENTS_K_M.keys())

    if (geopotential_altitude_m) < 0.0 or (
        geopotential_altitude_m
        > calc_geopotential_altitude(constants.STANDARD_ATMOSPHERE_MAX_ALTITUDE_M)
    ):

        raise ValueError("Altitude outside the atmospheric model range of 0m to 86000m")

    elif geopotential_altitude_m < altitude_ranges[1]:

        t_0 = constants.STANDARD_ATMOSPHERE_SEA_LEVEL_TEMPERATURE_K
        delta_t = constants.STANDARD_ATMOSPHERE_TEMPERATURE_GRADIENTS_K_M[altitude_ranges[0]]

        return t_0 + delta_t * geopotential_altitude_m

    else:

        idx = 0

        for i, altitude in enumerate(altitude_ranges):
            if altitude >= geopotential_altitude_m:
                idx = i - 1
                break
            else:
                idx = i

        t_0 = calc_standard_atmosphere_temperature(altitude_ranges[idx])
        delta_t = constants.STANDARD_ATMOSPHERE_TEMPERATURE_GRADIENTS_K_M[altitude_ranges[idx]]

        return t_0 + delta_t * (geopotential_altitude_m - altitude_ranges[idx])


def calc_standard_atmosphere_pressure(geopotential_altitude_m: float) -> float:
    """Given a geopotential altitude in meters returns the standard atmosphere pressure
    for that altitude.

    References:
    - Phillips, Warren F. - "Mechanics of Flight", 2nd edition, pages 10-14.
    - NASA - "U.S. Standard Atmosphere 1976", NASA-TM-X-74335

    Args:
        geopotential_altitude_m (float): geopotential altitude in meters
        temperature_offset_k (float, optional): temperature offset Kelvin in relation to the
            standard atmosphere temperature. Defaults to 0.0.

    Returns:
        float: standard atmosphere pressure in Pascals.
    """

    altitude_ranges = sorted(constants.STANDARD_ATMOSPHERE_TEMPERATURE_GRADIENTS_K_M.keys())
    r = constants.STANDARD_ATMOSPHERE_IDEAL_GAS_CONSTANT_NM_KGK
    g_0 = constants.EARTH_GRAVITY_ACCELERATION_M_S2

    if (geopotential_altitude_m) < 0.0 or (
        geopotential_altitude_m
        > calc_geopotential_altitude(constants.STANDARD_ATMOSPHERE_MAX_ALTITUDE_M)
    ):

        raise ValueError("Altitude outside the atmospheric model range of 0m to 86000m")

    elif geopotential_altitude_m < altitude_ranges[1]:

        t_0 = constants.STANDARD_ATMOSPHERE_SEA_LEVEL_TEMPERATURE_K
        p_0 = constants.STANDARD_ATMOSPHERE_SEA_LEVEL_PRESSURE_PA
        t = calc_standard_atmosphere_temperature(geopotential_altitude_m)
        delta_t = constants.STANDARD_ATMOSPHERE_TEMPERATURE_GRADIENTS_K_M[altitude_ranges[0]]

        if delta_t == 0.0:

            return p_0 * exp((-g_0 * geopotential_altitude_m) / (r * t))

        else:

            return p_0 * (t / t_0) ** ((-g_0) / (r * delta_t))

    else:

        idx = 0

        for i, altitude in enumerate(altitude_ranges):
            if altitude >= geopotential_altitude_m:
                idx = i - 1
                break
            else:
                idx = i

        t_0 = calc_standard_atmosphere_temperature(altitude_ranges[idx])
        p_0 = calc_standard_atmosphere_pressure(altitude_ranges[idx])
        t = calc_standard_atmosphere_temperature(geopotential_altitude_m)
        delta_t = constants.STANDARD_ATMOSPHERE_TEMPERATURE_GRADIENTS_K_M[altitude_ranges[idx]]

        if delta_t == 0.0:

            return p_0 * exp((-g_0 * (geopotential_altitude_m - altitude_ranges[idx])) / (r * t))

        else:

            return p_0 * (t / t_0) ** ((-g_0) / (r * delta_t))


def calc_standard_atmosphere(
    geometric_altitude_m: float, temperature_offset_k: float = 0.0
) -> AtmosState:
    """Given a geometric altitude in meters and a temperature offset in Kelvin, returns an
    AtmosState object containing the standard atmosphere properties for that altitude and
    temperature offset.

    References:
    - Phillips, Warren F. - "Mechanics of Flight", 2nd edition, pages 10-14.
    - NASA - "U.S. Standard Atmosphere 1976", NASA-TM-X-74335
    - Schlichting, Hermann; Gersten, Klaus - "Boundary-Layer Theory", 9th edition, page 243

    Args:
        geometric_altitude_m (float): geometric altitude im meters.
        temperature_offset_k (float, optional): temperature offset Kelvin in relation to the
            standard atmosphere temperature. Defaults to 0.0.

    Returns:
        AtmosState: an object containing the temperature in Kelvin, the pressure in Pascal,
            the density in kilograms per cubic meter, the speed of sound in meters per second,
            and the dynamic viscosity calculated for the standard atmosphere for the required
            altitude and temperature offset.
    """

    geopotential_altitude_m = calc_geopotential_altitude(geometric_altitude_m)

    if not (0.0 <= geometric_altitude_m <= constants.STANDARD_ATMOSPHERE_MAX_ALTITUDE_M):

        raise ValueError("Altitude outside the atmospheric model range of 0m to 86000m")

    temperature_k = (
        calc_standard_atmosphere_temperature(geopotential_altitude_m) + temperature_offset_k
    )
    pressure_pa = calc_standard_atmosphere_pressure(geopotential_altitude_m)
    density_kg_m3 = calc_standard_atmosphere_density(temperature_k, pressure_pa)
    dynamic_viscosity_ns_m2 = calc_standard_atmosphere_dynamic_viscosity(temperature_k)
    speed_of_sound_m_s = calc_standard_atmosphere_speed_of_sound(temperature_k)

    return AtmosState(
        geometric_altitude_m=geometric_altitude_m,
        geopotential_altitude_m=geopotential_altitude_m,
        temperature_k=temperature_k,
        pressure_pa=pressure_pa,
        density_kg_m3=density_kg_m3,
        speed_of_sound_m_s=speed_of_sound_m_s,
        dynamic_viscosity_ns_m2=dynamic_viscosity_ns_m2,
    )
