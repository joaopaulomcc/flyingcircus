from pathlib import Path

import context

import numpy as np
import matplotlib.pyplot as plt

from flyingcircus.atmosphere import (
    calc_standard_atmosphere_temperature,
    calc_standard_atmosphere_pressure,
    calc_standard_atmosphere,
)
import flyingcircus.constants as constants


def test_calc_standard_atmosphere_temperature() -> None:

    print("Geopotential Altitude [m] vs Temperature [K]")
    for altitude_m in constants.STANDARD_ATMOSPHERE_TEMPERATURE_GRADIENTS_K_M.keys():

        print(f"{altitude_m:6d} : {calc_standard_atmosphere_temperature(altitude_m):.3f}")

    geopotential_altitudes_m = np.linspace(0, 84850, 1001)
    temperatures_k = [
        calc_standard_atmosphere_temperature(altitude_m) for altitude_m in geopotential_altitudes_m
    ]

    _, ax = plt.subplots()
    ax.plot(temperatures_k, geopotential_altitudes_m)
    ax.set(
        title="Standard Atmosphere - Temperature vs Geopotential Altitude",
        xlabel="Temperature [K]",
        ylabel="Geopotential Altitude [m]",
    )
    ax.grid(True, axis="both")
    plt.show()


def test_calc_standard_atmosphere_pressure() -> None:

    print("Geopotential Altitude [m] vs Temperature [K]")
    for altitude_m in constants.STANDARD_ATMOSPHERE_TEMPERATURE_GRADIENTS_K_M.keys():

        print(f"{altitude_m:6d} : {calc_standard_atmosphere_pressure(altitude_m):.3f}")

    geopotential_altitudes_m = np.linspace(0, 84850, 1001)
    pressures_pa = [
        calc_standard_atmosphere_pressure(altitude_m) for altitude_m in geopotential_altitudes_m
    ]

    _, ax = plt.subplots()
    ax.semilogx(pressures_pa, geopotential_altitudes_m)
    ax.set(
        title="Standard Atmosphere - Pressure vs Geopotential Altitude",
        xlabel="Pressure [Pa]",
        ylabel="Geopotential Altitude [m]",
    )
    ax.grid(True, axis="both")
    plt.show()


def test_calc_standard_atmosphere() -> None:

    output_filepath = Path("tests/standard_atmosphere.csv")
    output_filepath.write_text(
        ",".join(
            [
                "Geometric Altitude [m]",
                "Geopotential Altitude [m]",
                "Temperature [K]",
                "Pressure [Pa]",
                "Density [kg/m^3]",
                "Speed of Sound [m/s]",
                "Dynamic Viscosity [N.S/m^2]",
                "\n",
            ]
        )
    )

    for altitude_m in range(0, 86_500, 500):

        atmos_state = calc_standard_atmosphere(altitude_m, temperature_offset_k=0)

        with output_filepath.open("a") as output_file:
            output_file.write(
                ",".join(
                    [
                        f"{atmos_state.geometric_altitude_m:.0f}",
                        f"{atmos_state.geopotential_altitude_m:.0f}",
                        f"{atmos_state.temperature_k:.3f}",
                        f"{atmos_state.pressure_pa:.4e}",
                        f"{atmos_state.density_kg_m3:.4e}",
                        f"{atmos_state.speed_of_sound_m_s:.2f}",
                        f"{atmos_state.dynamic_viscosity_ns_m2:.4e}",
                        "\n"
                    ]
                ),
            )

    geometric_altitudes_m = np.linspace(0, 86000, 1001)
    geopotential_altitudes_m = []
    temperatures_k = []
    pressures_pa = []
    densitys_kg_m3 = []
    speeds_of_sound_m_s = []
    dynamic_viscositys_ns_m2 = []

    for geometric_altitude_m in geometric_altitudes_m:

        atmos_state = calc_standard_atmosphere(geometric_altitude_m, temperature_offset_k=15)

        geopotential_altitudes_m.append(atmos_state.geopotential_altitude_m)
        temperatures_k.append(atmos_state.temperature_k)
        pressures_pa.append(atmos_state.pressure_pa)
        densitys_kg_m3.append(atmos_state.density_kg_m3)
        speeds_of_sound_m_s.append(atmos_state.speed_of_sound_m_s)
        dynamic_viscositys_ns_m2.append(atmos_state.dynamic_viscosity_ns_m2)

    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Standard Atmosphere 1976", fontsize=16)
    ax[0, 0].plot(geopotential_altitudes_m, geometric_altitudes_m)
    ax[0, 0].set(
        title="Geometric Altitude vs Geopotential Altitude",
        xlabel="Geopotential Altitude [m]",
        ylabel="Geometric Altitude [m]",
    )
    ax[0, 0].grid(True, axis="both")

    ax[0, 1].plot(temperatures_k, geometric_altitudes_m)
    ax[0, 1].set(
        title="Geometric Altitude vs Temperature",
        xlabel="Temperature [K]",
        ylabel="Geometric Altitude [m]",
    )
    ax[0, 1].grid(True, axis="both")

    ax[0, 2].semilogx(pressures_pa, geometric_altitudes_m)
    ax[0, 2].set(
        title="Geometric Altitude vs Pressure",
        xlabel="Pressure [Pa]",
        ylabel="Geometric Altitude [m]",
    )
    ax[0, 2].grid(True, axis="both")

    ax[1, 0].semilogx(densitys_kg_m3, geometric_altitudes_m)
    ax[1, 0].set(
        title="Geometric Altitude vs Density",
        xlabel="Density [kg / m^3]",
        ylabel="Geometric Altitude [m]",
    )
    ax[1, 0].grid(True, axis="both")

    ax[1, 1].plot(speeds_of_sound_m_s, geometric_altitudes_m)
    ax[1, 1].set(
        title="Geometric Altitude vs Speed of Sound",
        xlabel="Speed of Sound [m/s]",
        ylabel="Geometric Altitude [m]",
    )
    ax[1, 1].grid(True, axis="both")

    ax[1, 2].plot(dynamic_viscositys_ns_m2, geometric_altitudes_m)
    ax[1, 2].set(
        title="Geometric Altitude vs Dynamic Viscosity",
        xlabel="Dynamic Viscosity [N.s/m^2]",
        ylabel="Geometric Altitude [m]",
    )
    ax[1, 2].grid(True, axis="both")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_calc_standard_atmosphere_temperature()
    test_calc_standard_atmosphere_pressure()
    test_calc_standard_atmosphere()
