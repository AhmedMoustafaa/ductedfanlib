import numpy as np
import math
from ..utils.constants import AIR_DENSITY_STANDARD_SEA_LEVEL  # Relative import within ductedfanlib


def calculate_ideal_performance_hover(
        fan_diameter: float,
        thrust_target: float = None,
        power_target: float = None,
        air_density: float = AIR_DENSITY_STANDARD_SEA_LEVEL
) -> dict:
    """
    Calculates ideal ducted fan performance in hover (static thrust)
    using 1D momentum theory (Actuator Disk Theory).
    """
    if (thrust_target is None and power_target is None) or \
            (thrust_target is not None and power_target is not None):
        raise ValueError("Please provide either 'thrust_target' or 'power_target', but not both.")

    fan_radius = fan_diameter / 2.0
    disk_area = np.pi * (fan_radius ** 2)

    if thrust_target is not None:
        if thrust_target < 0:
            raise ValueError("Thrust target must be non-negative.")
        if thrust_target == 0:
            induced_velocity = 0
            power = 0
        else:
            induced_velocity = math.sqrt(thrust_target / (2 * air_density * disk_area))
            power = thrust_target * induced_velocity
        thrust = thrust_target
    else:  # power_target is not None
        if power_target < 0:
            raise ValueError("Power target must be non-negative.")
        if power_target == 0:
            induced_velocity = 0
            thrust = 0
        else:
            induced_velocity = (power_target / (air_density * disk_area * 2)) ** (1 / 3)
            thrust = 2 * air_density * disk_area * (induced_velocity ** 2)
        power = power_target

    if power > 0:
        figure_of_merit = (thrust * math.sqrt(thrust / (2 * air_density * disk_area))) / power
    else:
        figure_of_merit = 0.0

    mass_flow_rate = air_density * disk_area * induced_velocity

    return {
        'thrust': thrust,
        'power': power,
        'induced_velocity': induced_velocity,
        'disk_area': disk_area,
        'figure_of_merit': figure_of_merit,
        'mass_flow_rate': mass_flow_rate
    }


def calculate_ideal_performance_axial(
        fan_diameter: float,
        freestream_velocity: float,
        thrust_target: float = None,
        power_target: float = None,
        air_density: float = AIR_DENSITY_STANDARD_SEA_LEVEL
) -> dict:
    """
    Calculates ideal ducted fan performance in axial forward flight
    using 1D momentum theory (Actuator Disk Theory).
    """
    if (thrust_target is None and power_target is None) or \
            (thrust_target is not None and power_target is not None):
        raise ValueError("Please provide either 'thrust_target' or 'power_target', but not both.")

    fan_radius = fan_diameter / 2.0
    disk_area = np.pi * (fan_radius ** 2)

    if freestream_velocity < 0:
        raise ValueError("Freestream velocity cannot be negative (e.g., must be > 0 for forward flight).")

    if freestream_velocity == 0:
        return calculate_ideal_performance_hover(fan_diameter, thrust_target, power_target, air_density)

    if thrust_target is not None:
        if thrust_target < 0:
            raise ValueError("Thrust target must be non-negative.")
        term_sqrt = (freestream_velocity / 2.0) ** 2 + thrust_target / (2 * air_density * disk_area)
        if term_sqrt < 0:
            raise ValueError("Invalid parameters leading to complex induced velocity.")
        induced_velocity = -freestream_velocity / 2.0 + math.sqrt(term_sqrt)

        total_velocity_through_disk = freestream_velocity + induced_velocity
        power = thrust_target * total_velocity_through_disk
        thrust = thrust_target

    else:  # power_target is not None
        if power_target < 0:
            raise ValueError("Power target must be non-negative.")
        a = air_density * disk_area
        b = air_density * disk_area * freestream_velocity
        c = -power_target

        discriminant = b ** 2 - 4 * a * c
        if discriminant < 0:
            raise ValueError("Invalid parameters leading to complex induced velocity (discriminant < 0).")

        induced_velocity = (-b + math.sqrt(discriminant)) / (2 * a)

        total_velocity_through_disk = freestream_velocity + induced_velocity
        thrust = air_density * disk_area * total_velocity_through_disk * induced_velocity
        power = power_target

    if power > 0:
        propulsive_efficiency = (thrust * freestream_velocity) / power
    else:
        propulsive_efficiency = 0.0

    mass_flow_rate = air_density * disk_area * total_velocity_through_disk

    return {
        'thrust': thrust,
        'power': power,
        'induced_velocity': induced_velocity,
        'total_velocity_through_disk': total_velocity_through_disk,
        'disk_area': disk_area,
        'propulsive_efficiency': propulsive_efficiency,
        'mass_flow_rate': mass_flow_rate
    }