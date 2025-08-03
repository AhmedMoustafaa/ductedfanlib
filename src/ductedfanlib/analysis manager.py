"""
Provides high-level, user-friendly functions to run various analyses.
This module acts as a controller that selects the appropriate low-level
solver based on the provided operating conditions.
"""
from typing import Dict, Any
import numpy as np
from .core import DuctedFan  # We'll pass the whole object
from .geometry.meshing import get_rotor_bemt_stations
from .analysis.bemt2 import (
    calculate_bemt_performance_axial,
    calculate_bemt_performance_axial,
    DEFAULT_CLA_2PI
)
from ductedfanlib.core.OperatingConditions import OperatingConditions




def run_bemt_analysis(
        design: DuctedFan,
        op_conditions: OperatingConditions,
        num_stations: int = 21,
        use_dayhoum_model: bool = True  # Flag to control shrouded model specifics
) -> Dict[str, Any]:
    """
    Runs a BEMT analysis on a DuctedFan design for given operating conditions.

    This function automatically selects the most appropriate BEMT solver:
    - If axial velocity is zero (hover), it uses the specialized Dayhoum hover model.
    - If axial velocity is non-zero, it uses the general axial flight model.

    Args:
        design (DuctedFan): The DuctedFan object to be analyzed.
        op_conditions (OperatingConditions): The operating conditions for the analysis.
        num_stations (int): Number of radial stations for blade discretization.
        use_dayhoum_model (bool):
            If True, uses the advanced Dayhoum F_gap model for shrouded rotors.
            If False, uses standard Prandtl losses (suitable for open rotors).

    Returns:
        Dict[str, Any]: A dictionary containing the performance results.
    """
    rotor_stations = get_rotor_bemt_stations(design.rotor, num_stations=num_stations)
    omega_rads = op_conditions.rpm * (2 * np.pi) / 60

    # Get a representative Cl_alpha for the models that need it
    # Use polar data if available, otherwise on-the-fly analysis
    representative_Re = 500_000  # A typical Re for cl_alpha lookup
    assumed_cla_rad = DEFAULT_CLA_2PI
    if hasattr(design.rotor.blade_definition.airfoil_definition, 'get_lift_curve_slope'):
        try:
            assumed_cla_rad = design.rotor.blade_definition.airfoil_definition.get_lift_curve_slope(
                Re=representative_Re)
        except Exception:
            pass  # Use default if it fails

    # --- Logic to select the BEMT solver ---
    if op_conditions.axial_velocity_ms == 0.0:
        print("Info: Axial velocity is 0. Using specialized hover BEMT model.")
        # The hover model needs sigma_d_sq
        # Let's assume Duct has properties for inlet/outlet diameter
        try:
            # Note: This relies on the Duct class having these derived properties
            # If not, the user would need to provide sigma_d_sq manually.
            A_rotor = np.pi * (design.rotor.tip_radius ** 2 - design.rotor.hub_radius ** 2)
            A_exit = np.pi * (design.duct.derived_outlet_diameter / 2) ** 2
            sigma_d = A_exit / A_rotor
            sigma_d_sq = sigma_d ** 2
        except (AttributeError, TypeError):
            # Fallback if duct diameters are not available
            sigma_d_sq = 1.0  # Default to 1.0 (like an open rotor or straight duct)
            print("Warning: Could not derive sigma_d from duct. Defaulting to sigma_d_sq=1.0.")

        results = _solve_bemt_hover_dayhoum(
            rotor_stations_data=rotor_stations,
            omega_rads=omega_rads,
            num_blades=design.rotor.num_blades,
            rho_kgm3=op_conditions.rho_kgm3,
            mu_Pas=op_conditions.mu_Pas,
            root_radius_m=design.rotor.hub_radius,
            tip_radius_m=design.rotor.tip_radius,
            sigma_d_sq=sigma_d_sq,
            tip_gap_clearance_m=design.tip_clearance,
            assumed_cl_alpha_rad=assumed_cla_rad,
            use_dayhoum_F_gap_model=use_dayhoum_model
        )

    else:  # Axial flight condition
        print(f"Info: Axial velocity is {op_conditions.axial_velocity_ms} m/s. Using general axial BEMT model.")
        results = _solve_bemt_axial_general(
            rotor_stations_data=rotor_stations,
            V_axial_ms=op_conditions.axial_velocity_ms,
            omega_rads=omega_rads,
            num_blades=design.rotor.num_blades,
            rho_kgm3=op_conditions.rho_kgm3,
            mu_Pas=op_conditions.mu_Pas,
            root_radius_m=design.rotor.hub_radius,
            tip_radius_m=design.rotor.tip_radius,
            tip_gap_clearance_m=design.tip_clearance,
            use_dayhoum_F_gap_model=use_dayhoum_model
        )

    return results