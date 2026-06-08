"""
High-level analysis runner — selects the right solver automatically.


"""
from __future__ import annotations

from typing import Dict, Any
import numpy as np

from .bemt2 import calculate_bemt_performance_axial
from ..core.design import DuctedFan
from ..core.OperatingConditions import OperatingConditions
from ..geometry.meshing import get_rotor_bemt_stations


def run_bemt_analysis(
    design: DuctedFan,
    op_conditions: OperatingConditions,
    num_stations: int = 21,
    use_dayhoum_model: bool = False,
) -> Dict[str, Any]:
    """
    Run BEMT on a DuctedFan for the given operating conditions.

    Both hover (V=0) and forward flight are handled by bemt2, which uses a
    universal velocity-form phi-iteration that is non-degenerate at all V.

    Parameters
    ----------
    design          : assembled DuctedFan
    op_conditions   : OperatingConditions (rpm, axial_velocity_ms, rho, mu)
    num_stations    : number of radial blade elements
    use_dayhoum_model : True for Dayhoum shrouded tip loss; False (default) for open

    Returns
    -------
    dict — see bemt2.calculate_bemt_performance_axial for key list
    """
    stations = get_rotor_bemt_stations(design.rotor, num_stations=num_stations)

    return calculate_bemt_performance_axial(
        rotor_stations_data     = stations,
        V_axial_ms              = op_conditions.axial_velocity_ms,
        omega_rads              = op_conditions.omega_rads,
        num_blades              = design.rotor.num_blades,
        rho_kgm3                = op_conditions.rho_kgm3,
        mu_Pas                  = op_conditions.mu_Pas,
        root_radius_m           = design.rotor.hub_radius,
        tip_radius_m            = design.rotor.tip_radius,
        tip_gap_clearance_m     = design.tip_clearance,
        use_dayhoum_F_gap_model = use_dayhoum_model,
    )
