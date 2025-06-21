"""
Basic sanity checks for the BEMT implementation.
Ensures the BEMT function runs, produces outputs of the correct type,
and that basic physical trends are observed.
"""
import numpy as np
import matplotlib.pyplot as plt

from ductedfanlib.core import Blade, Rotor
from ductedfanlib.geometry.airfoils import Airfoil, generate_naca4_coordinates
from ductedfanlib.geometry.profiles import ConstantDistribution, LinearDistribution
from ductedfanlib.geometry.meshing import get_rotor_bemt_stations
from ductedfanlib.analysis.bemt import calculate_bemt_performance_dayhoum, BEMTAnalysisError

# --- Test Setup ---
"""Sets up a simple open rotor configuration for testing BEMT."""
# 1. Airfoil with simplified, predictable polar
naca2412_geom = generate_naca4_coordinates("2412", num_points_per_surface=200)
test_airfoil = Airfoil(name="NACA2412_TestLinearPolar", coordinates=naca2412_geom.coordinates)


blade_chord = ConstantDistribution(0.4)
blade_twist = ConstantDistribution(5)
test_blade = Blade(
    airfoil_definition=test_airfoil,
    chord_profile=blade_chord,
    twist_profile=blade_twist
)

# 3. Rotor Definition
rotor_params = {
    "num_blades": 3,
    "tip_radius": 2.5,  # m
    "hub_radius": 0.25,  # m
    "blade_definition": test_blade,
    "collective_pitch_deg": 0.0,
}
test_rotor = Rotor(**rotor_params)

# 4. Operating Conditions (Hover)
op_conditions = {
    "omega_rads": 2500 * 2*np.pi/60,  # RPM to rad/s
    "rho_kgm3": 1.225,
    "mu_Pas": 1.5e-5,
    "V_axial_ms_hover": 0
}

# 5. BEMT Configuration for Open Rotor Hover
bemt_config_open_rotor = {
    "sigma_d_sq": 1.0,
    "tip_gap_clearance_m": 0.0,  # No physical gap for open rotor Prandtl model
    "use_dayhoum_F_gap_model": False, #Use Prandtl tip/hub
    "assumed_cl_alpha_rad": test_airfoil.get_lift_curve_slope(Re=1e6)
}

# 6. Rotor Stations
num_stations = 50  # Fewer stations for a quick test
rotor_stations = get_rotor_bemt_stations(
    test_rotor,
    num_stations=num_stations,
    spacing_type="cosine"
)

setup =  {
    "rotor": test_rotor,
    "op_conditions": op_conditions,
    "bemt_config": bemt_config_open_rotor,
    "rotor_stations": rotor_stations
}
results = calculate_bemt_performance_dayhoum(
    rotor_stations_data=setup["rotor_stations"],
    omega_rads=setup["op_conditions"]["omega_rads"],
    v_inf=setup["op_conditions"]["V_axial_ms_hover"],
    num_blades=setup["rotor"].num_blades,
    rho_kgm3=setup["op_conditions"]["rho_kgm3"],
    mu_Pas=setup["op_conditions"]["mu_Pas"],
    root_radius_m=setup["rotor"].hub_radius,
    tip_radius_m=setup["rotor"].tip_radius,
    sigma_d_sq=setup["bemt_config"]["sigma_d_sq"],
    tip_gap_clearance_m=setup["bemt_config"]["tip_gap_clearance_m"],
    assumed_cl_alpha_rad=setup["bemt_config"]["assumed_cl_alpha_rad"],
    use_dayhoum_F_gap_model=setup["bemt_config"]["use_dayhoum_F_gap_model"]
)
print(f"  Thrust: {results['total_rotor_thrust_N']:.2f} N, Power: {results['total_power_W']:.2f} W, FM: {results['figure_of_merit_hover']:.3f}")
print(f"Advance Ratio = {op_conditions["V_axial_ms_hover"]/(rotor_params["tip_radius"]*2*op_conditions["omega_rads"]/2/np.pi)}")
