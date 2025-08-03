import numpy as np
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_dir = os.path.join(project_root, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from ductedfanlib.core import Blade, Rotor, Duct, DuctedFan
from ductedfanlib.geometry.airfoils import generate_naca4_coordinates
from ductedfanlib.geometry.profiles import LinearDistribution
from ductedfanlib.geometry.curves import BezierCurve
from ductedfanlib.study import ParametricStudy, OperatingConditions


def main():
    """
    Main function to set up and run parametric studies.
    """
    # --- 1. Define the Ducted Fan Design ---
    print("--- Defining Baseline Design ---")
    naca_airfoil = generate_naca4_coordinates("2412", default_analysis_method="neuralfoil")
    naca_airfoil.characterize_stall_properties(Re=500000, analysis_method="neuralfoil")

    # Define base profiles that we can modify in the sensitivity study
    base_twist_dist = LinearDistribution(start_value=20.0, end_value=10.0)
    base_chord_dist = LinearDistribution(start_value=0.12, end_value=0.06)

    blade = Blade(airfoil_definition=naca_airfoil, chord_profile=base_chord_dist, twist_profile=base_twist_dist)
    rotor = Rotor(num_blades=4, tip_radius=0.5, hub_radius=0.1, blade_definition=blade)
    dummy_duct_profile = BezierCurve(control_points=[[0, 0.6], [1, 0.6]])
    duct = Duct(profile_curve=dummy_duct_profile)
    open_rotor_design = DuctedFan(rotor=rotor, duct=duct, tip_clearance=0.0, rotor_axial_position=0.0)

    # --- 2. Set up and Run Parametric Studies ---
    study = ParametricStudy(design=open_rotor_design)

    # === ADVANCE RATIO SWEEP ===
    j_sweep_range = np.linspace(0.1, 0.9, 17)
    study.sweep_advance_ratio(rpm=2500, j_range=j_sweep_range, use_dayhoum_model=False)

    # === SENSITIVITY ANALYSIS / DESIGN PARAMETER SWEEP ===
    # We will sweep the tip twist of the rotor blade and see its effect on performance
    # at a fixed operating condition (e.g., a specific advance ratio).
    fixed_op_cond = OperatingConditions(axial_velocity_ms=25.0, rpm=2500)  # J approx 0.48

    study.sweep_design_parameter(
        parameter_path="rotor.blade_definition.twist_profile.end_value",  # Path to tip twist
        sweep_range=np.linspace(2.0, 12.0, 11),  # Sweep tip twist from 2 to 12 degrees
        op_conditions=fixed_op_cond,
        use_dayhoum_model=False
    )

    # --- 3. View the Results ---
    if study.sweep_results_df is not None:
        print("\n--- Advance Ratio Study Results ---")
        print(study.sweep_results_df.to_string(index=False, float_format="%.4f"))

    if study.sensitivity_results_df is not None:
        print("\n--- Sensitivity Study Results (vs. Tip Twist) ---")
        print(study.sensitivity_results_df.to_string(index=False, float_format="%.4f"))

    # Plot performance curves from the advance ratio sweep
    study.plot_performance_curves()

    # Plot spanwise distributions for a few advance ratios
    study.plot_spanwise_distributions(j_values_to_plot=[0.2, 0.5, 0.8])

    # Plot streamline visualization for a specific advance ratio
    study.plot_streamtube_visualization(j_value_to_plot=0.5, num_streamlines=12)

    # Plot results of the sensitivity analysis
    study.plot_sensitivity_curves(
        parameter_path="rotor.blade_definition.twist_profile.end_value",
        metrics=['Ct', 'efficiency']
    )


if __name__ == "__main__":
    # Add pandas and matplotlib to your project dependencies!
    # pip install pandas matplotlib
    main()