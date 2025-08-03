"""
Basic sanity checks for the BEMT implementation.
Ensures the BEMT function runs, produces outputs of the correct type,
and that basic physical trends are observed.
"""
import numpy as np
import pytest
import sys
import os

# Ensure ductedfanlib is importable
script_dir = os.path.dirname(os.path.abspath(__file__))  # tests/
project_root = os.path.dirname(script_dir)  # ductedfan_project_root/
src_dir = os.path.join(project_root, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from ductedfanlib.core import Blade, Rotor
from ductedfanlib.geometry.airfoils import Airfoil, generate_naca4_coordinates
from ductedfanlib.geometry.profiles import ConstantDistribution
from ductedfanlib.geometry.meshing import get_rotor_bemt_stations
from ductedfanlib.analysis.bemt import calculate_bemt_performance_dayhoum, BEMTAnalysisError


@pytest.fixture(scope="module")
def basic_open_rotor_setup():
    """Sets up a simple open rotor configuration for testing BEMT."""
    # 1. Airfoil with simplified, predictable polar
    naca0012_geom = generate_naca4_coordinates("0012", num_points_per_surface=80)
    test_airfoil = Airfoil(name="NACA0012_TestLinearPolar", coordinates=naca0012_geom.coordinates)

    # Define a simple linear polar: CL = 0.1 * alpha_deg, CD = 0.01 + 0.005 * CL^2
    # (This is more stable than 2*pi*alpha which can lead to very high CL quickly)
    alpha_polar_deg = np.linspace(-10, 20, 31)  # degrees
    cl_polar = 0.1 * alpha_polar_deg
    cd_polar = 0.001 + 0.005 * (cl_polar ** 2)
    cm_polar = -0.01 * np.ones_like(alpha_polar_deg)  # Constant Cm for completeness

    test_airfoil.add_polar(Re=1e6, alpha_deg=alpha_polar_deg, cl=cl_polar, cd=cd_polar, cm=cm_polar)
    test_airfoil.characterize_stall_properties(Re=1e6, analysis_method="polar")  # Use loaded polar

    # 2. Blade Definition
    # Constant chord, constant simple positive twist (pitch)
    chord_val = 0.03  # m
    twist_val = 8.0  # degrees (effective pitch)

    blade_chord = ConstantDistribution(value=chord_val)
    blade_twist = ConstantDistribution(value=twist_val)

    test_blade = Blade(
        airfoil_definition=test_airfoil,
        chord_profile=blade_chord,
        twist_profile=blade_twist
    )

    # 3. Rotor Definition
    rotor_params = {
        "num_blades": 2,
        "tip_radius": 0.20,  # m
        "hub_radius": 0.04,  # m (20%R)
        "blade_definition": test_blade,
        "collective_pitch_deg": 0.0  # Use blade's constant twist as pitch
    }
    test_rotor = Rotor(**rotor_params)

    # 4. Operating Conditions (Hover)
    op_conditions = {
        "omega_rads": 2500 * (2 * np.pi) / 60,  # RPM to rad/s
        "rho_kgm3": 1.225,
        "mu_Pas": 1.81e-5,
        "V_axial_ms_hover": 0.0  # Used conceptually by Dayhoum BEMT for hover mode
    }

    # 5. BEMT Configuration for Open Rotor Hover
    bemt_config_open_rotor = {
        "sigma_d_sq": 1.0,
        "tip_gap_clearance_m": 0.0,  # No physical gap for open rotor Prandtl model
        "use_dayhoum_F_gap_model": True,  # Use Prandtl tip/hub
        "assumed_cl_alpha_rad": test_airfoil.get_lift_curve_slope(Re=1e6, alpha0_deg=0.0, analysis_method="polar")
    }

    # 6. Rotor Stations
    num_stations = 11  # Fewer stations for a quick test
    rotor_stations = get_rotor_bemt_stations(
        test_rotor,
        num_stations=num_stations,
        spacing_type="cosine"
    )

    return {
        "rotor": test_rotor,
        "op_conditions": op_conditions,
        "bemt_config": bemt_config_open_rotor,
        "rotor_stations": rotor_stations
    }


def test_bemt_runs_without_errors(basic_open_rotor_setup):
    """Test if the BEMT calculation completes without raising an unhandled exception."""
    setup = basic_open_rotor_setup
    try:
        results = calculate_bemt_performance_dayhoum(
            rotor_stations_data=setup["rotor_stations"],
            omega_rads=setup["op_conditions"]["omega_rads"],
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
        assert results is not None, "BEMT results should not be None"
        print("\nTest 'test_bemt_runs_without_errors' PASSED: BEMT ran.")
        print(
            f"  Thrust: {results['total_rotor_thrust_N']:.2f} N, Power: {results['total_power_W']:.2f} W, FM: {results['figure_of_merit_hover']:.3f}")

    except Exception as e:
        pytest.fail(f"BEMT calculation raised an unexpected exception: {e}")


def test_bemt_output_structure_and_types(basic_open_rotor_setup):
    """Test the structure and data types of the BEMT results dictionary."""
    setup = basic_open_rotor_setup
    results = calculate_bemt_performance_dayhoum(
        rotor_stations_data=setup["rotor_stations"],
        omega_rads=setup["op_conditions"]["omega_rads"],
        num_blades=setup["rotor"].num_blades,
        rho_kgm3=setup["op_conditions"]["rho_kgm3"],
        mu_Pas=setup["op_conditions"]["mu_Pas"],
        root_radius_m=setup["rotor"].hub_radius,
        tip_radius_m=setup["rotor"].tip_radius,
        **setup["bemt_config"]  # Pass other config items
    )

    assert isinstance(results, dict), "Results should be a dictionary."

    expected_keys = [
        "total_rotor_thrust_N", "total_torque_Nm", "total_power_W",
        "figure_of_merit_hover", "Ct_rotor", "Cp_rotor", "spanwise_results"
    ]
    for key in expected_keys:
        assert key in results, f"Key '{key}' missing in BEMT results."

    assert isinstance(results["total_rotor_thrust_N"], float), "Thrust should be a float."
    assert isinstance(results["total_power_W"], float), "Power should be a float."
    assert isinstance(results["figure_of_merit_hover"], float), "FM should be a float."
    assert isinstance(results["spanwise_results"], list), "Spanwise results should be a list."

    if results["spanwise_results"]:
        assert len(results["spanwise_results"]) == len(setup["rotor_stations"]), \
            "Number of spanwise result entries should match number of stations."
        first_station_res = results["spanwise_results"][0]
        assert isinstance(first_station_res, dict), "Each spanwise result should be a dict."
        # Check a few keys in the spanwise results
        expected_spanwise_keys = ["radius_m", "eta", "lambda", "phi_deg", "alpha_deg", "Cl", "Cd", "F_sh"]
        for skey in expected_spanwise_keys:
            assert skey in first_station_res, f"Key '{skey}' missing in spanwise results."

    print("\nTest 'test_bemt_output_structure_and_types' PASSED: Output structure and types are correct.")


def test_bemt_basic_physical_plausibility(basic_open_rotor_setup):
    """Check for basic physical plausibility of BEMT results."""
    setup = basic_open_rotor_setup
    results = calculate_bemt_performance_dayhoum(
        rotor_stations_data=setup["rotor_stations"],
        omega_rads=setup["op_conditions"]["omega_rads"],
        num_blades=setup["rotor"].num_blades,
        rho_kgm3=setup["op_conditions"]["rho_kgm3"],
        mu_Pas=setup["op_conditions"]["mu_Pas"],
        root_radius_m=setup["rotor"].hub_radius,
        tip_radius_m=setup["rotor"].tip_radius,
        **setup["bemt_config"]
    )

    # For a positive pitch/twist and positive RPM, expect positive thrust and power
    assert results["total_rotor_thrust_N"] > 0, "Thrust should be positive for this setup."
    assert results["total_torque_Nm"] > 0, "Torque should be positive for this setup."
    assert results["total_power_W"] > 0, "Power should be positive for this setup."

    # Figure of Merit should be between 0 and 1 (inclusive)
    assert 0.0 <= results["figure_of_merit_hover"] <= 1.0, "Figure of Merit should be between 0 and 1."

    # Check some spanwise results plausibility
    if results["spanwise_results"]:
        for station_res in results["spanwise_results"]:
            assert -90 < station_res["phi_deg"] < 90, f"Inflow angle phi={station_res['phi_deg']} seems implausible."
            # Alpha might go high, but let's check it's not excessively so across the whole blade
            assert -45 < station_res[
                "alpha_deg"] < 45, f"Angle of attack alpha={station_res['alpha_deg']} seems very large/small."
            assert 0.0 < station_res["F_sh"] <= 1.0, f"Loss factor F_sh={station_res['F_sh']} is out of [0,1] bounds."
            # Lambda (inflow ratio) should generally be positive for hover thrust generation
            assert station_res[
                       "lambda"] >= 0, f"Inflow ratio lambda={station_res['lambda']} should be non-negative for hover thrust."

    print("\nTest 'test_bemt_basic_physical_plausibility' PASSED: Results are physically plausible.")


if __name__ == "__main__":
    # This allows running the tests directly with `python tests/test_bemt_basic_run.py`
    # although `pytest` is the recommended way.
    print("--- Running BEMT Basic Sanity Tests ---")

    # Create the setup once
    setup_data = basic_open_rotor_setup()

    # Call test functions
    try:
        test_bemt_runs_without_errors(setup_data)
        test_bemt_output_structure_and_types(setup_data)
        test_bemt_basic_physical_plausibility(setup_data)

        print("\nALL BASIC BEMT TESTS FINISHED.")
    except AssertionError as e:
        print(f"\nBASIC BEMT TEST FAILED: {e}")
    except Exception as e:
        print(f"\nAN ERROR OCCURRED DURING BASIC BEMT TESTS: {e}")