"""
Blade Element Momentum Theory (BEMT) analysis for rotors - Corrected Implementation
"""
from typing import List, Dict, Any, Tuple
import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import trapezoid as trapz
from scipy.special import ellipk
from scipy.optimize import brentq
from ductedfanlib.geometry.airfoils import Airfoil

# Constants
DEFAULT_A_INITIAL = 0.1
DEFAULT_AP_INITIAL = 0.01
DEFAULT_CONVERGENCE_TOL_FSOLVE = 1e-7
DEFAULT_CLA_2PI = 2 * np.pi
DEFAULT_MIN_VELOCITY = 1e-8  # Minimum velocity to avoid division by zero

class BEMTAnalysisError(Exception):
    """Custom exception for BEMT analysis errors."""
    pass


def _calculate_f_root_exponent(num_blades: int, r_norm: float, r_hub_norm: float, phi_rad: float) -> float:
    sin_phi_safe = max(1e-4, abs(np.sin(phi_rad)))
    if r_norm <= r_hub_norm + 1e-6 or r_hub_norm <= 1e-6:
        return 1000.0
    return (num_blades / 2.0) * ((r_norm - r_hub_norm) / (r_hub_norm * sin_phi_safe))

def _calculate_F_root(num_blades: int, r_norm: float, r_hub_norm: float, phi_rad: float) -> float:
    if r_norm <= r_hub_norm + 1e-6:
        return 0.01
    f_r_exp = _calculate_f_root_exponent(num_blades, r_norm, r_hub_norm, phi_rad)
    f_r_exp_clipped = np.clip(f_r_exp, 0, 700)
    F_r = (2.0 / np.pi) * np.arccos(np.exp(-f_r_exp_clipped))
    return np.nan_to_num(F_r, nan=0.01)

def _calculate_F_gap_simplified_prandtl_tip(num_blades: int, r_norm: float, phi_rad: float) -> float:
    sin_phi_safe = max(1e-4, abs(np.sin(phi_rad)))
    if r_norm >= 1.0 - 1e-6:
        return 0.01
    f_tip_exp_arg = (num_blades / 2.0) * (1.0 - r_norm) / sin_phi_safe
    f_tip_exp_arg_clipped = np.clip(abs(f_tip_exp_arg), 0, 700)
    F_tip = (2.0 / np.pi) * np.arccos(np.exp(-f_tip_exp_arg_clipped))
    return np.nan_to_num(F_tip, nan=0.01)

def _calculate_f_and_g_for_Fgap_dayhoum(num_blades, r_norm, tip_radius_m, tip_gap_clearance_m, phi_rad):
    sin_phi_safe = np.sin(phi_rad)
    if abs(sin_phi_safe) < 1e-4:
        sin_phi_safe = 1e-4 * np.sign(sin_phi_safe) if sin_phi_safe != 0 else 1e-4
    if tip_radius_m < 1e-6:
        raise ValueError("Tip radius must be positive.")
    f_val = (num_blades / 2.0) * (1.0 - r_norm) / sin_phi_safe
    g_val = (num_blades * tip_gap_clearance_m) / (2.0 * tip_radius_m * sin_phi_safe)
    return f_val, g_val

def _solve_cd_inverse_numerical(X_target, m_param, K_val):
    X_clipped = np.clip(X_target, 0.0, 1.0)
    if np.isclose(X_clipped, 1.0, atol=1e-9):
        return 0.0
    if np.isclose(X_clipped, 0.0, atol=1e-9):
        return K_val

    def func(u):
        from scipy.special import ellipj
        _, cn, dn, _ = ellipj(u, m_param)
        return cn / dn - X_clipped

    try:
        return brentq(func, 0.0, K_val, xtol=1e-7, rtol=1e-7)
    except (ValueError, RuntimeError):
        return K_val * (1.0 - X_clipped)

def _calculate_F_gap_dayhoum(num_blades, r_norm, tip_radius_m, tip_gap_clearance_m, phi_rad):
    if tip_gap_clearance_m <= 1e-7:
        return 1.0
    f_val, g_val = _calculate_f_and_g_for_Fgap_dayhoum(num_blades, r_norm, tip_radius_m, tip_gap_clearance_m, phi_rad)
    if abs(g_val) <= 1e-7:
        return 1.0

    sech_g_val = 1.0 / np.cosh(g_val)
    m_param = sech_g_val**2
    if m_param >= 1.0 - 1e-9:
        return 0.999

    try:
        K_val = ellipk(m_param)
    except Exception:
        return 0.5

    if not np.isfinite(K_val) or K_val <= 1e-7:
        return 0.5

    X_arg = np.cosh(g_val) / np.cosh(f_val + g_val)
    u_cd_inv = _solve_cd_inverse_numerical(X_arg, m_param, K_val)
    return np.clip(u_cd_inv / K_val, 0.01, 1.0)

def _calculate_F_sh(num_blades: int, r_norm: float, r_hub_norm: float, tip_radius_m: float,
                   tip_gap_clearance_m: float, phi_rad: float,
                   use_dayhoum_F_gap_model: bool = True) -> float:
    F_r = _calculate_F_root(num_blades, r_norm, r_hub_norm, phi_rad)
    if use_dayhoum_F_gap_model:
        F_g = _calculate_F_gap_dayhoum(num_blades, r_norm, tip_radius_m, tip_gap_clearance_m, phi_rad)
    else:
        F_g = _calculate_F_gap_simplified_prandtl_tip(num_blades, r_norm, phi_rad)
    return np.clip(F_r * F_g, 0.01, 1.0)


def _solve_element_induction_factors_axial(
    V_axial_ms: float, omega_rads: float, radius_m: float, chord_m: float, twist_deg: float,
    airfoil_obj: Airfoil, num_blades: int, rho_kgm3: float, mu_Pas: float,
    root_radius_m: float, tip_radius_m: float, tip_gap_clearance_m: float,
    use_dayhoum_F_gap_model: bool
) -> Tuple[float, float, float, float, float, float, float, float, float]:
    """
    Solves for axial (a) and tangential (a') induction factors for axial flight.
    Returns: (a, a_prime, local_phi_rad, local_alpha_deg, Cl, Cd, Re, F_sh_final, local_W_ms)
    """
    # Handle near-zero velocity to avoid division issues
    V_axial_ms_safe = V_axial_ms if abs(V_axial_ms) > DEFAULT_MIN_VELOCITY else (
        DEFAULT_MIN_VELOCITY if V_axial_ms >= 0 else -DEFAULT_MIN_VELOCITY
    )

    local_twist_rad = np.radians(twist_deg)
    if radius_m > 1e-6:
        local_solidity = (num_blades * chord_m) / (2 * np.pi * radius_m)
    else:
        local_solidity = np.inf

    # Determine mode based on velocity direction
    propeller_mode = V_axial_ms_safe < 0

    # Set initial guess based on mode
    if propeller_mode:
        a_initial = -0.1  # Negative induction for acceleration
    else:
        a_initial = 0.1   # Positive induction for deceleration
    initial_guess = [a_initial, DEFAULT_AP_INITIAL]

    def residuals(p: np.ndarray) -> np.ndarray:
        a, a_prime = p[0], p[1]
        a = np.clip(a, -0.99, 0.99)  # Avoid non-physical values

        # Calculate local flow conditions
        V_axial_element = V_axial_ms_safe * (1 + a)
        V_tangential_element = omega_rads * radius_m * (1 - a_prime)
        phi_rad = np.arctan2(V_axial_element, V_tangential_element)
        W_ms = np.sqrt(V_axial_element**2 + V_tangential_element**2)
        if np.isclose(W_ms, 0.0):
            return np.array([0.0, 0.0])

        # Blade element aerodynamics
        alpha_rad = phi_rad - local_twist_rad
        alpha_deg = np.degrees(alpha_rad)
        Re = max(1e3, (rho_kgm3 * W_ms * chord_m) / mu_Pas) if mu_Pas > 1e-9 else 1e5
        try:
            cl, cd = airfoil_obj.get_lift_drag_coeffs(alpha_deg, Re, apply_viterna_poststall=True)
        except Exception:
            cl, cd = 0.0, 0.0

        cn = cl * np.cos(phi_rad) + cd * np.sin(phi_rad)
        ct = cl * np.sin(phi_rad) - cd * np.cos(phi_rad)

        # Loss factor
        r_norm = radius_m / tip_radius_m
        r_hub_norm = root_radius_m / tip_radius_m
        F = _calculate_F_sh(num_blades, r_norm, r_hub_norm, tip_radius_m,
                           tip_gap_clearance_m, phi_rad, use_dayhoum_F_gap_model)

        # Unified momentum equation (works for both propeller and wind turbine modes)
        V_disk = V_axial_ms_safe * (1 + a)
        momentum_force = 4 * np.pi * radius_m * F * abs(V_disk) * (V_disk - V_axial_ms_safe)
        blade_force = 0.5 * rho_kgm3 * (W_ms**2) * num_blades * chord_m * cn
        res_a = momentum_force - blade_force

        # Tangential residual
        term_ap = 4 * F * abs(np.sin(phi_rad)) * np.cos(phi_rad)
        res_ap = term_ap * a_prime * (1 - a_prime) - local_solidity * ct

        return np.array([res_a, res_ap])

    try:
        solution, infodict, ier, mesg = fsolve(
            residuals, initial_guess, full_output=True, xtol=DEFAULT_CONVERGENCE_TOL_FSOLVE
        )
        a_final, ap_final = solution[0], solution[1]
    except Exception:
        a_final, ap_final = initial_guess

    # Recalculate final values with converged a, a_prime
    Vax_elem = V_axial_ms_safe * (1 + a_final)
    Vtan_elem = omega_rads * radius_m * (1 - ap_final)
    phi_rad = np.arctan2(Vax_elem, Vtan_elem)
    local_W_ms = np.sqrt(Vax_elem**2 + Vtan_elem**2)
    local_alpha_rad = phi_rad - local_twist_rad
    local_alpha_deg = np.degrees(local_alpha_rad)
    Re = max(1e3, (rho_kgm3 * local_W_ms * chord_m) / mu_Pas) if mu_Pas > 1e-9 else 1e5
    try:
        cl, cd = airfoil_obj.get_lift_drag_coeffs(local_alpha_deg, Re)
    except Exception:
        cl, cd = 0.0, 0.0

    r_norm = radius_m / tip_radius_m
    r_hub_norm = root_radius_m / tip_radius_m
    F_sh = _calculate_F_sh(num_blades, r_norm, r_hub_norm, tip_radius_m,
                          tip_gap_clearance_m, phi_rad, use_dayhoum_F_gap_model)

    return (a_final, ap_final, phi_rad, local_alpha_deg, cl, cd, Re, F_sh, local_W_ms)


# --- Main BEMT Functions ---
def calculate_bemt_performance_axial(
    rotor_stations_data: List[Dict[str, Any]],
    V_axial_ms: float, omega_rads: float, num_blades: int, rho_kgm3: float, mu_Pas: float,
    root_radius_m: float, tip_radius_m: float, tip_gap_clearance_m: float,
    use_dayhoum_F_gap_model: bool = True
) -> Dict[str, Any]:
    """
    Performs BEMT analysis for a rotor in AXIAL FLIGHT (including hover).
    Solves for induction factors a and a' at each element.
    """
    if not rotor_stations_data:
        raise ValueError("rotor_stations_data is empty.")
    if omega_rads <= 0:
        raise ValueError("omega_rads must be positive.")
    if tip_radius_m <= root_radius_m:
        raise ValueError("tip_radius_m must be greater than root_radius_m.")

    element_results = []
    integrand_dT_dr, integrand_dQ_dr, integration_radii = [], [], []

    for station in rotor_stations_data:
        r_m = station['radius_m']
        if r_m < root_radius_m - 1e-6 or r_m > tip_radius_m + 1e-6:
            continue

        chord_m = station['chord_m']
        twist_deg = station.get('base_twist_deg', station['twist_deg']) + station.get('collective_pitch_deg', 0.0)
        airfoil: Airfoil = station['airfoil_object']

        a, ap, phi_rad, alpha_deg, cl, cd, Re, F, W_ms = _solve_element_induction_factors_axial(
            V_axial_ms, omega_rads, r_m, chord_m, twist_deg, airfoil, num_blades,
            rho_kgm3, mu_Pas, root_radius_m, tip_radius_m, tip_gap_clearance_m,
            use_dayhoum_F_gap_model
        )

        cn = cl * np.cos(phi_rad) + cd * np.sin(phi_rad)
        ct = cl * np.sin(phi_rad) - cd * np.cos(phi_rad)

        dT_dr_val = 0.5 * rho_kgm3 * (W_ms**2) * num_blades * chord_m * cn * F
        dQ_dr_val = 0.5 * rho_kgm3 * (W_ms**2) * num_blades * chord_m * ct * r_m * F

        integrand_dT_dr.append(dT_dr_val)
        integrand_dQ_dr.append(dQ_dr_val)
        integration_radii.append(r_m)

        element_results.append({
            "radius_m": r_m,
            "eta": (r_m - root_radius_m) / (tip_radius_m - root_radius_m) if (tip_radius_m > root_radius_m) else 0.5,
            "a": a, "a_prime": ap, "phi_deg": np.degrees(phi_rad), "alpha_deg": alpha_deg,
            "Cl": cl, "Cd": cd, "Re": Re, "F_sh": F, "W_ms": W_ms,
            "dT_dr_N_m": dT_dr_val, "dQ_dr_Nm_m": dQ_dr_val
        })

    # Integration and result packaging
    if not integration_radii:
        return {
            "total_thrust_N": 0.0, "total_torque_Nm": 0.0, "total_power_W": 0.0,
            "efficiency": 0.0, "Ct_rotor": 0.0, "Cp_rotor": 0.0, "spanwise_results": []
        }

    # Sort by radius for proper integration
    sort_indices = np.argsort(integration_radii)
    radii_np = np.array(integration_radii)[sort_indices]
    dTdr_np = np.array(integrand_dT_dr)[sort_indices]
    dQdr_np = np.array(integrand_dQ_dr)[sort_indices]

    # Calculate total thrust and torque
    total_thrust_N = trapz(dTdr_np, x=radii_np)
    total_torque_Nm = trapz(dQdr_np, x=radii_np)
    total_power_W = total_torque_Nm * omega_rads

    # Calculate advance ratio and dimensionless coefficients
    n_rps = omega_rads / (2 * np.pi)
    D = 2 * tip_radius_m
    if n_rps > 1e-6:
        J = V_axial_ms / (n_rps * D)  # Advance ratio
    else:
        J = 0.0

    # Calculate efficiency
    if abs(total_power_W) > 1e-6:
        efficiency = (total_thrust_N * V_axial_ms) / total_power_W
        efficiency = np.clip(efficiency, -1.0, 1.0)
    else:
        efficiency = 0.0

    # Calculate dimensionless coefficients
    if n_rps > 1e-6:
        Ct = total_thrust_N / (rho_kgm3 * n_rps**2 * D**4)
        Cp = total_power_W / (rho_kgm3 * n_rps**3 * D**5)
        Cq = total_torque_Nm / (rho_kgm3 * n_rps**2 * D**5)
    else:  # Hover condition
        Ct = total_thrust_N / (rho_kgm3 * omega_rads**2 * tip_radius_m**4)
        Cp = total_power_W / (rho_kgm3 * omega_rads**3 * tip_radius_m**5)
        Cq = Cp  # For hover, Cq = Cp since P = Q*ω

    # Sort spanwise results
    final_spanwise_results = [element_results[i] for i in sort_indices] if element_results else []

    return {
        "total_thrust_N": total_thrust_N,
        "total_torque_Nm": total_torque_Nm,
        "total_power_W": total_power_W,
        "efficiency": efficiency,
        "Ct_rotor": Ct,
        "Cp_rotor": Cp,
        "Cq_rotor": Cq,
        "advance_ratio": J,
        "spanwise_results": final_spanwise_results
    }