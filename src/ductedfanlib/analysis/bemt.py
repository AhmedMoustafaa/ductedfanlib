"""
BEMT for shrouded rotors in hover (Dayhoum et al. model).
"""
from typing import List, Dict, Any, Tuple, Callable
import numpy as np
from scipy.optimize import brentq # For cd_inverse
from scipy.integrate import trapezoid as trapz
from scipy.special import ellipk, ellipj # For F_gap

# Assuming Airfoil class is in ductedfanlib.geometry.airfoils
from ductedfanlib.geometry.airfoils import Airfoil

# --- Constants ---
#DEFAULT_PHI_INITIAL_DEG = 10.0 # Initial inflow angle guess (deg)
DEFAULT_MAX_ITERATIONS_ELEMENT = 50 # Max iterations for element solution
DEFAULT_CONVERGENCE_TOL_PHI = 1e-6 # Phi convergence tolerance (rad)
DEFAULT_CLA_2PI = 2 * np.pi # Default lift curve slope (1/rad)
DEFAULT_RELAXATION_PHI = 0.1 # Relaxation for phi iteration

class BEMTAnalysisError(Exception):
    """Custom BEMT analysis errors."""
    pass

# --- Helper Functions for Loss Factors ---

def _calculate_f_root_exponent(num_blades: int, r_norm: float, r_hub_norm: float, phi_rad: float) -> float:
    """
    Exponent 'f_root_exp_arg' for F_root. Uses Prandtl hub loss.
    (Original Dayhoum Eq 19 for f_root was problematic).
    """
    if r_norm <= r_hub_norm + 1e-6 or r_hub_norm <= 1e-6: # At/inside hub or point hub
        return 1000.0 # Large value for small F_root (high loss)

    f_root_exp_arg = (num_blades / 2.0) * ( (r_norm - r_hub_norm) / (r_hub_norm * phi_rad) )
    return f_root_exp_arg

def _calculate_F_root(num_blades: int, r_norm: float, r_hub_norm: float, phi_rad: float) -> float:
    """Calculates F_root (hub loss factor)."""
    if r_norm <= r_hub_norm + 1e-6 : return 0.01 # Total loss at/inside hub

    f_r_exp = _calculate_f_root_exponent(num_blades, r_norm, r_hub_norm, phi_rad)
    f_r_exp_clipped = np.clip(abs(f_r_exp), 0, 700) # Positive and avoid overflow

    F_r = (2.0 / np.pi) * np.arccos(np.exp(-f_r_exp_clipped))
    return np.nan_to_num(F_r, nan=0.01) # Handle NaNs


def _calculate_F_gap_simplified_prandtl_tip(num_blades: int, r_norm: float, phi_rad: float) -> float:
    """
    Simplified F_gap using Prandtl's tip loss.
    Uses exponent 'f' from Dayhoum.
    """
    phi_rad_safe = max(1e-4, abs(np.sin(phi_rad))) # Use sin(phi)
    if r_norm >= 1.0 - 1e-6: return 0.01 # Total loss at/near tip

    f_tip_exp_arg = (num_blades / 2.0) * (1.0 - r_norm) / phi_rad_safe
    #f_tip_exp_arg_clipped = np.clip(abs(f_tip_exp_arg), 0, 700) # Positive
    F_tip = (2.0 / np.pi) * np.arccos(np.exp(-f_tip_exp_arg))
    return np.nan_to_num(F_tip, nan=0.01)


def _calculate_f_and_g_for_Fgap_dayhoum(
    num_blades: int, r_norm: float, tip_radius_m: float, tip_gap_clearance_m: float, phi_rad: float
) -> Tuple[float, float]:
    """Calculates f and g for Dayhoum F_gap model."""
    # sin(phi) is standard for these factors.
    sin_phi_safe = np.sin(phi_rad)
    if abs(sin_phi_safe) < 1e-4: # Avoid division by zero
        sin_phi_safe = 1e-4 * np.sign(sin_phi_safe if sin_phi_safe else 1.0)

    # f = (Nb/2) * ((1-r) / sin(phi))
    f_val = (num_blades / 2.0) * (1.0 - r_norm) / sin_phi_safe

    # g = (Nb * d_gap) / (2 * R * sin(phi))
    if tip_radius_m < 1e-6: raise ValueError("Tip radius must be positive for F_gap.")
    g_val = (num_blades * tip_gap_clearance_m) / (2.0 * tip_radius_m * sin_phi_safe)

    return f_val, g_val

def _solve_cd_inverse_numerical(X_target: float, m_param: float, K_val: float) -> float:
    """Numerically solves u for cd(u|m) = X_target, u in [0, K_val]."""
    X_clipped = np.clip(X_target, 0.0, 1.0) # cd is in [0,1]

    # cd(0|m) = 1, cd(K|m) = 0 for m in [0,1)
    if np.isclose(X_clipped, 1.0): return 0.0
    if np.isclose(X_clipped, 0.0): return K_val

    def func_to_solve(u_var: float) -> float:
        sn_u, cn_u, dn_u, _ = ellipj(u_var, m_param)
        if abs(dn_u) < 1e-9: # dn(u) near zero, cd undefined/infinite
            return np.sign(cn_u) * 1e9 - X_clipped if not np.isclose(cn_u,0.0) else -X_clipped # Large residual
        cd_u_val = cn_u / dn_u
        return cd_u_val - X_clipped

    try:
        if not (np.isfinite(K_val) and K_val > 1e-9): # K_val must be positive
            raise ValueError(f"K_val ({K_val}) invalid for brentq with m={m_param}.")

        # Check bracketing
        f_at_0 = 1.0 - X_clipped
        f_at_K = 0.0 - X_clipped
        if np.sign(f_at_0) == np.sign(f_at_K) and not (np.isclose(f_at_0,0) or np.isclose(f_at_K,0)):
            raise ValueError(f"Root not bracketed. X={X_target:.3f}, m={m_param:.3f}.")

        u_solution = brentq(func_to_solve, 0.0, K_val, xtol=1e-7, rtol=1e-7)
        return u_solution
    except ValueError as e: # brentq issues
        # Fallback: linear interpolation u = K * (1 - X_target)
        # print(f"Warning: brentq failed for cd_inverse (X={X_target:.4f}, m={m_param:.4f}, K={K_val:.4f}). Error: {e}. Using fallback.")
        return K_val * (1.0 - X_clipped)


def _calculate_F_gap_dayhoum(
    num_blades: int, r_norm: float, tip_radius_m: float,
    tip_gap_clearance_m: float, phi_rad: float
) -> float:
    """Calculates F_gap using Dayhoum et al. model."""
    if tip_gap_clearance_m <= 1e-7: # No/negligible gap
        return 1.0

    f_val, g_val = _calculate_f_and_g_for_Fgap_dayhoum(
        num_blades, r_norm, tip_radius_m, tip_gap_clearance_m, phi_rad
    )

    if abs(g_val) <= 1e-7: # g effectively zero (e.g., small d_gap)
        return 1.0

    cosh_g = np.cosh(g_val)
    if np.isinf(cosh_g) or cosh_g < 1e-9: # g too large or problematic cosh_g
        return 0.01 # Assume high loss

    sech_g_val = 1.0 / cosh_g
    m_param = sech_g_val**2 # k^2 for SciPy elliptic functions

    if m_param >= 1.0 - 1e-9: # m ~1 (g ~0)
        # print(f"Info: F_gap m_param {m_param:.4f} ~1 (g={g_val:.3e}). F_gap should be ~1.0.")
        # Formula unstable as K -> inf. Goodman model gives F_gap=1 as g->0.
        return 0.999 # Close to 1

    try:
        K_val = ellipk(m_param) # K(m) = K(k^2)
    except Exception as e:
        # print(f"Warning: ellipk failed for m={m_param:.4f} (g={g_val:.3e}). Error: {e}. Defaulting F_gap.")
        return 0.5 # Fallback

    if not np.isfinite(K_val) or K_val <= 1e-7:
        # print(f"Warning: K_val is {K_val} (m={m_param:.4f}). Problematic. Defaulting F_gap.")
        return 0.5 # Fallback

    cosh_f_plus_g = np.cosh(f_val + g_val)
    if np.isinf(cosh_f_plus_g) or cosh_f_plus_g < 1e-9:
        return 0.01 # Denominator issues, X_arg small/invalid

    X_arg = cosh_g / cosh_f_plus_g
    u_cd_inv = _solve_cd_inverse_numerical(X_arg, m_param, K_val)

    F_g = u_cd_inv / K_val
    return np.clip(F_g, 0.01, 1.0)


def _calculate_F_sh(num_blades: int, r_norm: float, r_hub_norm: float, tip_radius_m: float,
                    tip_gap_clearance_m: float, phi_rad: float,
                    use_dayhoum_F_gap_model: bool = True) -> float:
    """
    Overall loss F_sh = F_root * F_gap.
    F_root is Prandtl-like hub loss. Switch for F_gap model.
    """
    F_r = _calculate_F_root(num_blades, r_norm, r_hub_norm, phi_rad) # Standard Prandtl hub

    if use_dayhoum_F_gap_model:
        F_g = _calculate_F_gap_dayhoum(num_blades, r_norm, tip_radius_m, tip_gap_clearance_m, phi_rad)
    else: # Open rotor or simplified shrouded model
        F_g = _calculate_F_gap_simplified_prandtl_tip(num_blades, r_norm, phi_rad)

    return np.clip(F_r * F_g, 0.01, 1.0) # Sensible bounds for F_sh


def _solve_element_lambda_iterative(
    rho_kgm3: float, mu_Pas: float, omega_rads: float, V_inf: float,
    radius_m: float, tip_radius_m: float, r_hub_norm: float,
    chord_m: float, twist_deg: float, airfoil_obj: Airfoil,
    num_blades: int, sigma_d_sq: float,
    tip_gap_clearance_m: float,
    assumed_cl_alpha_rad: float = DEFAULT_CLA_2PI,
    use_dayhoum_F_gap_model: bool = True
) -> Tuple[float, float, float, float, float, float, float, float, float, float]:
    """
    Iteratively solves inflow ratio lambda(r) for one element (Dayhoum et al.).
    """
    r_norm = radius_m / tip_radius_m if tip_radius_m > 1e-6 else 0
    local_twist_rad = np.radians(twist_deg)

    if np.isclose(radius_m, 0.0): # Avoid div by zero for solidity at center
        local_solidity_sigma = np.inf
    else:
        local_solidity_sigma = (num_blades * chord_m) / (2 * np.pi * radius_m)
    u_t_ms = omega_rads * radius_m

    #phi_rad_iter = np.arccos(u_t_ms/V_inf)

    phi_rad_iter = np.maximum(np.atan2(V_inf, u_t_ms),1e-4)
    lambda_r = 0.0 # Init lambda_r

    for iteration in range(DEFAULT_MAX_ITERATIONS_ELEMENT):
        F_sh_iter = _calculate_F_sh(
            num_blades, r_norm, r_hub_norm, tip_radius_m, tip_gap_clearance_m, phi_rad_iter,
            use_dayhoum_F_gap_model=use_dayhoum_F_gap_model
        )

        # Lambda calculation terms (Dayhoum Eq 31)
        # lambda(r) = Term1 * (sqrt(1 + Term_in_sqrt_arg * theta_r) - 1)
        # Term1 = (sigma_local * cl_alpha * sigma_d_sq) / (4 * F_sh)
        # Term_in_sqrt_arg = (8 * F_sh) / (sigma_local * cl_alpha * sigma_d_sq)

        num_T1_den_TisA = local_solidity_sigma * assumed_cl_alpha_rad * sigma_d_sq
        den_T1_num_TisA = 4 * F_sh_iter

        if abs(den_T1_num_TisA) < 1e-9: # F_sh ~zero
            lambda_r_new_calc = 0.0 # No force, no induced flow
        elif abs(num_T1_den_TisA) < 1e-9: # sigma, Cl_alpha, or sigma_d_sq ~zero
            lambda_r_new_calc = 0.0
        else:
            term1_lambda = num_T1_den_TisA / den_T1_num_TisA
            # term_theta_r is local_twist_rad * r_norm. local_twist_rad is effective pitch theta.
            term_inside_sqrt_factor = (8 * F_sh_iter) / num_T1_den_TisA
            arg_of_sqrt = 1.0 + term_inside_sqrt_factor * local_twist_rad * r_norm

            if arg_of_sqrt < 0:
                # print(f"Warning: r={radius_m:.3f}m, sqrt arg negative ({arg_of_sqrt:.3e}). Lambda=0.")
                lambda_r_new_calc = 0.0
            else:
                lambda_r_new_calc = term1_lambda * (np.sqrt(arg_of_sqrt) - 1.0)

        #lambda_r_new_calc = np.clip(lambda_r_new_calc, -0.5, 2.0) # Practical limits

        V_axial_element = lambda_r_new_calc * omega_rads * tip_radius_m

        #if np.isclose(u_t_ms, 0.0): # Centerline or no rotation
        #    phi_rad_new = np.pi / 2.0 if V_axial_element >= 0 else -np.pi / 2.0
        #else:
        phi_rad_new = lambda_r_new_calc / r_norm

        if abs(phi_rad_new - phi_rad_iter) < DEFAULT_CONVERGENCE_TOL_PHI:
            phi_rad_iter = phi_rad_new
            lambda_r = lambda_r_new_calc
            break

        phi_rad_iter = phi_rad_iter + DEFAULT_RELAXATION_PHI * (phi_rad_new - phi_rad_iter)
        lambda_r = lambda_r_new_calc
    # else: # No convergence
        # print(f"Warning: Element lambda iter r={radius_m:.3f}m no converge. Last phi={np.degrees(phi_rad_iter):.2f}, lambda={lambda_r:.4f}")

    F_sh_final = _calculate_F_sh(
        num_blades, r_norm, r_hub_norm, tip_radius_m, tip_gap_clearance_m, phi_rad_iter,
        use_dayhoum_F_gap_model=use_dayhoum_F_gap_model
    )

    V_axial_final_element = lambda_r * omega_rads * tip_radius_m
    u_t_final_ms = omega_rads * radius_m
    W_ms_final = np.sqrt(V_axial_final_element**2 + u_t_final_ms**2)

    #if np.isclose(W_ms_final, 0.0): # No local flow
    #    return lambda_r, phi_rad_iter, 0.0, 0.0, 0.0, 0.0, F_sh_final, 0.0, 0.0, 0.0

    local_alpha_rad = phi_rad_iter - local_twist_rad
    local_alpha_deg_final = np.degrees(local_alpha_rad)
    Re_final = (rho_kgm3 * W_ms_final * chord_m) / mu_Pas
    cl, cd = airfoil_obj.get_lift_drag_coeffs(local_alpha_deg_final, Re_final)
    #except AttributeError: raise AttributeError(f"Airfoil '{airfoil_obj.name}' needs 'get_lift_drag_coeffs'.")
    #except Exception: cl, cd = 0.0, 0.0 # Fallback

    term_force_coeff = 0.5 * rho_kgm3 * W_ms_final**2 * chord_m * F_sh_final * num_blades
    dT_dr_val = term_force_coeff * (cl * np.cos(phi_rad_iter) - cd * np.sin(phi_rad_iter))
    dFx_dr_val = term_force_coeff * (cl * np.sin(phi_rad_iter) + cd * np.cos(phi_rad_iter))
    dQ_dr_val = dFx_dr_val * radius_m

    return (lambda_r, phi_rad_iter, local_alpha_deg_final, cl, cd, Re_final, F_sh_final,
            dT_dr_val, dQ_dr_val, W_ms_final)


# --- Main BEMT Calculation Function ---
def calculate_bemt_performance_dayhoum(
    rotor_stations_data: List[Dict[str, Any]],
    omega_rads: float,
    v_inf: float,
    num_blades: int,
    rho_kgm3: float,
    mu_Pas: float,
    root_radius_m: float,
    tip_radius_m: float,
    sigma_d_sq: float,
    tip_gap_clearance_m: float,
    assumed_cl_alpha_rad: float = DEFAULT_CLA_2PI,
    use_dayhoum_F_gap_model: bool = True
) -> Dict[str, Any]:
    """
    BEMT for SHROUDED ROTOR IN HOVER (Dayhoum et al. model).

    Args:
        rotor_stations_data: List of station dicts (radius_m, chord_m, twist_deg, airfoil_object).
                             Airfoil needs get_lift_drag_coeffs, optionally get_lift_curve_slope.
        omega_rads: Rotational speed (rad/s).
        num_blades: Number of blades.
        rho_kgm3: Fluid density (kg/m^3).
        mu_Pas: Fluid dynamic viscosity (Pa.s).
        root_radius_m: Hub radius (m).
        tip_radius_m: Tip radius (m).
        sigma_d_sq: Diffuser area ratio squared (A_exit / A_rotor_disk)^2.
        tip_gap_clearance_m: Tip gap clearance (m).
        assumed_cl_alpha_rad: Default lift curve slope (1/rad).
        use_dayhoum_F_gap_model: True for Dayhoum F_gap, False for Prandtl tip loss.

    Returns:
        Dict: Performance results.
    """
    if not rotor_stations_data: raise ValueError("rotor_stations_data is empty.")
    if omega_rads <= 0: raise ValueError("omega_rads must be positive.")
    if tip_radius_m <= root_radius_m + 1e-6 : raise ValueError("tip_radius_m > root_radius_m required.")
    if tip_gap_clearance_m < 0: raise ValueError("tip_gap_clearance_m cannot be negative.")

    r_hub_norm = root_radius_m / tip_radius_m if tip_radius_m > 1e-6 else 0

    element_results = []
    integrand_dT_dr = []
    integrand_dQ_dr = []
    integration_radii = []

    for station_idx, station in enumerate(rotor_stations_data):
        r_m = station['radius_m']
        # Filter stations outside active blade span
        if r_m < root_radius_m - 1e-6 or r_m > tip_radius_m + 1e-6: continue
        # Skip calculation at exact hub (if non-zero) or center (if zero hub), unless first station
        if (np.isclose(r_m, root_radius_m) and root_radius_m > 1e-6 and station_idx > 0) or \
           (np.isclose(r_m, 0.0) and root_radius_m < 1e-6 and station_idx > 0):
            continue

        chord_m = station['chord_m']
        # Use base_twist_deg + collective if available, else twist_deg
        twist_deg = station.get('base_twist_deg', station['twist_deg']) \
                        + station.get('collective_pitch_deg',0.0)

        airfoil: Airfoil = station['airfoil_object']

        cl_alpha_rad_local = assumed_cl_alpha_rad
        # Try to get cl_alpha from airfoil (estimate Re)
        if hasattr(airfoil, 'get_lift_curve_slope'):
            try:
                # Re_for_cl_alpha = 500000 # Using fixed typical Re for cl_alpha
                cl_alpha_rad_local = airfoil.get_lift_curve_slope(Re=500000, alpha0_deg=0.0)
                if not (0.1 < cl_alpha_rad_local < 8.0) : # Sanity check
                    cl_alpha_rad_local = assumed_cl_alpha_rad # Revert
            except Exception: pass # Use default

        lambda_r, phi_r, alpha_r, cl_r, cd_r, Re_r, Fsh_r, dTdr, dQdr, W_r = \
            _solve_element_lambda_iterative(
                rho_kgm3, mu_Pas, omega_rads, v_inf,
                r_m, tip_radius_m, r_hub_norm,
                chord_m, twist_deg, airfoil,
                num_blades, sigma_d_sq,
                tip_gap_clearance_m,
                cl_alpha_rad_local,
                use_dayhoum_F_gap_model=use_dayhoum_F_gap_model
            )


        blade_span = tip_radius_m - root_radius_m
        eta_val = (r_m - root_radius_m) / blade_span

        element_results.append({
            "radius_m": r_m, "eta": eta_val,
            "lambda": lambda_r, "phi_deg": np.degrees(phi_r), "alpha_deg": alpha_r,
            "Cl": cl_r, "Cd": cd_r, "Re": Re_r, "F_sh": Fsh_r, "W_ms": W_r,
            "dT_dr_N_m": dTdr, "dQ_dr_Nm_m": dQdr
        })
        integrand_dT_dr.append(dTdr)
        integrand_dQ_dr.append(dQdr)
        integration_radii.append(r_m)

    if not integration_radii:
        print("Warning: BEMT processed no valid elements.")
        return {
            "total_rotor_thrust_N": 0.0, "total_torque_Nm": 0.0, "total_power_W": 0.0,
            "figure_of_merit_hover": 0.0, "Ct_rotor":0.0, "Cp_rotor":0.0, "spanwise_results": []
        }

    sort_indices = np.argsort(integration_radii)
    radii_np = np.array(integration_radii)[sort_indices]
    dTdr_np = np.array(integrand_dT_dr)[sort_indices]
    dQdr_np = np.array(integrand_dQ_dr)[sort_indices]

    valid_T = ~np.isnan(dTdr_np) & ~np.isinf(dTdr_np)
    valid_Q = ~np.isnan(dQdr_np) & ~np.isinf(dQdr_np)

    total_rotor_thrust_N = 0.0
    if np.any(valid_T) and len(radii_np[valid_T]) >= 2:
        total_rotor_thrust_N = trapz(dTdr_np[valid_T], x=radii_np[valid_T])
    elif np.any(valid_T) and len(radii_np[valid_T]) == 1 and (tip_radius_m - root_radius_m > 1e-6):
        total_rotor_thrust_N = dTdr_np[valid_T][0] * (tip_radius_m - root_radius_m)

    total_torque_Nm = 0.0
    if np.any(valid_Q) and len(radii_np[valid_Q]) >= 2:
        total_torque_Nm = trapz(dQdr_np[valid_Q], x=radii_np[valid_Q])
    elif np.any(valid_Q) and len(radii_np[valid_Q]) == 1 and (tip_radius_m - root_radius_m > 1e-6):
        total_torque_Nm = dQdr_np[valid_Q][0] * (tip_radius_m - root_radius_m)

    total_power_W = total_torque_Nm * omega_rads
    A_rotor_disk = np.pi * (tip_radius_m**2 - root_radius_m**2)

    figure_of_merit = 0.0
    if total_power_W > 1e-6 and total_rotor_thrust_N > 1e-6 and A_rotor_disk > 1e-6 and rho_kgm3 > 1e-6:
        v_ideal_hover = np.sqrt(abs(total_rotor_thrust_N) / (2 * rho_kgm3 * A_rotor_disk)) # abs for safety
        ideal_power_hover = abs(total_rotor_thrust_N) * v_ideal_hover
        figure_of_merit = ideal_power_hover / abs(total_power_W)
        figure_of_merit = np.clip(figure_of_merit, 0.0, 1.0) # FM <= 1

    Area_disk_for_coeff = np.pi * tip_radius_m**2
    Ct_rotor = 0.0; Cp_rotor = 0.0
    if Area_disk_for_coeff > 1e-6 and rho_kgm3 > 1e-6 and omega_rads > 1e-6 and tip_radius_m > 1e-6:
        Ct_rotor = total_rotor_thrust_N / (rho_kgm3 * Area_disk_for_coeff * (omega_rads * tip_radius_m)**2)
        Cp_rotor = total_power_W / (rho_kgm3 * Area_disk_for_coeff * (omega_rads * tip_radius_m)**3)

    # Re-sort element_results for consistent output
    final_spanwise_results = [element_results[i] for i in sort_indices] if element_results else []

    return {
        "total_rotor_thrust_N": total_rotor_thrust_N,
        "total_torque_Nm": total_torque_Nm,
        "total_power_W": total_power_W,
        "figure_of_merit_hover": figure_of_merit,
        "Ct_rotor": Ct_rotor,
        "Cp_rotor": Cp_rotor,
        "spanwise_results": final_spanwise_results
    }