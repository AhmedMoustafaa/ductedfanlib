"""
Blade Element Momentum Theory (BEMT) analysis for rotors,
adapted for shrouded rotors based on the Dayhoum et al. model.
"""
from typing import List, Dict, Any, Tuple, Callable
import numpy as np
from scipy.optimize import newton
from scipy.integrate import trapezoid as trapz
from scipy.special import ellipk

from ductedfanlib.geometry.airfoils import Airfoil

# Constants
DEFAULT_PHI_INITIAL_DEG = 10.0  # Initial guess for inflow angle phi
DEFAULT_MAX_ITERATIONS_ELEMENT = 50
DEFAULT_CONVERGENCE_TOL_PHI = 1e-5  # Tolerance for phi convergence in radians
DEFAULT_CLA_2PI = 2 * np.pi  # Default lift curve slope (1/rad) if not available from airfoil
DEFAULT_RELAXATION = 0.5

class BEMTAnalysisError(Exception):
    """Custom exception for BEMT analysis errors."""
    pass

def _calculate_f_root(num_blades: int, r_norm: float, r_hub_norm: float, phi_rad: float) -> float:
    """
    """
    phi_rad_safe = max(1e-3, abs(phi_rad))
    if r_norm >= 1.0 or r_norm <= r_hub_norm:
        return 1000

    f_root_exp_arg = (num_blades / 2.0) * (r_norm - r_hub_norm) / (
                r_hub_norm * phi_rad_safe)
    return f_root_exp_arg


def _calculate_F_root(num_blades: int, r_norm: float, r_hub_norm: float, phi_rad: float) -> float:
    """ Calculates F_root using Equation 18."""
    if r_norm <= r_hub_norm + 1e-6: return 0.01
    f_r = _calculate_f_root(num_blades, r_norm, r_hub_norm, phi_rad)
    f_r_clipped = np.clip(f_r, -700, 700)
    F_r = (2.0 / np.pi) * np.arccos(np.exp(-f_r_clipped))
    return np.nan_to_num(F_r, nan=0.01)


def _calculate_F_gap_simplified_prandtl_tip(num_blades: int, r_norm: float, phi_rad: float) -> float:
    """
    Calculates a simplified F_gap using Prandtl's tip loss formula as a placeholder
    for the complex Equation 20.
    This uses Equation 21 from the paper for 'f'.
    """
    phi_rad_safe = max(1e-3, abs(phi_rad))
    if r_norm >= 1.0 - 1e-6: return 0.01

    # f from Equation 21 (Prandtl's tip loss exponent argument)
    f_tip_exp_arg = (num_blades / 2.0) * (1.0 - r_norm) / np.sin(phi_rad_safe)
    f_tip_exp_arg_clipped = np.clip(f_tip_exp_arg, -700, 700)
    F_tip = (2.0 / np.pi) * np.arccos(np.exp(-f_tip_exp_arg_clipped))
    return np.nan_to_num(F_tip, nan=0.01)


def _calculate_F_sh(num_blades: int, r_norm: float, r_hub_norm: float, phi_rad: float,
                    use_simplified_F_gap: bool = True) -> float:
    """
    Calculates the overall loss correction factor F_sh = F_root * F_gap (Equation 17).
    If use_simplified_F_gap is True, uses Prandtl tip loss for F_gap.
    """
    F_r = _calculate_F_root(num_blades, r_norm, r_hub_norm, phi_rad)

    if use_simplified_F_gap:
        F_g = _calculate_F_gap_simplified_prandtl_tip(num_blades, r_norm, phi_rad)
    else:
        raise NotImplementedError("Full F_gap (Equation 20) for shrouded rotors is not yet implemented.")
    return F_r * F_g


def _solve_element_lambda_iterative(
        rho_kgm3: float, mu_Pas: float, V_axial_at_disk_ms: float, omega_rads: float,
        radius_m: float, tip_radius_m: float, r_hub_norm: float,  # r_hub_norm = hub_radius / tip_radius
        chord_m: float, twist_deg: float, airfoil_obj: Airfoil,
        num_blades: int, sigma_d_sq: float,  # sigma_d_sq is (A4/AR)^2, diffuser expansion ratio squared
        assumed_cl_alpha_rad: float = DEFAULT_CLA_2PI  # Lift curve slope per radian
) -> Tuple[float, float, float, float, float, float, float, float, float, float]:
    """
    Iteratively solves for inflow ratio lambda(r) for a single blade element
    based on Dayhoum et al., Equation 31.

    Returns: (lambda_final, phi_rad_final, local_alpha_deg_final, Cl, Cd, Re, F_sh_final,
              dT_dr_val, dQ_dr_val, W_ms_final)
    """
    r_norm = radius_m / tip_radius_m
    local_twist_rad = np.radians(twist_deg)
    local_solidity_sigma = (num_blades * chord_m) / (2 * np.pi * radius_m)

    phi_rad_iter = np.radians(DEFAULT_PHI_INITIAL_DEG)  # Initial guess for phi

    for iteration in range(DEFAULT_MAX_ITERATIONS_ELEMENT):
        # 1. Calculate F_sh based on current phi_rad_iter
        F_sh_iter = _calculate_F_sh(num_blades, r_norm, r_hub_norm, phi_rad_iter)
        if F_sh_iter <= 1e-6: F_sh_iter = 1e-6  # Avoid division by zero

        # 2. Calculate lambda(r) using Equation 31
        # lambda(r) = (sigma * cl_alpha * sigma_d_sq / (4 * F_sh)) * (sqrt(1 + (8 * F_sh / (sigma * cl_alpha * sigma_d_sq)) * theta * r_norm) - 1)

        term1_lambda = (local_solidity_sigma * assumed_cl_alpha_rad * sigma_d_sq) / (4 * F_sh_iter)
        term_sqrt_lambda = (8 * F_sh_iter * local_twist_rad * r_norm) / \
                           (local_solidity_sigma * assumed_cl_alpha_rad * sigma_d_sq)
        if 1.0 + term_sqrt_lambda < 0:
            lambda_r = 0.0
        else:
            lambda_r = term1_lambda * (np.sqrt(1.0 + term_sqrt_lambda) - 1.0)

        lambda_r = np.clip(lambda_r, -0.5, 2.0)

        # 3. Calculate v_ind, u_p, u_t
        v_ind_ms = lambda_r * omega_rads * tip_radius_m
        u_p_ms = -v_ind_ms

        V_axial_element = lambda_r * omega_rads * tip_radius_m

        u_t_ms = omega_rads * radius_m

        # 4. Calculate new phi_rad
        if np.isclose(u_t_ms, 0.0):
            phi_rad_new = np.pi / 2.0 if V_axial_element > 0 else (
                -np.pi / 2.0 if V_axial_element < 0 else phi_rad_iter)
        else:
            phi_rad_new = np.arctan2(V_axial_element, u_t_ms)

        # 5. Check convergence
        if abs(phi_rad_new - phi_rad_iter) < DEFAULT_CONVERGENCE_TOL_PHI:
            phi_rad_iter = phi_rad_new
            break  # Converged

        phi_rad_iter = phi_rad_iter + DEFAULT_RELAXATION * (phi_rad_new - phi_rad_iter)  # Relaxation
    else:  # Loop finished without break (no convergence)
        # print(f"Warning: Element at r={radius_m:.3f}m did not converge for phi/lambda. Last phi={np.degrees(phi_rad_iter):.2f}deg")
        pass

    # Converged (or max iterations reached), calculate final properties for this element
    lambda_final = lambda_r
    phi_rad_final = phi_rad_iter
    F_sh_final = _calculate_F_sh(num_blades, r_norm, r_hub_norm, phi_rad_final)
    if F_sh_final <= 1e-6: F_sh_final = 1e-6

    V_axial_final_element = lambda_final * omega_rads * tip_radius_m
    u_t_final_ms = omega_rads * radius_m
    W_ms_final = np.sqrt(V_axial_final_element ** 2 + u_t_final_ms ** 2)

    if np.isclose(W_ms_final, 0.0):
        return lambda_final, phi_rad_final, 0.0, 0.0, 0.0, 0.0, F_sh_final, 0.0, 0.0, 0.0

    local_alpha_rad = phi_rad_final - local_twist_rad
    local_alpha_deg_final = np.degrees(local_alpha_rad)
    Re_final = max(1e3, (rho_kgm3 * W_ms_final * chord_m) / mu_Pas)

    try:
        # Airfoil object needs to handle post-stall internally based on alpha_deg_final
        cl, cd = airfoil_obj.get_lift_drag_coeffs(local_alpha_deg_final, Re_final)
    except AttributeError:
        raise AttributeError(f"Airfoil object '{airfoil_obj.name}' needs 'get_lift_drag_coeffs' method.")
    except Exception:
        cl, cd = 0.0, 0.0

    term_force_coeff = 0.5 * rho_kgm3 * W_ms_final ** 2 * chord_m * F_sh_final * num_blades

    dT_dr_val = term_force_coeff * (cl * np.cos(phi_rad_final) - cd * np.sin(phi_rad_final))
    dFx_dr_val = term_force_coeff * (cl * np.sin(phi_rad_final) + cd * np.cos(phi_rad_final))
    dQ_dr_val = dFx_dr_val * radius_m

    return (lambda_final, phi_rad_final, local_alpha_deg_final, cl, cd, Re_final, F_sh_final,
            dT_dr_val, dQ_dr_val, W_ms_final)


def calculate_bemt_performance_dayhoum(
        rotor_stations_data: List[Dict[str, Any]],
        # V_axial_ms: float, # For hover, V_axial_ms (freestream) is 0. Eq 31 gives v_ind which becomes the axial vel.
        omega_rads: float,
        num_blades: int,
        rho_kgm3: float,
        mu_Pas: float,
        root_radius_m: float,
        tip_radius_m: float,
        sigma_d_sq: float,  # Diffuser expansion ratio (A_exit/A_rotor)^2
        assumed_cl_alpha_rad: float = DEFAULT_CLA_2PI
) -> Dict[str, Any]:
    """
    Performs BEMT analysis for a SHROUDED ROTOR IN HOVER based on Dayhoum et al. model.
    This model uses an analytical solution for inflow ratio lambda, iterated with loss factors.

    Args:
        rotor_stations_data: List of station data. Each dict needs 'radius_m', 'chord_m',
                             'twist_deg', 'airfoil_object'.
                             Airfoil object needs methods:
                                - get_lift_drag_coeffs(alpha_deg, Re) -> (cl, cd)
                                - (Ideally) get_lift_curve_slope() -> cl_alpha_rad
                                - (Ideally) get_stall_parameters() for post-stall if not internal
        omega_rads: Rotational speed (rad/s).
        num_blades: Number of blades.
        rho_kgm3: Fluid density (kg/m^3).
        mu_Pas: Fluid dynamic viscosity (Pa.s).
        root_radius_m: Hub radius of the bladed section (m).
        tip_radius_m: Tip radius of the rotor (m).
        sigma_d_sq: Shroud diffuser area ratio squared, (A_exit / A_rotor_disk)^2.
                     Note: Paper uses sigma_d = A4/AR. Eq 25, 27 has sigma_d^2 in denominator.
                     Eq 31 has sigma_d^2 in numerator of terms. Let's assume sigma_d_sq passed is (sigma_d from paper)^2.
                     The paper's sigma_d in Eq 31 is A4/AR. So sigma_d_sq here should be (A4/AR)^2.
        assumed_cl_alpha_rad: Assumed lift curve slope (per radian) if not available from airfoil object.

    Returns:
        Dict[str, Any]: Performance results including 'total_rotor_thrust_N', 'total_torque_Nm', etc.
    """
    if not rotor_stations_data: raise ValueError("rotor_stations_data is empty.")
    if omega_rads <= 0: raise ValueError("omega_rads must be positive.")
    if tip_radius_m <= root_radius_m: raise ValueError("tip_radius_m must be greater than root_radius_m.")

    r_hub_norm = root_radius_m / tip_radius_m

    element_results = []
    integrand_dT_dr = []
    integrand_dQ_dr = []
    integration_radii = []

    for station in rotor_stations_data:
        r_m = station['radius_m']
        if r_m < root_radius_m - 1e-6 or r_m > tip_radius_m + 1e-6:
            # print(f"Warning: Station radius {r_m} outside [root,tip] range. Skipping.")
            continue
        if np.isclose(r_m, 0.0):
            continue

        chord_m = station['chord_m']
        twist_deg = station['twist_deg']
        airfoil: Airfoil = station['airfoil_object']

        cl_alpha_rad_local = assumed_cl_alpha_rad
        if hasattr(airfoil, 'get_lift_curve_slope'):
            try:
                cl_alpha_rad_local = airfoil.get_lift_curve_slope()
            except TypeError:
                pass
            except AttributeError:
                pass

        lambda_r, phi_r, alpha_r, cl_r, cd_r, Re_r, Fsh_r, dTdr, dQdr, W_r = \
            _solve_element_lambda_iterative(
                rho_kgm3, mu_Pas, 0.0, omega_rads,
                r_m, tip_radius_m, r_hub_norm,
                chord_m, twist_deg, airfoil,
                num_blades, sigma_d_sq, cl_alpha_rad_local
            )

        element_results.append({
            "radius_m": r_m, "eta": (r_m - root_radius_m) / (tip_radius_m - root_radius_m) if (
                                                                                                          tip_radius_m - root_radius_m) > 0 else 0.5,
            "lambda": lambda_r, "phi_deg": np.degrees(phi_r), "alpha_deg": alpha_r,
            "Cl": cl_r, "Cd": cd_r, "Re": Re_r, "F_sh": Fsh_r, "W_ms": W_r,
            "dT_dr_N_m": dTdr, "dQ_dr_Nm_m": dQdr
        })
        integrand_dT_dr.append(dTdr)
        integrand_dQ_dr.append(dQdr)
        integration_radii.append(r_m)

    if not integration_radii:  # No valid elements processed
        return {
            "total_rotor_thrust_N": 0.0, "total_torque_Nm": 0.0, "total_power_W": 0.0,
            "rotor_efficiency_hover": 0.0, "Ct": 0.0, "Cp": 0.0, "spanwise_results": []
        }

    sort_indices = np.argsort(integration_radii)
    radii_np = np.array(integration_radii)[sort_indices]
    dTdr_np = np.array(integrand_dT_dr)[sort_indices]
    dQdr_np = np.array(integrand_dQ_dr)[sort_indices]

    valid_T = ~np.isnan(dTdr_np) & ~np.isinf(dTdr_np)
    valid_Q = ~np.isnan(dQdr_np) & ~np.isinf(dQdr_np)

    total_rotor_thrust_N = 0.0
    if np.any(valid_T) and len(radii_np[valid_T]) > 1:
        total_rotor_thrust_N = trapz(dTdr_np[valid_T], x=radii_np[valid_T])
    elif len(radii_np[valid_T]) == 1 and (tip_radius_m - root_radius_m > 0):  # single element approx.
        total_rotor_thrust_N = dTdr_np[valid_T][0] * (tip_radius_m - root_radius_m)

    total_torque_Nm = 0.0
    if np.any(valid_Q) and len(radii_np[valid_Q]) > 1:
        total_torque_Nm = trapz(dQdr_np[valid_Q], x=radii_np[valid_Q])
    elif len(radii_np[valid_Q]) == 1 and (tip_radius_m - root_radius_m > 0):
        total_torque_Nm = dQdr_np[valid_Q][0] * (tip_radius_m - root_radius_m)

    total_power_W = total_torque_Nm * omega_rads

    A_rotor_disk = np.pi * (tip_radius_m ** 2 - (root_radius_m ** 2 if root_radius_m > 0 else 0))

    figure_of_merit = 0.0
    if total_power_W > 1e-6 and total_rotor_thrust_N > 0 and A_rotor_disk > 0:
        v_ideal_hover = np.sqrt(total_rotor_thrust_N / (2 * rho_kgm3 * A_rotor_disk))
        ideal_power_hover = total_rotor_thrust_N * v_ideal_hover
        figure_of_merit = ideal_power_hover / total_power_W
        figure_of_merit = np.clip(figure_of_merit, 0.0, 1.0)  # FM is <= 1


    Area_disk_for_coeff = np.pi * tip_radius_m ** 2
    Ct = 0.0
    Cp = 0.0
    if Area_disk_for_coeff > 1e-6 and rho_kgm3 > 1e-6 and omega_rads > 1e-6 and tip_radius_m > 1e-6:
        Ct = total_rotor_thrust_N / (rho_kgm3 * Area_disk_for_coeff * (omega_rads * tip_radius_m) ** 2)
        Cp = total_power_W / (rho_kgm3 * Area_disk_for_coeff * (omega_rads * tip_radius_m) ** 3)

    sorted_element_results = [element_results[i] for i in sort_indices]

    return {
        "total_rotor_thrust_N": total_rotor_thrust_N,
        "total_torque_Nm": total_torque_Nm,
        "total_power_W": total_power_W,
        "figure_of_merit_hover": figure_of_merit,
        "Ct_rotor": Ct,
        "Cp_rotor": Cp,
        "spanwise_results": sorted_element_results
    }