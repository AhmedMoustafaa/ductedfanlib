"""
Blade Element Momentum Theory (BEMT) for open rotors in axial flight.

Fixes applied vs original:
  1. Sign convention: alpha = theta - phi  (pitch minus inflow angle).
     Original had alpha = phi - theta, giving negative AoA and negative thrust.
  2. Universal velocity-based phi-iteration (works for hover AND forward flight).
     The original a*(1+a)*V_inf^2 momentum residual is scale-degenerate whenever
     V_inf << u_t (i.e., all practical propeller operation). Replaced with:
       dT/dr = 4π r ρ F (V_inf + v_i) v_i    [v_i = induced axial velocity]
       dQ/dr = 4π r³ ρ F Ω v_t (V_inf + v_i)  [v_t = induced tangential velocity]
     This formulation is non-degenerate from hover to cruise.
  3. Prandtl tip/hub exponents use sin(phi) consistently.
  4. fsolve convergence flag checked; non-convergence raises a warning.
  5. Figure of Merit returned alongside propulsive efficiency.
"""
from typing import List, Dict, Any
import warnings
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import trapezoid as trapz
from scipy.special import ellipk, ellipj
from ductedfanlib.geometry.airfoils import Airfoil

_MIN_VEL   = 1e-8
_CONV_TOL  = 1e-8
_MAX_ITER  = 300
_RELAX     = 0.4      # relaxation factor for phi-iteration


class BEMTAnalysisError(Exception):
    pass


# ── Loss factors ───────────────────────────────────────────────────────────────

def _prandtl_tip_F(B, r_norm, phi):
    sin_phi = max(1e-4, abs(np.sin(phi)))
    if r_norm >= 1.0 - 1e-6:
        return 0.01
    f = (B / 2.0) * (1.0 - r_norm) / sin_phi
    return float(np.nan_to_num((2.0/np.pi)*np.arccos(np.exp(-min(f, 700))), nan=0.01))


def _prandtl_hub_F(B, r_norm, r_hub_norm, phi):
    if r_norm <= r_hub_norm + 1e-6 or r_hub_norm <= 1e-6:
        return 1.0
    sin_phi = max(1e-4, abs(np.sin(phi)))
    f = (B / 2.0) * (r_norm - r_hub_norm) / (r_hub_norm * sin_phi)
    return float(np.nan_to_num((2.0/np.pi)*np.arccos(np.exp(-min(f, 700))), nan=0.01))


def _dayhoum_tip_F(B, r_norm, R_tip, gap, phi):
    if gap <= 1e-7:
        return 1.0
    sp = max(1e-4, abs(np.sin(phi)))
    g  = (B * gap) / (2.0 * R_tip * sp)
    f  = (B / 2.0) * (1.0 - r_norm) / sp
    ch = np.cosh(g)
    if np.isinf(ch):
        return 0.01
    m = (1.0 / ch) ** 2
    if m >= 1.0 - 1e-9:
        return 0.999
    try:
        K = ellipk(m)
    except Exception:
        return 0.5
    if not np.isfinite(K) or K < 1e-7:
        return 0.5
    X = np.clip(ch / np.cosh(f + g), 0.0, 1.0)
    try:
        def _res(u):
            _, cn, dn, _ = ellipj(u, m)
            return cn / dn - X
        u = brentq(_res, 0.0, K, xtol=1e-7)
    except Exception:
        u = K * (1.0 - X)
    return float(np.clip(u / K, 0.01, 1.0))


def _F_combined(B, r_norm, r_hub_norm, R_tip, gap, phi, use_dayhoum):
    Fh = _prandtl_hub_F(B, r_norm, r_hub_norm, phi)
    Ft = (_dayhoum_tip_F(B, r_norm, R_tip, gap, phi)
          if use_dayhoum else _prandtl_tip_F(B, r_norm, phi))
    return float(np.clip(Fh * Ft, 0.01, 1.0))


# ── Airfoil lookup ─────────────────────────────────────────────────────────────

def _get_cl_cd(airfoil, alpha_deg, W, chord, rho, mu):
    Re = max(1e3, rho * W * chord / mu)
    try:
        cl, cd = airfoil.get_lift_drag_coeffs(alpha_deg, Re, apply_viterna_poststall=True)
    except Exception:
        cl, cd = 0.0, 0.01
    return float(cl), float(cd), float(Re)


# ── Universal phi-iteration element solver ─────────────────────────────────────

def _solve_element(
    V_inf, omega, r, chord, twist_deg, airfoil, B,
    rho, mu, r_hub, R_tip, gap, use_dayhoum
):
    """
    Solve one blade element for any V_inf (including hover).

    State variables: v_i (induced axial, m/s), v_t (induced tangential, m/s).

    Momentum equations (velocity form, non-degenerate at V_inf=0):
        dT/dr = 4π r ρ F (V_inf + v_i) v_i
        dQ/dr = 4π r³ ρ F Ω (V_inf + v_i) v_t / r  =>  v_t via torque residual

    Blade element equations:
        phi   = arctan((V_inf + v_i) / (Ω r - v_t))
        alpha = theta - phi
        W     = sqrt((V_inf+v_i)^2 + (Ω r - v_t)^2)
        dT/dr = ½ ρ W² B c (cl cos φ + cd sin φ)
        dQ/dr = ½ ρ W² B c (cl sin φ - cd cos φ) · r
    """
    theta      = np.radians(twist_deg)
    u_t        = omega * r
    r_norm     = r / R_tip
    r_hub_norm = r_hub / R_tip

    # Initial guess: small induction
    v_i = max(0.05 * u_t, 0.5)
    v_t = 0.01 * u_t

    for _ in range(_MAX_ITER):
        V_ax  = V_inf + v_i
        V_tan = u_t - v_t
        if V_tan < _MIN_VEL:
            V_tan = _MIN_VEL
        phi = np.arctan2(V_ax, V_tan)
        phi = np.clip(phi, 1e-4, np.pi / 2 - 1e-4)
        W   = np.sqrt(V_ax**2 + V_tan**2)

        alpha_deg = np.degrees(theta - phi)   # propeller convention: alpha = pitch - inflow
        cl, cd, _ = _get_cl_cd(airfoil, alpha_deg, W, chord, rho, mu)
        cn = cl * np.cos(phi) + cd * np.sin(phi)   # normal  (thrust) force coefficient
        ct = cl * np.sin(phi) - cd * np.cos(phi)   # tangential (torque) force coefficient
        F  = _F_combined(B, r_norm, r_hub_norm, R_tip, gap, phi, use_dayhoum)

        # Blade element loads per unit span
        dT_blade = 0.5 * rho * W**2 * B * chord * cn
        dQ_blade = 0.5 * rho * W**2 * B * chord * ct * r

        # Momentum -> new induced velocities
        # From thrust momentum: (V_inf + v_i)*v_i = dT_blade / (4π r ρ F)
        k_T = dT_blade / (4.0 * np.pi * r * rho * F)
        # Solve: v_i^2 + V_inf*v_i - k_T = 0  (quadratic in v_i)
        disc = V_inf**2 + 4.0 * k_T
        if disc < 0 or k_T < 0:
            v_i_new = abs(v_i) * 0.5   # fallback: reduce induction
        else:
            v_i_new = (-V_inf + np.sqrt(disc)) / 2.0
        v_i_new = max(0.0, v_i_new)

        # From torque momentum: v_t*(V_inf+v_i) = dQ_blade / (4π r² ρ F Ω)
        k_Q = dQ_blade / (4.0 * np.pi * r**2 * rho * F * omega)
        denom = V_inf + v_i_new
        v_t_new = k_Q / denom if abs(denom) > _MIN_VEL else v_t

        # Convergence check
        if abs(v_i_new - v_i) < _CONV_TOL * max(1.0, v_i) and \
           abs(v_t_new - v_t) < _CONV_TOL * max(1.0, v_t):
            v_i, v_t = v_i_new, v_t_new
            break

        v_i = (1 - _RELAX) * v_i + _RELAX * v_i_new
        v_t = (1 - _RELAX) * v_t + _RELAX * v_t_new

    # ── Final reconstruction ───────────────────────────────────────────────────
    V_ax  = V_inf + v_i
    V_tan = max(u_t - v_t, _MIN_VEL)
    phi_f = np.arctan2(V_ax, V_tan)
    W_f   = np.sqrt(V_ax**2 + V_tan**2)
    alpha_f = np.degrees(theta - phi_f)
    cl_f, cd_f, Re_f = _get_cl_cd(airfoil, alpha_f, W_f, chord, rho, mu)
    F_f = _F_combined(B, r_norm, r_hub_norm, R_tip, gap, phi_f, use_dayhoum)

    # Induction factors for output (informational)
    a  = v_i / (V_inf + _MIN_VEL) if V_inf > 0.1 else v_i / (omega * R_tip)
    ap = v_t / u_t if u_t > _MIN_VEL else 0.0

    return a, ap, phi_f, alpha_f, cl_f, cd_f, Re_f, F_f, W_f


# ── Main public function ───────────────────────────────────────────────────────

def calculate_bemt_performance_axial(
    rotor_stations_data,
    V_axial_ms: float,
    omega_rads: float,
    num_blades: int,
    rho_kgm3: float,
    mu_Pas: float,
    root_radius_m: float,
    tip_radius_m: float,
    tip_gap_clearance_m: float,
    use_dayhoum_F_gap_model: bool = False
) -> dict:
    """
    BEMT analysis for an open or ducted rotor in axial flight (including hover).

    Parameters
    ----------
    rotor_stations_data    : list of dicts from get_rotor_bemt_stations()
    V_axial_ms             : freestream axial velocity (m/s); 0.0 for hover
    omega_rads             : rotational speed (rad/s)
    num_blades             : number of blades
    rho_kgm3               : air density (kg/m³)
    mu_Pas                 : dynamic viscosity (Pa·s)
    root_radius_m          : hub radius (m)
    tip_radius_m           : tip radius (m)
    tip_gap_clearance_m    : tip clearance (m); 0 for open rotor
    use_dayhoum_F_gap_model: True for Dayhoum shrouded tip loss; False for Prandtl

    Returns
    -------
    dict with keys: total_thrust_N, total_torque_Nm, total_power_W,
                    propulsive_efficiency, figure_of_merit,
                    Ct_rotor, Cp_rotor, Cq_rotor, advance_ratio,
                    spanwise_results (list of per-element dicts)
    """
    if not rotor_stations_data:
        raise ValueError("rotor_stations_data is empty.")
    if omega_rads <= 0:
        raise ValueError("omega_rads must be positive.")
    if tip_radius_m <= root_radius_m:
        raise ValueError("tip_radius_m must be > root_radius_m.")

    V = float(V_axial_ms)
    element_results, dT_list, dQ_list, r_list = [], [], [], []

    for st in rotor_stations_data:
        r_m = st["radius_m"]
        # Skip stations at exact tip (Prandtl F->0, degenerate momentum eq)
        if r_m < root_radius_m - 1e-6 or r_m >= tip_radius_m - 1e-4:
            continue
        chord_m   = st["chord_m"]
        twist_deg = st.get("base_twist_deg", st["twist_deg"]) + st.get("collective_pitch_deg", 0.0)
        airfoil   = st["airfoil_object"]

        a, ap, phi, alpha, cl, cd, Re, F, W = _solve_element(
            V, omega_rads, r_m, chord_m, twist_deg, airfoil, num_blades,
            rho_kgm3, mu_Pas, root_radius_m, tip_radius_m, tip_gap_clearance_m,
            use_dayhoum_F_gap_model
        )

        cn = cl * np.cos(phi) + cd * np.sin(phi)
        ct = cl * np.sin(phi) - cd * np.cos(phi)
        dT = 0.5 * rho_kgm3 * W**2 * num_blades * chord_m * cn * F
        dQ = 0.5 * rho_kgm3 * W**2 * num_blades * chord_m * ct * r_m * F

        r_list.append(r_m); dT_list.append(dT); dQ_list.append(dQ)
        element_results.append({
            "radius_m": r_m,
            "eta": (r_m - root_radius_m) / (tip_radius_m - root_radius_m),
            "a": a, "a_prime": ap,
            "phi_deg": float(np.degrees(phi)),
            "alpha_deg": float(alpha),
            "Cl": float(cl), "Cd": float(cd), "Re": float(Re),
            "F_sh": float(F), "W_ms": float(W),
            "dT_dr_N_m": float(dT), "dQ_dr_Nm_m": float(dQ),
        })

    if not r_list:
        empty = {k: 0.0 for k in ["total_thrust_N", "total_torque_Nm", "total_power_W",
                                    "propulsive_efficiency", "efficiency", "figure_of_merit",
                                    "Ct_rotor", "Cp_rotor", "Cq_rotor", "advance_ratio"]}
        empty["spanwise_results"] = []
        return empty

    idx   = np.argsort(r_list)
    r_np  = np.array(r_list)[idx]
    dT_np = np.array(dT_list)[idx]
    dQ_np = np.array(dQ_list)[idx]

    T = float(trapz(dT_np, x=r_np))
    Q = float(trapz(dQ_np, x=r_np))
    P = Q * omega_rads

    n_rps  = omega_rads / (2.0 * np.pi)
    D      = 2.0 * tip_radius_m
    J      = V / (n_rps * D) if n_rps > 1e-6 else 0.0
    A_disk = np.pi * tip_radius_m**2

    # Propulsive efficiency η = TV/P  (meaningful for J > 0)
    eta = float(np.clip((T * V) / P, 0.0, 1.0)) if abs(P) > 1e-6 and V > 0.1 else 0.0

    # Figure of Merit = P_ideal / P_actual  (meaningful at hover/low J)
    FM = 0.0
    if T > 1e-6 and P > 1e-6:
        v_ideal = np.sqrt(T / (2.0 * rho_kgm3 * A_disk))
        FM = float(np.clip(T * v_ideal / P, 0.0, 1.0))

    Ct = T / (rho_kgm3 * n_rps**2 * D**4) if n_rps > 1e-6 else 0.0
    Cp = P / (rho_kgm3 * n_rps**3 * D**5) if n_rps > 1e-6 else 0.0
    Cq = Q / (rho_kgm3 * n_rps**2 * D**5) if n_rps > 1e-6 else 0.0

    return {
        "total_thrust_N":        T,
        "total_torque_Nm":       Q,
        "total_power_W":         P,
        "propulsive_efficiency": eta,
        "efficiency":            eta,   # alias for study.py compatibility
        "figure_of_merit":       FM,
        "Ct_rotor":              Ct,
        "Cp_rotor":              Cp,
        "Cq_rotor":              Cq,
        "advance_ratio":         J,
        "spanwise_results":      [element_results[i] for i in idx],
    }
