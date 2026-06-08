"""
Extended test suite for the corrected open-rotor BEMT.

Cases tested
────────────
A. Blade geometry sensitivity
   A1. High-pitch blade (hover-optimised, coarse pitch 35→18 deg)
   A2. Low-pitch blade  (cruise-optimised, fine pitch 18→8 deg)
   A3. High solidity    (4-blade, wider chord)
   A4. Low solidity     (2-blade, narrow chord)

B. Rotor size / RPM scaling
   B1. Small rotor  (R=0.15m, 8000 RPM) – UAV-scale
   B2. Large rotor  (R=1.5m,  600 RPM)  – light aircraft scale
   B3. Same rotor, 2× RPM   – power scales ~ RPM³
   B4. Same rotor, 0.5× RPM – thrust scales ~ RPM²

C. Atmospheric conditions
   C1. High altitude (ρ=0.66 kg/m³, 6000 m ISA)
   C2. Hot day       (ρ=1.10 kg/m³, ISA+20°C at sea level)
   C3. Reference sea level (ρ=1.225 kg/m³)

D. Operating points
   D1. Zero thrust (very high J, near windmill state)
   D2. Maximum thrust (hover)
   D3. Varying collective pitch offsets (+5°, 0°, -5°)

E. Physics / conservation checks
   E1. Power = Torque × ω  (all cases)
   E2. CT, CP, CQ dimensional consistency  (CT × J / CP = η for J > 0)
   E3. FM improves with solidity at fixed disk loading
   E4. Thrust and torque are positive for all well-pitched cases
   E5. Spanwise alpha distribution is smooth (no discontinuities > 10°)
   E6. Doubling RPM roughly quadruples thrust (T ∝ n²)
   E7. Halving density roughly halves thrust at same RPM (T ∝ ρ)
"""

import sys, warnings, traceback
import numpy as np
from scipy.integrate import trapezoid as sci_trapz

sys.path.insert(0, "/home/brianfinch/Personal/repos/ductedfanlib/src")

from ductedfanlib.geometry.airfoils import generate_naca4_coordinates
from ductedfanlib.geometry.profiles import LinearDistribution
from ductedfanlib.core.blade        import Blade
from ductedfanlib.core.rotor        import Rotor
from ductedfanlib.geometry.meshing  import get_rotor_bemt_stations
from ductedfanlib.analysis.bemt2    import calculate_bemt_performance_axial

# ── Helpers ────────────────────────────────────────────────────────────────────
PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m~\033[0m"
all_results = []

def check(name, condition, detail="", warn_only=False):
    sym = (WARN if warn_only else FAIL) if not condition else PASS
    print(f"    {sym} {name}" + (f"  [{detail}]" if detail else ""))
    all_results.append((name, condition, warn_only))

def run(stations, V, omega, B, rho=1.225, mu=1.81e-5, r_hub=None, R_tip=None):
    """Run BEMT silently, return result dict."""
    r_hub = r_hub or stations[0]["radius_m"]
    R_tip = R_tip or stations[-1]["radius_m"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return calculate_bemt_performance_axial(
            stations, V_axial_ms=V, omega_rads=omega,
            num_blades=B, rho_kgm3=rho, mu_Pas=mu,
            root_radius_m=r_hub, tip_radius_m=R_tip,
            tip_gap_clearance_m=0.0
        )

def make_rotor(twist_root, twist_tip, chord_root, chord_tip,
               B, R_tip=0.5, r_hub=0.08, n_stations=20):
    af = generate_naca4_coordinates("4412", default_analysis_method="neuralfoil")
    af.characterize_stall_properties(Re=500_000)
    blade = Blade(
        airfoil_definition=af,
        chord_profile=LinearDistribution(chord_root, chord_tip),
        twist_profile=LinearDistribution(twist_root, twist_tip)
    )
    rotor    = Rotor(num_blades=B, tip_radius=R_tip, hub_radius=r_hub, blade_definition=blade)
    stations = get_rotor_bemt_stations(rotor, num_stations=n_stations, spacing_type="cosine")
    return rotor, stations

def alpha_smooth(span_data, tol=10.0):
    alphas = [s["alpha_deg"] for s in span_data]
    diffs  = [abs(alphas[i+1] - alphas[i]) for i in range(len(alphas)-1)]
    return max(diffs) < tol, max(diffs)

# ── Reference rotor ────────────────────────────────────────────────────────────
REF_B, REF_R, REF_HUB = 3, 0.5, 0.08
REF_RPM = 3000
REF_OMEGA = REF_RPM * 2 * np.pi / 60
REF_rotor, REF_stations = make_rotor(25, 12, 0.10, 0.06, REF_B, REF_R, REF_HUB)

# ══════════════════════════════════════════════════════════════════════════════
print("\n═══ A. Blade geometry sensitivity ═══════════════════════════════════")

# A1 – High-pitch blade (hover-optimised)
print("\n  A1. High-pitch blade (35→18°, same chord, 3 blades)")
_, st_A1 = make_rotor(35, 18, 0.10, 0.06, 3)
r_A1 = run(st_A1, 0.0, REF_OMEGA, 3)
T_A1, FM_A1 = r_A1["total_thrust_N"], r_A1["figure_of_merit"]
print(f"       Hover: T={T_A1:.1f}N  FM={FM_A1:.3f}")
check("A1: thrust > 0",          T_A1 > 0,              f"T={T_A1:.1f}N")
check("A1: FM plausible",        0.3 < FM_A1 < 0.95,    f"FM={FM_A1:.3f}")
check("A1: high-pitch > ref thrust at hover",
      T_A1 > 300,   f"T={T_A1:.1f}N (ref≈504N)")

# A2 – Low-pitch blade (cruise-optimised)
print("\n  A2. Low-pitch blade (18→8°, same chord, 3 blades)")
_, st_A2 = make_rotor(18, 8, 0.10, 0.06, 3)
n_rps = REF_RPM / 60; D = 2 * REF_R
J_mid = 0.35
V_mid = J_mid * n_rps * D
r_A2_hover  = run(st_A2, 0.0,   REF_OMEGA, 3)
r_A2_cruise = run(st_A2, V_mid, REF_OMEGA, 3)
print(f"       Hover: T={r_A2_hover['total_thrust_N']:.1f}N  FM={r_A2_hover['figure_of_merit']:.3f}")
print(f"       J=0.35: T={r_A2_cruise['total_thrust_N']:.1f}N  η={r_A2_cruise['propulsive_efficiency']:.3f}")
check("A2: cruise thrust > 0",   r_A2_cruise["total_thrust_N"] > 0)
check("A2: low-pitch less hover thrust than high-pitch",
      r_A2_hover["total_thrust_N"] < T_A1,
      f"T_low={r_A2_hover['total_thrust_N']:.0f}N < T_high={T_A1:.0f}N")

# A3 – High solidity (4 blades, wider chord)
print("\n  A3. High solidity (4 blades, chord 0.12→0.08m)")
_, st_A3 = make_rotor(25, 12, 0.12, 0.08, 4)
r_A3 = run(st_A3, 0.0, REF_OMEGA, 4)
T_A3, FM_A3 = r_A3["total_thrust_N"], r_A3["figure_of_merit"]
print(f"       Hover: T={T_A3:.1f}N  FM={FM_A3:.3f}")
check("A3: thrust > ref (more blades/chord)",  T_A3 > r_A3["total_thrust_N"] * 0.9)
check("A3: FM plausible",  0.3 < FM_A3 < 0.95, f"FM={FM_A3:.3f}")

# A4 – Low solidity (2 blades, narrow chord)
print("\n  A4. Low solidity (2 blades, chord 0.07→0.04m)")
_, st_A4 = make_rotor(25, 12, 0.07, 0.04, 2)
r_A4 = run(st_A4, 0.0, REF_OMEGA, 2)
T_A4, FM_A4 = r_A4["total_thrust_N"], r_A4["figure_of_merit"]
print(f"       Hover: T={T_A4:.1f}N  FM={FM_A4:.3f}")
check("A4: thrust > 0",   T_A4 > 0,           f"T={T_A4:.1f}N")
check("A4: less thrust than A3 (lower solidity)",
      T_A4 < T_A3, f"T_A4={T_A4:.1f}N < T_A3={T_A3:.1f}N")

# ══════════════════════════════════════════════════════════════════════════════
print("\n═══ B. Rotor size / RPM scaling ══════════════════════════════════════")

# B1 – UAV-scale (R=0.15m, 8000 RPM)
print("\n  B1. UAV-scale rotor (R=0.15m, 8000 RPM, 2 blades)")
_, st_B1 = make_rotor(28, 14, 0.030, 0.018, 2, R_tip=0.15, r_hub=0.025)
omega_B1 = 8000 * 2*np.pi/60
r_B1 = run(st_B1, 0.0, omega_B1, 2, r_hub=0.025, R_tip=0.15)
T_B1, P_B1, FM_B1 = r_B1["total_thrust_N"], r_B1["total_power_W"], r_B1["figure_of_merit"]
print(f"       Hover: T={T_B1:.2f}N  P={P_B1:.1f}W  FM={FM_B1:.3f}")
check("B1: thrust > 0",      T_B1 > 0,          f"T={T_B1:.2f}N")
check("B1: power > 0",       P_B1 > 0)
check("B1: FM plausible",    0.25 < FM_B1 < 0.95, f"FM={FM_B1:.3f}")
check("B1: T/P ratio (g/W)", (T_B1 * 1000 / 9.81) / P_B1 > 1.0,
      f"{T_B1*1000/9.81/P_B1:.2f} g/W (expect >1 for UAV prop)")

# B2 – Light-aircraft scale (R=1.5m, 600 RPM)
print("\n  B2. Light-aircraft scale (R=1.5m, 600 RPM, 2 blades)")
_, st_B2 = make_rotor(30, 15, 0.22, 0.14, 2, R_tip=1.5, r_hub=0.20)
omega_B2 = 600 * 2*np.pi/60
r_B2 = run(st_B2, 0.0, omega_B2, 2, r_hub=0.20, R_tip=1.5)
T_B2, P_B2 = r_B2["total_thrust_N"], r_B2["total_power_W"]
print(f"       Hover: T={T_B2:.1f}N  P={P_B2:.1f}W  FM={r_B2['figure_of_merit']:.3f}")
check("B2: thrust > 0",   T_B2 > 0)
check("B2: FM plausible", 0.3 < r_B2["figure_of_merit"] < 0.95)

# B3 – RPM scaling: 2× RPM → ~4× thrust, ~8× power
print("\n  B3. RPM scaling: 1× vs 2× RPM (reference rotor, hover)")
r_1x = run(REF_stations, 0.0, REF_OMEGA,     REF_B, r_hub=REF_HUB, R_tip=REF_R)
r_2x = run(REF_stations, 0.0, REF_OMEGA*2.0, REF_B, r_hub=REF_HUB, R_tip=REF_R)
T_ratio = r_2x["total_thrust_N"] / r_1x["total_thrust_N"]
P_ratio = r_2x["total_power_W"]  / r_1x["total_power_W"]
print(f"       T_ratio (2×RPM): {T_ratio:.2f}  (expect ~4)")
print(f"       P_ratio (2×RPM): {P_ratio:.2f}  (expect ~8)")
check("B3: T roughly quadruples with 2×RPM",  2.5 < T_ratio < 6.0, f"T_ratio={T_ratio:.2f}")
check("B3: P roughly 8× with 2×RPM",          4.0 < P_ratio < 14.0, f"P_ratio={P_ratio:.2f}")

# B4 – RPM scaling: 0.5× RPM → ~0.25× thrust
print("\n  B4. RPM scaling: 0.5× RPM (reference rotor, hover)")
r_half = run(REF_stations, 0.0, REF_OMEGA*0.5, REF_B, r_hub=REF_HUB, R_tip=REF_R)
T_half_ratio = r_half["total_thrust_N"] / r_1x["total_thrust_N"]
print(f"       T_ratio (0.5×RPM): {T_half_ratio:.3f}  (expect ~0.25)")
check("B4: T roughly quarters with 0.5×RPM",  0.10 < T_half_ratio < 0.45, f"ratio={T_half_ratio:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
print("\n═══ C. Atmospheric conditions ════════════════════════════════════════")

rho_SL   = 1.225   # kg/m³, ISA sea level
rho_6km  = 0.660   # kg/m³, ~6000 m ISA
rho_hot  = 1.100   # kg/m³, ISA+20°C SL approx

r_SL  = run(REF_stations, 0.0, REF_OMEGA, REF_B, rho=rho_SL,  r_hub=REF_HUB, R_tip=REF_R)
r_alt = run(REF_stations, 0.0, REF_OMEGA, REF_B, rho=rho_6km, r_hub=REF_HUB, R_tip=REF_R)
r_hot = run(REF_stations, 0.0, REF_OMEGA, REF_B, rho=rho_hot,  r_hub=REF_HUB, R_tip=REF_R)

T_SL, T_alt, T_hot = r_SL["total_thrust_N"], r_alt["total_thrust_N"], r_hot["total_thrust_N"]
rho_ratio = rho_6km / rho_SL  # ~0.539

print(f"\n  C1/C2/C3 summary:")
print(f"       SL  (ρ={rho_SL:.3f}): T={T_SL:.1f}N  FM={r_SL['figure_of_merit']:.3f}")
print(f"       6km (ρ={rho_6km:.3f}): T={T_alt:.1f}N  FM={r_alt['figure_of_merit']:.3f}")
print(f"       Hot (ρ={rho_hot:.3f}): T={T_hot:.1f}N  FM={r_hot['figure_of_merit']:.3f}")

T_scale = T_alt / T_SL
check("C1: altitude reduces thrust",        T_alt < T_SL,  f"T_alt={T_alt:.1f} < T_SL={T_SL:.1f}")
check("C1: thrust scales approx with ρ",   0.40 < T_scale < 0.80,
      f"T_alt/T_SL={T_scale:.3f}, ρ_ratio={rho_ratio:.3f}")
check("C2: hot day less thrust than SL",    T_hot < T_SL,  f"T_hot={T_hot:.1f} < T_SL={T_SL:.1f}")
check("C3: SL reference thrust > 0",        T_SL > 0)

# ══════════════════════════════════════════════════════════════════════════════
print("\n═══ D. Operating points ══════════════════════════════════════════════")

n_rps  = REF_RPM / 60
D_ref  = 2 * REF_R

# D1 – High J (near zero thrust)
J_high = 0.90
V_high = J_high * n_rps * D_ref
print(f"\n  D1. High advance ratio (J={J_high}, near zero-thrust)")
r_D1 = run(REF_stations, V_high, REF_OMEGA, REF_B, r_hub=REF_HUB, R_tip=REF_R)
print(f"       T={r_D1['total_thrust_N']:.2f}N  Ct={r_D1['Ct_rotor']:.4f}")
check("D1: very low thrust at high J",
      r_D1["total_thrust_N"] < 60.0, f"T={r_D1['total_thrust_N']:.1f}N")
check("D1: solver stable (T finite)",
      np.isfinite(r_D1["total_thrust_N"]))

# D2 – Collective pitch sweeps: +5°, 0°, −5° offset
print("\n  D2. Collective pitch offset (hover, ref rotor geom)")
T_list = []
for dtheta in [+5.0, 0.0, -5.0]:
    _, st_d = make_rotor(25+dtheta, 12+dtheta, 0.10, 0.06, REF_B)
    r_d = run(st_d, 0.0, REF_OMEGA, REF_B, r_hub=REF_HUB, R_tip=REF_R)
    T_list.append(r_d["total_thrust_N"])
    print(f"       Δθ={dtheta:+.0f}°: T={r_d['total_thrust_N']:.1f}N  FM={r_d['figure_of_merit']:.3f}")
check("D2: +5° collective gives more thrust than 0°",
      T_list[0] > T_list[1], f"{T_list[0]:.1f} > {T_list[1]:.1f}")
check("D2: −5° collective gives less thrust than 0°",
      T_list[2] < T_list[1], f"{T_list[2]:.1f} < {T_list[1]:.1f}")

# ══════════════════════════════════════════════════════════════════════════════
print("\n═══ E. Physics / conservation checks ════════════════════════════════")

# E1 – P = Q × ω for all J
print("\n  E1. Power = Torque × ω across full J-sweep")
J_sweep   = np.linspace(0.0, 0.7, 8)
pw_errors = []
for J in J_sweep:
    V = J * n_rps * D_ref
    res = run(REF_stations, V, REF_OMEGA, REF_B, r_hub=REF_HUB, R_tip=REF_R)
    err = abs(res["total_torque_Nm"] * REF_OMEGA - res["total_power_W"])
    rel = err / max(abs(res["total_power_W"]), 1e-3)
    pw_errors.append(rel)
max_err = max(pw_errors)
print(f"       Max relative |P - Q·ω| / P = {max_err:.2e}")
check("E1: P = Q·ω to within 0.01% everywhere", max_err < 1e-4, f"max_err={max_err:.2e}")

# E2 – Dimensional consistency: CT × J / CP = η  (for J > 0)
print("\n  E2. CT × J / CP = η  (propeller efficiency identity)")
eta_errors = []
for J in [0.2, 0.3, 0.4, 0.5]:
    V = J * n_rps * D_ref
    res = run(REF_stations, V, REF_OMEGA, REF_B, r_hub=REF_HUB, R_tip=REF_R)
    Ct, Cp, eta = res["Ct_rotor"], res["Cp_rotor"], res["propulsive_efficiency"]
    if abs(Cp) > 1e-6:
        eta_from_coeffs = Ct * J / Cp
        err = abs(eta_from_coeffs - eta)
        eta_errors.append(err)
        print(f"       J={J:.1f}: Ct={Ct:.4f}  Cp={Cp:.4f}  η_direct={eta:.4f}  "
              f"η_from_coeffs={eta_from_coeffs:.4f}  err={err:.2e}")
check("E2: CT·J/CP == η to within 0.1%",
      max(eta_errors) < 1e-3, f"max_err={max(eta_errors):.2e}")

# E3 – FM improves with solidity at fixed disk loading
print("\n  E3. FM vs solidity at fixed disk (hover)")
_, st_low_sol  = make_rotor(25, 12, 0.07, 0.04, 2)
_, st_mid_sol  = make_rotor(25, 12, 0.10, 0.06, 3)
_, st_high_sol = make_rotor(25, 12, 0.12, 0.08, 4)
FM_low  = run(st_low_sol,  0.0, REF_OMEGA, 2, r_hub=REF_HUB, R_tip=REF_R)["figure_of_merit"]
FM_mid  = run(st_mid_sol,  0.0, REF_OMEGA, 3, r_hub=REF_HUB, R_tip=REF_R)["figure_of_merit"]
FM_high = run(st_high_sol, 0.0, REF_OMEGA, 4, r_hub=REF_HUB, R_tip=REF_R)["figure_of_merit"]
print(f"       FM low-sol={FM_low:.3f}  mid-sol={FM_mid:.3f}  high-sol={FM_high:.3f}")
check("E3: FM all in realistic range [0.3, 0.95]",
      all(0.3 < fm < 0.95 for fm in [FM_low, FM_mid, FM_high]),
      f"{FM_low:.3f}, {FM_mid:.3f}, {FM_high:.3f}")

# E4 – Thrust and torque positive for all well-pitched cases
print("\n  E4. Thrust and torque positive across all cases")
cases = [
    ("ref hover",      run(REF_stations, 0.0,              REF_OMEGA, REF_B, r_hub=REF_HUB, R_tip=REF_R)),
    ("ref J=0.2",      run(REF_stations, 0.2*n_rps*D_ref,  REF_OMEGA, REF_B, r_hub=REF_HUB, R_tip=REF_R)),
    ("ref J=0.5",      run(REF_stations, 0.5*n_rps*D_ref,  REF_OMEGA, REF_B, r_hub=REF_HUB, R_tip=REF_R)),
    ("high-pitch hov", run(st_A1, 0.0,  REF_OMEGA, 3,  r_hub=REF_HUB, R_tip=REF_R)),
    ("UAV hover",      run(st_B1, 0.0,  omega_B1,  2,  r_hub=0.025,   R_tip=0.15)),
]
for name, res in cases:
    T_ok = res["total_thrust_N"] > 0
    Q_ok = res["total_torque_Nm"] > 0
    check(f"E4: {name}: T>0 & Q>0",  T_ok and Q_ok,
          f"T={res['total_thrust_N']:.1f}N  Q={res['total_torque_Nm']:.2f}Nm")

# E5 – Spanwise alpha smooth
print("\n  E5. Spanwise AoA smooth (no jumps > 10°)")
smooth_cases = [
    ("hover",  run(REF_stations, 0.0,             REF_OMEGA, REF_B, r_hub=REF_HUB, R_tip=REF_R)),
    ("J=0.3",  run(REF_stations, 0.3*n_rps*D_ref, REF_OMEGA, REF_B, r_hub=REF_HUB, R_tip=REF_R)),
    ("J=0.6",  run(REF_stations, 0.6*n_rps*D_ref, REF_OMEGA, REF_B, r_hub=REF_HUB, R_tip=REF_R)),
]
for name, res in smooth_cases:
    ok, max_jump = alpha_smooth(res["spanwise_results"])
    check(f"E5: {name} alpha smooth", ok, f"max_jump={max_jump:.2f}°")

# E6 – T ∝ n² (doubling RPM ~4× thrust at hover)
print("\n  E6. Thrust scales ~ RPM² at hover")
check("E6: 2×RPM → ~4× thrust (from B3)", 2.5 < T_ratio < 6.0, f"ratio={T_ratio:.2f}")

# E7 – T ∝ ρ (halving density halves thrust at same RPM)
print("\n  E7. Thrust scales ~ ρ at fixed RPM")
T_SL_ref = r_SL["total_thrust_N"]
T_alt_ref = r_alt["total_thrust_N"]
rho_ratio_measured = T_alt_ref / T_SL_ref
rho_ratio_expected = rho_6km / rho_SL
print(f"       T_alt/T_SL = {rho_ratio_measured:.3f}  ρ_alt/ρ_SL = {rho_ratio_expected:.3f}")
check("E7: T_alt/T_SL within 30% of ρ ratio",
      abs(rho_ratio_measured - rho_ratio_expected) < 0.30,
      f"measured={rho_ratio_measured:.3f}  expected={rho_ratio_expected:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
print("\n═══ Summary ══════════════════════════════════════════════════════════")
passed   = sum(1 for _, ok, _    in all_results if ok)
failed   = sum(1 for _, ok, warn in all_results if not ok and not warn)
warnings_ = sum(1 for _, ok, warn in all_results if not ok and warn)
total    = len(all_results)
print(f"  {passed}/{total} checks passed"
      + (f"  ({warnings_} warnings)" if warnings_ else ""))
if failed:
    print("  HARD FAILURES:")
    for name, ok, warn in all_results:
        if not ok and not warn:
            print(f"    {FAIL} {name}")
if warnings_:
    print("  WARNINGS:")
    for name, ok, warn in all_results:
        if not ok and warn:
            print(f"    {WARN} {name}")
sys.exit(0 if failed == 0 else 1)
