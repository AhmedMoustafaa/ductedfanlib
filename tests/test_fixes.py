"""
Integration test covering every fix made in this session.
"""
import sys, warnings, os, tempfile
import numpy as np
sys.path.insert(0, "/home/brianfinch/Personal/repos/ductedfanlib/src")

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
results = []

def check(name, condition, detail=""):
    print(f"  {'  ' if not name.startswith('──') else ''}{PASS if condition else FAIL} {name}"
          + (f"  [{detail}]" if detail else ""))
    results.append((name, condition))

# ── 1. Top-level imports ──────────────────────────────────────────────────────
print("\n── 1. Public API imports (all 17 names) ─────────────────────────────")
import ductedfanlib as df
expected = [
    "Airfoil","generate_naca4_coordinates","load_airfoil_from_file",
    "LinearDistribution","ConstantDistribution","PolynomialDistribution",
    "BezierCurve","SplineCurve","get_rotor_bemt_stations",
    "Blade","Rotor","Duct","DuctedFan","OperatingConditions",
    "calculate_bemt_performance_axial",
    "calculate_ideal_performance_hover","calculate_ideal_performance_axial",
    "run_bemt_analysis","ParametricStudy",
]
for name in expected:
    check(f"df.{name} importable", hasattr(df, name))

# ── 2. OperatingConditions ────────────────────────────────────────────────────
print("\n── 2. OperatingConditions ───────────────────────────────────────────")
from ductedfanlib import OperatingConditions
op = OperatingConditions(axial_velocity_ms=0.0, rpm=3000)
check("omega_rads property", abs(op.omega_rads - 3000*2*np.pi/60) < 1e-9)
check("is_hover property",   op.is_hover)
op2 = OperatingConditions(axial_velocity_ms=20.0, rpm=3000)
check("is_hover=False for V=20",  not op2.is_hover)
try:
    OperatingConditions(axial_velocity_ms=0, rpm=-100)
    check("negative rpm raises ValueError", False)
except ValueError:
    check("negative rpm raises ValueError", True)

# ── 3. Duplicate OperatingConditions gone from study ─────────────────────────
print("\n── 3. No duplicate OperatingConditions in study.py ─────────────────")
import inspect, ductedfanlib.study as study_mod
check("study.py imports OC from core",
      "OperatingConditions" not in [
          name for name, obj in inspect.getmembers(study_mod)
          if inspect.isclass(obj) and obj.__module__ == "ductedfanlib.study"
      ])

# ── 4. analysis manager importable by module path ────────────────────────────
print("\n── 4. analysis/manager.py importable (no space in filename) ─────────")
from ductedfanlib.analysis.manager import run_bemt_analysis
check("run_bemt_analysis importable from analysis.manager", callable(run_bemt_analysis))

# ── 5. pyproject.toml non-empty and no self-reference ────────────────────────
print("\n── 5. pyproject.toml valid ──────────────────────────────────────────")
with open("/home/brianfinch/Personal/repos/ductedfanlib/pyproject.toml") as f:
    toml_text = f.read()
check("pyproject.toml non-empty",      len(toml_text) > 100)
check("build-system defined",          "[build-system]" in toml_text)
check("no self-referential dependency","ductedfanlib" not in toml_text[toml_text.find("dependencies"):])

# ── 6. requirements.txt no self-reference ────────────────────────────────────
print("\n── 6. requirements.txt clean ────────────────────────────────────────")
with open("/home/brianfinch/Personal/repos/ductedfanlib/requirements.txt") as f:
    req_text = f.read()
check("no ductedfanlib in requirements.txt", "ductedfanlib" not in req_text)
check("numpy listed",    "numpy" in req_text)
check("scipy listed",    "scipy" in req_text)
check("matplotlib listed","matplotlib" in req_text)

# ── 7. Airfoil neuralfoil fix (CL != 0 at alpha=5°) ─────────────────────────
print("\n── 7. Airfoil neuralfoil fix ─────────────────────────────────────────")
from ductedfanlib import generate_naca4_coordinates
af = generate_naca4_coordinates("4412", default_analysis_method="neuralfoil")
af.characterize_stall_properties(Re=500_000)
cl, cd = af.get_lift_drag_coeffs(5.0, 500_000, apply_viterna_poststall=False)
check("CL at 5° is non-zero (neuralfoil branch fixed)",  cl > 0.7, f"CL={cl:.3f}")
check("CD at 5° is positive and < 0.05",  0 < cd < 0.05, f"CD={cd:.4f}")

# ── 8. ParametricStudy full workflow ─────────────────────────────────────────
print("\n── 8. ParametricStudy end-to-end ────────────────────────────────────")
from ductedfanlib import (Blade, Rotor, Duct, DuctedFan, LinearDistribution,
                           BezierCurve, ParametricStudy, OperatingConditions)

blade  = Blade(airfoil_definition=af,
               chord_profile=LinearDistribution(0.10, 0.06),
               twist_profile=LinearDistribution(25.0, 12.0))
rotor  = Rotor(num_blades=3, tip_radius=0.5, hub_radius=0.08,
               blade_definition=blade)
duct   = Duct(profile_curve=BezierCurve([[0,0.6],[0.5,0.6],[1.0,0.6]]))
design = DuctedFan(rotor=rotor, duct=duct, tip_clearance=0.0)
study  = ParametricStudy(design, num_stations=15)

# sweep
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    df_res = study.sweep_advance_ratio(rpm=3000, j_range=np.linspace(0.0, 0.6, 7))

check("sweep returns DataFrame",       hasattr(df_res, "columns"))
check("7 rows in results",             len(df_res) == 7, f"rows={len(df_res)}")
check("thrust positive at J=0",        df_res.iloc[0]["thrust_N"] > 0,
      f"T={df_res.iloc[0]['thrust_N']:.1f}N")
check("thrust positive at J=0.3",      df_res.iloc[3]["thrust_N"] > 0)
check("eta column present",            "eta" in df_res.columns)
check("FM column present",             "FM" in df_res.columns)
check("spanwise_results populated",    len(study.spanwise_results) == 7)

# summary_table
tbl = study.summary_table()
check("summary_table returns DataFrame", hasattr(tbl, "columns"))
check("summary_table has J column",      "J" in tbl.columns)

# export CSV
with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
    tmp_csv = f.name
study.export_results(tmp_csv, fmt="csv")
import pandas as pd
df_reloaded = pd.read_csv(tmp_csv)
check("CSV export round-trip",  len(df_reloaded) == len(df_res))
os.unlink(tmp_csv)

# export JSON
with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
    tmp_json = f.name
study.export_results(tmp_json, fmt="json")
df_json = pd.read_json(tmp_json)
check("JSON export round-trip", len(df_json) == len(df_res))
os.unlink(tmp_json)

# ── 9. sweep_design_parameter (set_nested_attr fix) ──────────────────────────
print("\n── 9. sweep_design_parameter (set_nested_attr fix) ──────────────────")
op_hover = OperatingConditions(axial_velocity_ms=0.0, rpm=3000)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    df_sens = study.sweep_design_parameter(
        parameter_path="rotor.collective_pitch_deg",
        sweep_range=[-5.0, 0.0, 5.0],
        op_conditions=op_hover,
    )
check("sensitivity returns 3 rows",    len(df_sens) == 3, f"rows={len(df_sens)}")
check("thrust increases with pitch",
      df_sens.iloc[2]["thrust_N"] > df_sens.iloc[1]["thrust_N"] > df_sens.iloc[0]["thrust_N"],
      f"T: {df_sens['thrust_N'].tolist()}")

# ── 10. run_bemt_analysis convenience wrapper ─────────────────────────────────
print("\n── 10. run_bemt_analysis wrapper ────────────────────────────────────")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    res_hover   = run_bemt_analysis(design, OperatingConditions(0.0,  3000))
    res_forward = run_bemt_analysis(design, OperatingConditions(15.0, 3000))
check("hover thrust > 0",   res_hover["total_thrust_N"] > 0,
      f"T={res_hover['total_thrust_N']:.1f}N")
check("forward thrust > 0", res_forward["total_thrust_N"] > 0,
      f"T={res_forward['total_thrust_N']:.1f}N")
check("FM key present",     "figure_of_merit" in res_hover)
check("eta key present",    "propulsive_efficiency" in res_forward)

# ── 11. plot_streamtube hover safety (no div-by-zero) ────────────────────────
print("\n── 11. plot_streamtube at hover (no crash) ───────────────────────────")
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig = study.plot_streamtube(j_value=0.0, show=False)
    check("plot_streamtube(J=0) returns Figure", fig is not None)
except Exception as exc:
    check("plot_streamtube(J=0) no crash", False, str(exc))

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n── Summary ───────────────────────────────────────────────────────────")
passed = sum(1 for _, ok in results if ok)
failed = len(results) - passed
print(f"  {passed}/{len(results)} checks passed")
if failed:
    print("  FAILED:")
    for name, ok in results:
        if not ok:
            print(f"    {FAIL} {name}")
    sys.exit(1)
else:
    print("  All checks passed.")
