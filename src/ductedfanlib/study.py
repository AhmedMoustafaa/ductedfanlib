"""
Parametric studies and post-processing for ductedfanlib.


"""
from __future__ import annotations

from typing import Dict, Any, List, Union, Optional
from copy import deepcopy
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

from .core import DuctedFan, Rotor, Blade, OperatingConditions
from .geometry.meshing import get_rotor_bemt_stations
from .analysis.bemt2 import calculate_bemt_performance_axial


# ── helpers ───────────────────────────────────────────────────────────────────

def _set_nested_attr(obj: Any, path: str, value: Any) -> None:
    """Set a dotted attribute path on an object, e.g. 'rotor.collective_pitch_deg'."""
    parts = path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def _get_nested_attr(obj: Any, path: str) -> Any:
    """Get a dotted attribute path from an object."""
    parts = path.split(".")
    for part in parts:
        obj = getattr(obj, part)
    return obj


# ── ParametricStudy ────────────────────────────────────────────────────────────

class ParametricStudy:
    """
    Manages and executes parametric studies for a given DuctedFan design.

    Usage
    -----
    study = ParametricStudy(design)
    study.sweep_advance_ratio(rpm=3000, j_range=np.linspace(0.05, 0.7, 14))
    study.plot_performance_curves()
    study.plot_spanwise_distributions([0.2, 0.4, 0.6])
    study.export_results("my_results.csv")
    """

    def __init__(self, design: DuctedFan, num_stations: int = 21,
                 use_dayhoum_model: bool = False):
        if not isinstance(design, DuctedFan):
            raise TypeError("design must be a DuctedFan instance.")
        self.base_design          = design
        self.num_stations         = num_stations
        self.use_dayhoum_model    = use_dayhoum_model
        self.sweep_results_df:    Optional[pd.DataFrame] = None
        self.sensitivity_results_df: Optional[pd.DataFrame] = None
        self.spanwise_results:    Dict[str, List[Dict]] = {}

    # ── advance-ratio sweep ───────────────────────────────────────────────────

    def sweep_advance_ratio(
        self,
        rpm: float,
        j_range: Union[List[float], np.ndarray],
        rho: float = 1.225,
        mu:  float = 1.81e-5,
    ) -> pd.DataFrame:
        """
        Run BEMT at each J in j_range and store results.

        Parameters
        ----------
        rpm     : rotational speed (RPM)
        j_range : advance ratio values (0 = hover)
        rho     : air density (kg/m³)
        mu      : dynamic viscosity (Pa·s)

        Returns
        -------
        pd.DataFrame with one row per J point.
        """
        j_range   = np.asarray(j_range, dtype=float)
        omega     = rpm * 2.0 * np.pi / 60.0
        n_rps     = rpm / 60.0
        D         = self.base_design.rotor.diameter
        stations  = get_rotor_bemt_stations(self.base_design.rotor,
                                            num_stations=self.num_stations)
        r_hub     = self.base_design.rotor.hub_radius
        R_tip     = self.base_design.rotor.tip_radius
        gap       = self.base_design.tip_clearance
        B         = self.base_design.rotor.num_blades

        print(f"--- Advance-ratio sweep  J = {j_range.min():.2f} … {j_range.max():.2f} "
              f"({len(j_range)} points) ---")

        rows = []
        self.spanwise_results = {}

        for J in j_range:
            V = float(J) * n_rps * D
            print(f"  J={J:.3f}  V={V:.2f} m/s", end="  ", flush=True)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = calculate_bemt_performance_axial(
                        rotor_stations_data   = stations,
                        V_axial_ms            = V,
                        omega_rads            = omega,
                        num_blades            = B,
                        rho_kgm3              = rho,
                        mu_Pas                = mu,
                        root_radius_m         = r_hub,
                        tip_radius_m          = R_tip,
                        tip_gap_clearance_m   = gap,
                        use_dayhoum_F_gap_model = self.use_dayhoum_model,
                    )
                rows.append({
                    "J":          J,
                    "V_ms":       V,
                    "rpm":        rpm,
                    "thrust_N":   res["total_thrust_N"],
                    "torque_Nm":  res["total_torque_Nm"],
                    "power_W":    res["total_power_W"],
                    "eta":        res["propulsive_efficiency"],
                    "FM":         res["figure_of_merit"],
                    "Ct":         res["Ct_rotor"],
                    "Cp":         res["Cp_rotor"],
                    "Cq":         res["Cq_rotor"],
                })
                key = f"J={J:.3f}"
                self.spanwise_results[key] = res["spanwise_results"]
                print(f"T={res['total_thrust_N']:.1f}N  η={res['propulsive_efficiency']:.3f}")
            except Exception as exc:
                print(f"FAILED: {exc}")

        self.sweep_results_df = pd.DataFrame(rows)
        print("--- Sweep finished ---")
        return self.sweep_results_df

    # ── design-parameter sweep ────────────────────────────────────────────────

    def sweep_design_parameter(
        self,
        parameter_path: str,
        sweep_range: Union[List, np.ndarray],
        op_conditions: OperatingConditions,
    ) -> pd.DataFrame:
        """
        Vary a single design parameter and run BEMT at each value.

        Parameters
        ----------
        parameter_path : dotted attribute path, e.g. 'rotor.collective_pitch_deg'
        sweep_range    : iterable of values to set
        op_conditions  : OperatingConditions for each run

        Returns
        -------
        pd.DataFrame with one row per parameter value.
        """
        print(f"--- Design-parameter sweep: '{parameter_path}' ---")
        omega    = op_conditions.omega_rads
        r_hub    = self.base_design.rotor.hub_radius
        gap      = self.base_design.tip_clearance

        rows = []
        for value in sweep_range:
            print(f"  {parameter_path} = {value}", end="  ", flush=True)
            variant = deepcopy(self.base_design)
            try:
                _set_nested_attr(variant, parameter_path, value)
            except AttributeError as exc:
                print(f"SKIP (bad path): {exc}")
                continue

            try:
                stations = get_rotor_bemt_stations(variant.rotor,
                                                   num_stations=self.num_stations)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = calculate_bemt_performance_axial(
                        rotor_stations_data     = stations,
                        V_axial_ms              = op_conditions.axial_velocity_ms,
                        omega_rads              = omega,
                        num_blades              = variant.rotor.num_blades,
                        rho_kgm3                = op_conditions.rho_kgm3,
                        mu_Pas                  = op_conditions.mu_Pas,
                        root_radius_m           = r_hub,
                        tip_radius_m            = variant.rotor.tip_radius,
                        tip_gap_clearance_m     = gap,
                        use_dayhoum_F_gap_model = self.use_dayhoum_model,
                    )
                rows.append({
                    parameter_path: value,
                    "thrust_N":  res["total_thrust_N"],
                    "power_W":   res["total_power_W"],
                    "eta":       res["propulsive_efficiency"],
                    "FM":        res["figure_of_merit"],
                    "Ct":        res["Ct_rotor"],
                    "Cp":        res["Cp_rotor"],
                })
                print(f"T={res['total_thrust_N']:.1f}N")
            except Exception as exc:
                print(f"FAILED: {exc}")

        self.sensitivity_results_df = pd.DataFrame(rows)
        print("--- Parameter sweep finished ---")
        return self.sensitivity_results_df

    # ── plots ─────────────────────────────────────────────────────────────────

    def plot_performance_curves(self, show: bool = True) -> plt.Figure:
        """Plot Ct, Cp, and propulsive efficiency vs advance ratio."""
        if self.sweep_results_df is None or self.sweep_results_df.empty:
            raise RuntimeError("Run sweep_advance_ratio first.")

        df  = self.sweep_results_df.sort_values("J")
        fig, ax1 = plt.subplots(figsize=(10, 6))

        color1 = "tab:blue"
        ax1.set_xlabel("Advance Ratio J")
        ax1.set_ylabel("Thrust / Power Coefficients", color=color1)
        ax1.plot(df["J"], df["Ct"],       "o-",  color=color1, label="$C_T$")
        ax1.plot(df["J"], df["Cp"] * 10,  "s--", color=color1,
                 fillstyle="none", label="$10\\times C_P$")
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.grid(True, linestyle="--", alpha=0.5)

        ax2 = ax1.twinx()
        color2 = "tab:red"
        ax2.set_ylabel("Propulsive Efficiency η", color=color2)
        ax2.plot(df["J"], df["eta"], "^-", color=color2, label="η")
        ax2.set_ylim(0, 1.05)
        ax2.tick_params(axis="y", labelcolor=color2)

        lines  = ax1.get_legend_handles_labels()
        lines2 = ax2.get_legend_handles_labels()
        ax2.legend(lines[0] + lines2[0], lines[1] + lines2[1], loc="best")

        name = getattr(
            self.base_design.rotor.blade_definition.airfoil_definition, "name", "")
        fig.suptitle(f"Performance Curves — {name} rotor", fontsize=14)
        fig.tight_layout()
        if show:
            plt.show()
        return fig

    def plot_spanwise_distributions(
        self,
        j_values: List[float],
        show: bool = True,
    ) -> plt.Figure:
        """Plot spanwise α, φ, Cl, dT/dr, dQ/dr and F for selected J values."""
        if not self.spanwise_results:
            raise RuntimeError("Run sweep_advance_ratio first.")

        # resolve closest stored keys
        keys = []
        for j_target in j_values:
            best = min(self.spanwise_results,
                       key=lambda k: abs(float(k.split("=")[1]) - j_target))
            if self.spanwise_results[best]:
                keys.append(best)
        keys = sorted(set(keys))
        if not keys:
            raise ValueError("No valid spanwise data for the requested J values.")

        metrics = [
            ("alpha_deg",    "Angle of Attack (°)"),
            ("phi_deg",      "Inflow Angle (°)"),
            ("Cl",           "Lift Coefficient Cl"),
            ("dT_dr_N_m",    "Thrust Loading dT/dr (N/m)"),
            ("dQ_dr_Nm_m",   "Torque Loading dQ/dr (Nm/m)"),
            ("F_sh",         "Loss Factor F"),
        ]

        fig, axes = plt.subplots(len(metrics), 1,
                                 figsize=(10, 4 * len(metrics)), sharex=True)

        for key in keys:
            df = pd.DataFrame(self.spanwise_results[key]).sort_values("eta")
            J_label = float(key.split("=")[1])
            for i, (col, _) in enumerate(metrics):
                if col in df.columns:
                    axes[i].plot(df["eta"], df[col], ".-", label=f"J={J_label:.2f}")

        for i, (_, ylabel) in enumerate(metrics):
            axes[i].set_ylabel(ylabel)
            axes[i].grid(True, linestyle="--", alpha=0.5)
            axes[i].legend(title="Advance Ratio", fontsize=8)
            axes[i].set_xlim(0, 1)

        axes[-1].set_xlabel("Normalised Span η")
        name = getattr(
            self.base_design.rotor.blade_definition.airfoil_definition, "name", "")
        fig.suptitle(f"Spanwise Distributions — {name} rotor", fontsize=14)
        fig.tight_layout(rect=[0, 0.02, 1, 0.97])
        if show:
            plt.show()
        return fig

    def plot_sensitivity(
        self,
        parameter_path: str,
        metrics: List[str] = ("thrust_N", "eta"),
        show: bool = True,
    ) -> plt.Figure:
        """Plot one or two metrics from a design-parameter sweep."""
        if self.sensitivity_results_df is None or self.sensitivity_results_df.empty:
            raise RuntimeError("Run sweep_design_parameter first.")

        df  = self.sensitivity_results_df.sort_values(parameter_path)
        fig, ax1 = plt.subplots(figsize=(10, 5))
        colors = ["tab:blue", "tab:red", "tab:green", "tab:orange"]

        ax1.set_xlabel(parameter_path)
        axes = [ax1]
        if len(metrics) > 1:
            axes.append(ax1.twinx())

        for i, metric in enumerate(metrics[:len(axes)]):
            if metric not in df.columns:
                print(f"Warning: metric '{metric}' not found, skipping.")
                continue
            ax = axes[i]
            color = colors[i % len(colors)]
            ax.plot(df[parameter_path], df[metric], "o-", color=color, label=metric)
            ax.set_ylabel(metric, color=color)
            ax.tick_params(axis="y", labelcolor=color)

        fig.legend(loc="upper right", bbox_to_anchor=(0.88, 0.88))
        fig.suptitle(f"Sensitivity: {parameter_path}", fontsize=14)
        fig.tight_layout()
        if show:
            plt.show()
        return fig

    def plot_streamtube(
        self,
        j_value: float,
        num_streamlines: int = 10,
        show: bool = True,
    ) -> plt.Figure:
        """
        Streamtube visualisation for a given J.

        Uses induced axial velocity v_i (= a × ω × R_tip for hover, a × V_inf for 
        forward flight) to compute stream-tube contraction correctly.
        """
        if self.sweep_results_df is None or not self.spanwise_results:
            raise RuntimeError("Run sweep_advance_ratio first.")

        key   = min(self.spanwise_results,
                    key=lambda k: abs(float(k.split("=")[1]) - j_value))
        J_val = float(key.split("=")[1])
        df_sw = pd.DataFrame(self.spanwise_results[key]).sort_values("radius_m")

        row   = self.sweep_results_df[
            np.isclose(self.sweep_results_df["J"], J_val)].iloc[0]
        V_inf  = float(row["V_ms"])
        omega  = float(row["rpm"]) * 2.0 * np.pi / 60.0
        R_tip  = self.base_design.rotor.tip_radius
        hover  = V_inf < 0.5

        r_arr = df_sw["radius_m"].values
        a_arr = df_sw["a"].values
        a_spl = make_interp_spline(r_arr, a_arr, k=1)

        fig, ax = plt.subplots(figsize=(12, 6))

        # duct outline
        try:
            dp = self.base_design.duct.get_profile_points(num_points=101)
            dz = dp[:, 0] - self.base_design.rotor_axial_position
            ax.plot(dz,  dp[:, 1], "k-", lw=1.5, label="Duct")
            ax.plot(dz, -dp[:, 1], "k-", lw=1.5)
        except Exception:
            pass

        # rotor disk
        r_hub = self.base_design.rotor.hub_radius
        ax.plot([0, 0], [ r_hub,  R_tip], "b-", lw=5, alpha=0.8, label="Rotor")
        ax.plot([0, 0], [-r_hub, -R_tip], "b-", lw=5, alpha=0.8)

        z_up   = -1.5 * 2 * R_tip
        z_down =  2.0 * 2 * R_tip

        for r_disk in np.linspace(r_hub, R_tip * 0.97, num_streamlines):
            a_loc = float(a_spl(r_disk))

            if hover:
                # v_i is the induced axial velocity normalised to omega*R_tip
                v_i   = a_loc * omega * R_tip
                v_disk = v_i
                # upstream: flow is drawn in from rest, radius expands to infinity
                # approximate upstream contraction: r_up = r_disk * (v_disk/v_ref)^0.5
                # use v_ref = v_disk at a large distance -> limit expansion to 2x
                r_up   = min(r_disk * 2.0, R_tip * 1.1)
                v_wake = 2.0 * v_i
                r_down = r_disk * np.sqrt(v_disk / v_wake) if v_wake > 1e-6 else r_disk
            else:
                v_disk = V_inf * (1.0 + a_loc)
                v_wake = V_inf * (1.0 + 2.0 * a_loc)
                r_up   = (r_disk * np.sqrt(v_disk / V_inf)
                          if V_inf > 1e-6 else r_disk * 1.5)
                r_down = (r_disk * np.sqrt(v_disk / v_wake)
                          if v_wake > 1e-6 else r_disk)

            if not (np.isfinite(r_up) and np.isfinite(r_down)):
                continue

            z_pts = np.array([z_up, 0.0, z_down])
            r_pts = np.array([r_up, r_disk, r_down])
            try:
                spl     = make_interp_spline(z_pts, r_pts, k=2)
                z_dense = np.linspace(z_up, z_down, 200)
                r_dense = spl(z_dense)
                ax.plot(z_dense,  r_dense, "c-", lw=0.8)
                ax.plot(z_dense, -r_dense, "c-", lw=0.8)
            except Exception:
                pass

        ax.set_xlabel("Axial position (m)")
        ax.set_ylabel("Radial position (m)")
        ax.set_title(f"Streamtube — J={J_val:.3f}")
        ax.set_aspect("equal")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend()
        fig.tight_layout()
        if show:
            plt.show()
        return fig

    # ── export / summary ──────────────────────────────────────────────────────

    def export_results(self, path: str, fmt: str = "csv") -> None:
        """
        Export sweep results to a file.

        Parameters
        ----------
        path : output file path (e.g. 'results.csv' or 'results.json')
        fmt  : 'csv' or 'json'
        """
        if self.sweep_results_df is None or self.sweep_results_df.empty:
            raise RuntimeError("No sweep results to export. Run sweep_advance_ratio first.")
        if fmt == "csv":
            self.sweep_results_df.to_csv(path, index=False)
        elif fmt == "json":
            self.sweep_results_df.to_json(path, orient="records", indent=2)
        else:
            raise ValueError(f"Unknown format '{fmt}'. Use 'csv' or 'json'.")
        print(f"Results exported to {path}")

    def summary_table(self) -> pd.DataFrame:
        """
        Return a formatted summary of the advance-ratio sweep.
        Columns: J, V(m/s), T(N), P(W), Ct, Cp, η, FM.
        """
        if self.sweep_results_df is None or self.sweep_results_df.empty:
            raise RuntimeError("Run sweep_advance_ratio first.")
        df = self.sweep_results_df.sort_values("J").reset_index(drop=True)
        display_cols = ["J", "V_ms", "thrust_N", "power_W", "Ct", "Cp", "eta", "FM"]
        display_cols = [c for c in display_cols if c in df.columns]
        return df[display_cols].round(4)
