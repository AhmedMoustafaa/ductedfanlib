"""
Provides classes and functions for running parametric studies and plotting results.
"""
from typing import Dict, Any, List, Union, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.interpolate import make_interp_spline
from copy import deepcopy

from .core import DuctedFan, Rotor, Blade
from .geometry.meshing import get_rotor_bemt_stations
from .analysis.bemt2 import (
    calculate_bemt_performance_axial,
    DEFAULT_CLA_2PI
)

@dataclass
class OperatingConditions:
    """A data class to hold operating conditions."""
    axial_velocity_ms: float
    rpm: float
    rho_kgm3: float = 1.225
    mu_Pas: float = 1.81e-5

class ParametricStudy:
    """
    Manages and executes parametric studies for a given DuctedFan design.
    """
    def __init__(self, design: DuctedFan):
        if not isinstance(design, DuctedFan):
            raise TypeError("design must be an instance of the DuctedFan class.")
        self.base_design = design
        self.sweep_results_df = None # For advance ratio sweeps
        self.sensitivity_results_df = None # For design parameter sweeps
        self.spanwise_results = {}

    def sweep_advance_ratio(self, rpm: float, j_range: Union[List, np.ndarray], num_stations: int=21, use_dayhoum_model: bool=True):
        print(f"--- Starting Advance Ratio Sweep for J = {np.min(j_range):.2f} to {np.max(j_range):.2f} ---")
        omega_rads = rpm * (2*np.pi)/60
        revolutions_per_sec = rpm / 60.0
        rotor_diameter = self.base_design.rotor.diameter
        rotor_stations = get_rotor_bemt_stations(self.base_design.rotor, num_stations=num_stations)
        assumed_cla_rad = DEFAULT_CLA_2PI
        try:
            assumed_cla_rad = self.base_design.rotor.blade_definition.airfoil_definition.get_lift_curve_slope(Re=500_000, analysis_method="polar")
        except Exception: pass
        results_list = []
        self.spanwise_results = {}
        for j_val in j_range:
            v_axial_ms = j_val * revolutions_per_sec * rotor_diameter
            op_conds = OperatingConditions(axial_velocity_ms=v_axial_ms, rpm=rpm)
            print(f"  Running for J = {j_val:.3f} (V_axial = {v_axial_ms:.2f} m/s)")
            try:
                bemt_results = calculate_bemt_performance_axial(
                    rotor_stations_data=rotor_stations, V_axial_ms=op_conds.axial_velocity_ms, omega_rads=omega_rads,
                    num_blades=self.base_design.rotor.num_blades, rho_kgm3=op_conds.rho_kgm3, mu_Pas=op_conds.mu_Pas,
                    root_radius_m=self.base_design.rotor.hub_radius, tip_radius_m=self.base_design.rotor.tip_radius,
                    tip_gap_clearance_m=self.base_design.tip_clearance, use_dayhoum_F_gap_model=use_dayhoum_model)
                run_data = {"advance_ratio_J": j_val, "axial_velocity_ms": v_axial_ms, "rpm": rpm, "thrust_N": bemt_results["total_thrust_N"],
                            "torque_Nm": bemt_results["total_torque_Nm"], "power_W": bemt_results["total_power_W"],
                            "efficiency": bemt_results["efficiency"], "Ct": bemt_results["Ct_rotor"], "Cp": bemt_results["Cp_rotor"]}
                results_list.append(run_data)
                self.spanwise_results[f"J={j_val:.3f}"] = bemt_results["spanwise_results"]
            except Exception as e: print(f"    ERROR: BEMT analysis failed for J = {j_val:.3f}. Error: {e}")
        self.sweep_results_df = pd.DataFrame(results_list)
        print("--- Advance Ratio Sweep Finished ---")

    def sweep_design_parameter(self, parameter_path: str, sweep_range: Union[List, np.ndarray], op_conditions: OperatingConditions,
                             num_stations: int=21, use_dayhoum_model: bool=True):
        print(f"--- Starting Design Parameter Sweep for '{parameter_path}' ---")
        results_list = []
        omega_rads = op_conditions.rpm * (2 * np.pi) / 60
        for value in sweep_range:
            print(f"  Running for {parameter_path} = {value}")
            design_variant = deepcopy(self.base_design)
            try:
                def set_nested_attr(obj, path, val):
                    parts = path.split('.');
                    for part in parts[:-1]: obj = getattr(obj, part)
                    setattr(obj, parts[-1], val)
                set_nested_attr(design_variant, parameter_path, value)
            except AttributeError: print(f"    ERROR: Could not find parameter path '{parameter_path}'. Skipping."); continue
            try:
                rotor_stations = get_rotor_bemt_stations(design_variant.rotor, num_stations=num_stations)
                bemt_results = calculate_bemt_performance_axial(
                    rotor_stations_data=rotor_stations, V_axial_ms=op_conditions.axial_velocity_ms, omega_rads=omega_rads,
                    num_blades=design_variant.rotor.num_blades, rho_kgm3=op_conditions.rho_kgm3, mu_Pas=op_conditions.mu_Pas,
                    root_radius_m=design_variant.rotor.hub_radius, tip_radius_m=design_variant.rotor.tip_radius,
                    tip_gap_clearance_m=design_variant.tip_clearance, use_dayhoum_F_gap_model=use_dayhoum_model)
                run_data = {parameter_path: value, "thrust_N": bemt_results["total_thrust_N"], "power_W": bemt_results["total_power_W"],
                            "efficiency": bemt_results["efficiency"], "Ct": bemt_results["Ct_rotor"], "Cp": bemt_results["Cp_rotor"]}
                results_list.append(run_data)
            except Exception as e: print(f"    ERROR: BEMT analysis failed for {parameter_path} = {value}. Error: {e}")
        self.sensitivity_results_df = pd.DataFrame(results_list)
        print("--- Design Parameter Sweep Finished ---")

    def plot_performance_curves(self):
        if self.sweep_results_df is None or self.sweep_results_df.empty:
            print("No advance ratio sweep results to plot. Please run `sweep_advance_ratio` first."); return
        df = self.sweep_results_df.sort_values(by="advance_ratio_J")
        fig, ax1 = plt.subplots(figsize=(10, 6)); color1 = 'tab:blue'
        ax1.set_xlabel("Advance Ratio (J)"); ax1.set_ylabel("Thrust & Power Coefficients", color=color1)
        ax1.plot(df["advance_ratio_J"], df["Ct"], 'o-', color=color1, label="$C_T$ (Thrust Coeff)")
        ax1.plot(df["advance_ratio_J"], df["Cp"] * 10, 's-', color=color1, fillstyle='none', label="$10 \\times C_P$ (Power Coeff)")
        ax1.tick_params(axis='y', labelcolor=color1); ax1.grid(True, linestyle='--')
        ax2 = ax1.twinx(); color2 = 'tab:red'; ax2.set_ylabel("Efficiency (η)", color=color2)
        ax2.plot(df["advance_ratio_J"], df["efficiency"], '^-', color=color2, label="Efficiency (η)")
        ax2.tick_params(axis='y', labelcolor=color2); ax2.set_ylim(bottom=0)
        lines, labels = ax1.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='best'); fig.tight_layout()
        plt.title(f"Performance Curves for '{self.base_design.rotor.blade_definition.airfoil_definition.name}' Rotor", fontsize=16)
        plt.show()

    def plot_sensitivity_curves(self, parameter_path: str, metrics: List[str]):
        if self.sensitivity_results_df is None or self.sensitivity_results_df.empty:
            print("No sensitivity results to plot. Please run `sweep_design_parameter` first."); return
        df = self.sensitivity_results_df.sort_values(by=parameter_path)
        fig, ax1 = plt.subplots(figsize=(10, 6)); ax1.set_xlabel(parameter_path)
        colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange']; axes = [ax1]
        if len(metrics) > 2: print("Plotting more than 2 metrics may result in overlapping y-axes.")
        if 'efficiency' in metrics and ('Ct' in metrics or 'Cp' in metrics): axes.append(ax1.twinx())
        plotted_metrics = 0
        for i, metric in enumerate(metrics):
            if metric not in df.columns: print(f"Warning: Metric '{metric}' not found. Skipping."); continue
            current_ax = axes[1] if metric == 'efficiency' and len(axes) > 1 else axes[0]
            color = colors[i % len(colors)]
            current_ax.set_ylabel(metric, color=color); current_ax.plot(df[parameter_path], df[metric], 'o-', color=color, label=metric)
            current_ax.tick_params(axis='y', labelcolor=color)
            if i > 1 and current_ax != axes[0] and current_ax != axes[1]: current_ax.spines["right"].set_position(("axes", 1.15 * (i-1)))
            plotted_metrics += 1
        fig.tight_layout(); plt.title(f"Sensitivity Analysis: Performance vs. {parameter_path}", fontsize=16)
        lines_all = [ax.get_legend_handles_labels()[0][0] for ax in fig.axes]; labels_all = [ax.get_legend_handles_labels()[1][0] for ax in fig.axes]
        fig.legend(lines_all, labels_all); plt.grid(True); plt.show()

    def plot_spanwise_distributions(self, j_values_to_plot: List[float]):
        if not self.spanwise_results:
            print("No spanwise results to plot. Ensure `sweep_advance_ratio` was run successfully.")
            return

        plot_keys = []
        for j_target in j_values_to_plot:
            closest_key = min(self.spanwise_results.keys(), key=lambda k: abs(float(k.split('=')[1]) - j_target))
            # It's important to only add keys that actually have data
            if self.spanwise_results[closest_key]:
                plot_keys.append(closest_key)
            else:
                print(
                    f"Warning: No valid spanwise results found for J={j_target:.3f} (key: {closest_key}). Skipping this J value.")

        plot_keys = sorted(list(set(plot_keys)))  # Remove duplicates and sort

        if not plot_keys:  # <--- CHECK AGAIN AFTER FILTERING
            print("No valid J values with spanwise data to plot after filtering.")
            return

        plot_metrics = [
            ("alpha_deg", "Angle of Attack (deg)"),
            ("phi_deg", "Inflow Angle (deg)"),
            ("Cl", "Lift Coefficient (Cl)"),
            ("dT_dr_N_m", "Thrust Loading (dT/dr) [N/m]"),
            ("dQ_dr_Nm_m", "Torque Loading (dQ/dr) [Nm/m]"),
            ("F_sh", "Total Loss Factor (F)")
        ]

        num_plots = len(plot_metrics)
        fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots), sharex=True)
        if num_plots == 1:
            axes = [axes]  # Ensure axes is always iterable for single plot case

        for key in plot_keys:
            data = self.spanwise_results[key]
            df_span = pd.DataFrame(data).sort_values(by="eta")
            eta = df_span["eta"]

            for i, (metric_key, ylabel) in enumerate(plot_metrics):
                if metric_key in df_span.columns:
                    axes[i].plot(eta, df_span[metric_key], '.-', label=f"J={float(key.split('=')[1]):.2f}")
                else:
                    print(
                        f"Warning: Metric '{metric_key}' not found for J={float(key.split('=')[1]):.2f}. Skipping plot for this metric/J.")

        for i, (_, ylabel) in enumerate(plot_metrics):
            axes[i].set_ylabel(ylabel)
            axes[i].grid(True)
            axes[i].legend(title="Advance Ratio")
            axes[i].set_xlim([0, 1])

        axes[-1].set_xlabel("Normalized Blade Span (eta)")
        fig.suptitle(
            f"Spanwise Distributions for '{self.base_design.rotor.blade_definition.airfoil_definition.name}' Rotor",
            fontsize=16)
        fig.tight_layout(rect=[0, 0.02, 1, 0.96])
        plt.show()

    def plot_streamtube_visualization(self, j_value_to_plot: float, num_streamlines: int = 10):
        if self.sweep_results_df is None or not self.spanwise_results: print("No results to visualize."); return
        closest_key = min(self.spanwise_results.keys(), key=lambda k: abs(float(k.split('=')[1]) - j_value_to_plot))
        j_val = float(closest_key.split('=')[1]); df_span = pd.DataFrame(self.spanwise_results[closest_key])
        if 'a' not in df_span.columns: print("Warning: 'a' (axial induction) not found in spanwise results. Cannot plot streamtube."); return
        op_cond_row = self.sweep_results_df[self.sweep_results_df['advance_ratio_J'] == j_val].iloc[0]; v_axial_ms = op_cond_row['axial_velocity_ms']
        fig, ax = plt.subplots(figsize=(12, 7))
        duct_points = self.base_design.duct.get_profile_points(num_points=101); duct_z = duct_points[:, 0] - self.base_design.rotor_axial_position
        ax.plot(duct_z, duct_points[:, 1], 'k-', linewidth=1.5, label='Duct Profile'); ax.plot(duct_z, -duct_points[:, 1], 'k-', linewidth=1.5)
        rotor_z = 0.0
        ax.plot([rotor_z, rotor_z], [self.base_design.rotor.hub_radius, self.base_design.rotor.tip_radius], 'b-', linewidth=5, alpha=0.8, label='Rotor')
        ax.plot([rotor_z, rotor_z], [-self.base_design.rotor.hub_radius, -self.base_design.rotor.tip_radius], 'b-', linewidth=5, alpha=0.8)
        radii_at_disk = df_span['radius_m'].values; axial_induction_a = df_span['a'].values
        sort_indices = np.argsort(radii_at_disk); radii_at_disk = radii_at_disk[sort_indices]; axial_induction_a = axial_induction_a[sort_indices]
        a_interp = make_interp_spline(radii_at_disk, axial_induction_a, k=1)
        start_radii = np.linspace(self.base_design.rotor.hub_radius, self.base_design.rotor.tip_radius, num_streamlines)
        plot_upstream_z = -1.5 * self.base_design.rotor.diameter; plot_downstream_z = 2.0 * self.base_design.rotor.diameter
        for r_disk in start_radii:
            a_local = a_interp(r_disk); v_disk = v_axial_ms * (1 + a_local)
            z_points = np.array([plot_upstream_z, rotor_z, plot_downstream_z]); r_points = np.zeros(3); r_points[1] = r_disk

            upstream_ratio = v_disk / v_axial_ms if v_axial_ms > 1e-6 else 10.0 # Avoid division by zero
            if upstream_ratio >= 0:
                r_points[0] = r_disk * np.sqrt(upstream_ratio)
            else:
                print(f"Warning: Cannot draw upstream streamline for r={r_disk:.3f}m, flow state is complex (a < -1).")
                continue

            v_wake = v_axial_ms * (1 + 2 * a_local)
            if v_disk > 1e-6 and v_wake > 1e-6:
                downstream_ratio = v_disk / v_wake
                if downstream_ratio >= 0:
                    r_points[2] = r_disk * np.sqrt(downstream_ratio)
                else:
                    print(f"Warning: Cannot draw downstream streamline for r={r_disk:.3f}m, flow state is complex (a < -0.5).")
                    continue
            else: r_points[2] = r_disk

            if len(z_points) > 2 and not np.any(np.isnan(r_points)):
                spline = make_interp_spline(z_points, r_points, k=2); z_smooth = np.linspace(plot_upstream_z, plot_downstream_z, 200); r_smooth = spline(z_smooth)
                ax.plot(z_smooth, r_smooth, 'c-', linewidth=0.8); ax.plot(z_smooth, -r_smooth, 'c-', linewidth=0.8)
        ax.set_xlabel("Axial Position (m)"); ax.set_ylabel("Radial Position (m)"); ax.set_title(f"Streamtube Visualization for J = {j_val:.3f}")
        ax.axis('equal'); ax.grid(True, linestyle=':', alpha=0.7); ax.legend(); plt.show()
