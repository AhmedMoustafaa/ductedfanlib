# DuctedFanLib: A Python Framework for Ducted Fan Analysis

**DuctedFanLib** is an open-source Python library for the parametric design, aerodynamic analysis, and performance optimization of ducted fan propulsion systems. It is built to provide researchers, students, and engineers with a flexible and extensible toolkit for exploring the complex aerodynamics of ducted and open rotors.

## Key Features

* **Parametric Geometry Engine**:
    * Define complex rotor blades with variable chord, twist, and airfoil sections along the span.
    * Model duct/shroud profiles using parametric curves like Bezier and Splines.
    * Assemble full `DuctedFan` objects with validation checks to ensure geometric consistency.

* **On-the-Fly Airfoil Analysis**:
    * Generate standard NACA 4-digit airfoil coordinates.
    * Integrates with `neuralfoil` and `AeroSandbox` (XFoil wrapper) to calculate aerodynamic coefficients ($C_L, C_D$) directly from coordinates, removing the need for pre-existing polar tables.
    * Includes the Viterna-Corrigan model for robust post-stall aerodynamics.
    * Supports loading of explicit polar data for validation and specific use cases.

* **Blade Element Momentum Theory (BEMT) Analysis**:
    * A robust, corrected BEMT solver for **axial flight** (including hover, $V_0=0$) that iteratively solves for induction factors and incorporates Glauert's correction for high loading.
    * A specialized BEMT solver for **shrouded rotors in hover**, based on the Dayhoum et al. model, which includes an advanced tip-gap loss model using elliptical integrals.
    * Loss models are configurable to analyze both **ducted** and **open rotor** configurations.

* **Parametric Study and Visualization Tools**:
    * Easily perform parametric sweeps over a range of operating conditions (e.g., advance ratio) or geometric design variables (e.g., blade tip twist).
    * Built-in plotting functions to instantly visualize key performance curves ($C_T$, $C_P$, $\eta$ vs. J), spanwise loading distributions, and conceptual streamtube flow.

## Installation

To get started with DuctedFanLib, clone the repository and install it in your Python environment. A virtual environment is highly recommended.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/AhmedMoustafaa/DuctedFanLib](https://github.com/AhmedMoustafaa/DuctedFanLib)
    cd DuctedFanLib
    ```

2.  **Create and activate a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the package in editable mode with all dependencies:**
    The `-e` flag allows you to make changes to the source code and have them immediately reflected in your environment. The `[dev]` option installs development dependencies like `pytest`.
    ```bash
    pip install -e .[dev]
    ```

### Dependencies

DuctedFanLib relies on a robust scientific Python stack. Main dependencies include:

* `numpy`
* `scipy`
* `matplotlib`
* `pandas`
* `neuralfoil`
* `aerosandbox`

These will be automatically installed via `pip` when you install the package.

## Quickstart: Analyzing an Open Rotor

This example demonstrates how to define a simple open rotor, run a BEMT analysis over a range of advance ratios, and plot the performance curves.

```python
import numpy as np
from ductedfanlib import (
    Blade, Rotor, Duct, DuctedFan,
    LinearDistribution, BezierCurve,
    generate_naca4_coordinates,
    ParametricStudy
)

# 1. Define the Rotor Geometry
print("--- Defining Rotor Design ---")
# Use on-the-fly analysis to get airfoil data for a NACA 4412
naca_airfoil = generate_naca4_coordinates("4412", default_analysis_method="neuralfoil")
naca_airfoil.characterize_stall_properties(Re=500_000) # Pre-calculate stall for post-stall model

# Define parametric chord and twist distributions
twist_dist = LinearDistribution(start_value=20.0, end_value=10.0) # degrees
chord_dist = LinearDistribution(start_value=0.12, end_value=0.06) # meters

# Create blade and rotor objects
blade = Blade(airfoil_definition=naca_airfoil, chord_profile=chord_dist, twist_profile=twist_dist)
rotor = Rotor(num_blades=3, tip_radius=0.5, hub_radius=0.1, blade_definition=blade)

# For an open rotor analysis, create a dummy duct (it won't be used by the BEMT)
dummy_duct = Duct(profile_curve=BezierCurve([[0,1],[1,1]]))
open_rotor_design = DuctedFan(rotor=rotor, duct=dummy_duct, tip_clearance=0.0)

# 2. Set up and Run the Parametric Study
study = ParametricStudy(design=open_rotor_design)

# Define a sweep over a range of advance ratios (J)
j_sweep = np.linspace(0.1, 0.9, 15)

# Run the sweep at a constant RPM
study.sweep_advance_ratio(
    rpm=2500,
    j_range=j_sweep,
    use_dayhoum_model=False  # IMPORTANT: Set to False for open rotor analysis
)

# 3. View Results
print("\n--- Performance Results ---")
if study.sweep_results_df is not None:
    # Print a rounded version of the results dataframe
    print(study.sweep_results_df.round(4).to_string(index=False))

# Plot the main performance curves (CT, CP, eta vs J)
study.plot_performance_curves()

# Plot detailed spanwise loading for a few key operating points
study.plot_spanwise_distributions(j_values_to_plot=[0.2, 0.5, 0.8])