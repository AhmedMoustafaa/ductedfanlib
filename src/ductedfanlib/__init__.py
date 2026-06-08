"""
DuctedFanLib — Ducted Fan Design, Analysis, and Post-processing
===============================================================

Quick start
-----------
>>> from ductedfanlib import (
...     Airfoil, generate_naca4_coordinates,
...     Blade, Rotor, Duct, DuctedFan,
...     OperatingConditions, ParametricStudy,
...     calculate_bemt_performance_axial,
...     run_bemt_analysis,
... )
"""

__version__ = "0.0.2"

# ── geometry ──────────────────────────────────────────────────────────────────
from .geometry.airfoils  import Airfoil, generate_naca4_coordinates, load_airfoil_from_file
from .geometry.profiles  import LinearDistribution, ConstantDistribution, PolynomialDistribution
from .geometry.curves    import BezierCurve, SplineCurve
from .geometry.meshing   import get_rotor_bemt_stations

# ── core design objects ───────────────────────────────────────────────────────
from .core.blade             import Blade
from .core.rotor             import Rotor
from .core.duct              import Duct
from .core.design            import DuctedFan
from .core.OperatingConditions import OperatingConditions

# ── analysis ──────────────────────────────────────────────────────────────────
from .analysis.bemt2   import calculate_bemt_performance_axial
from .analysis.adt     import calculate_ideal_performance_hover, calculate_ideal_performance_axial
from .analysis.manager import run_bemt_analysis

# ── study / post-processing ───────────────────────────────────────────────────
from .study import ParametricStudy

__all__ = [
    # geometry
    "Airfoil", "generate_naca4_coordinates", "load_airfoil_from_file",
    "LinearDistribution", "ConstantDistribution", "PolynomialDistribution",
    "BezierCurve", "SplineCurve", "get_rotor_bemt_stations",
    # core
    "Blade", "Rotor", "Duct", "DuctedFan", "OperatingConditions",
    # analysis
    "calculate_bemt_performance_axial",
    "calculate_ideal_performance_hover", "calculate_ideal_performance_axial",
    "run_bemt_analysis",
    # study
    "ParametricStudy",
    # meta
    "__version__",
]
