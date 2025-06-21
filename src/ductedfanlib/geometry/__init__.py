"""
Geometry Module for DuctedFanLib
--------------------------------

This package contains tools for generating, representing, and manipulating
geometric entities relevant to ducted fan design. This includes:
- Airfoil definitions and generators.
- Profile functions for distributions (e.g., chord, twist).
- Curve generators for duct shapes.
- Meshing and discretization utilities for analysis.
"""

from .airfoils import Airfoil, generate_naca4_coordinates, load_airfoil_from_file
from .profiles import ConstantDistribution, LinearDistribution,PolynomialDistribution, discretize_distribution
from .curves import BezierCurve, SplineCurve
from .meshing import get_rotor_bemt_stations, generate_duct_axisymmetric_panels # Added meshing functions

__all__ = [
    "Airfoil",
    "generate_naca4_coordinates",
    "load_airfoil_from_file",
    "ConstantDistribution",
    "LinearDistribution",
    "PolynomialDistribution",
    "discretize_distribution",
    "BezierCurve",
    "SplineCurve",
    "get_rotor_bemt_stations", # Added
    "generate_duct_axisymmetric_panels", # Added
]