"""
DuctedFanLib: A Python Library for Ducted Fan Design, Analysis, and Optimization
================================================================================

This library provides tools for the parametric design of ducted fan components,
analysis using various fidelity methods, and optimization of designs.

Main components can be imported directly from this top-level package.
"""

# Define the package version
__version__ = "0.0.1"  # Start with a pre-release version

# Import key classes and functions from submodules to make them
# available at the top-level package import.


# From the 'geometry' module (specifically airfoils for now)
from .geometry.airfoils import Airfoil, generate_naca4_coordinates, load_airfoil_from_file
# As you add more to geometry (profiles, curves), you can expose them here too if desired.
# e.g., from .geometry.profiles import LinearProfile

# (Optional) Define what 'from ductedfanlib import *' would import.
# It's generally better for users to import specific names, but __all__ can be defined.
__all__ = [
    # Geometry components
    "Airfoil",
    "generate_naca4_coordinates",
    "load_airfoil_from_file",
    # Version
    "__version__",
]

# You could also set up a default logger here if your library will do extensive logging
# import logging
# logging.getLogger(__name__).addHandler(logging.NullHandler())
# print(f"DuctedFanLib version {__version__} loaded.") # Optional: for debug or verbose loading