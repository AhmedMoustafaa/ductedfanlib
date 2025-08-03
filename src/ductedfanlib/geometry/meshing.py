"""
Meshing and Discretization Utilities for DuctedFanLib.

This module provides functions to convert parametric design objects from the 'core'
module into discretized geometric representations suitable for various analysis methods.
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from ductedfanlib.core.rotor import Rotor
from ductedfanlib.core.duct import Duct


def get_rotor_bemt_stations(
        rotor: Rotor,
        num_stations: Optional[int] = None,
        spacing_type: str = "cosine",
        eta_values_override: Optional[np.ndarray] = None
) -> List[Dict[str, Any]]:
    """
    Retrieves blade element properties at discrete radial stations for BEMT analysis.

    This function is largely a wrapper around the Rotor.get_radial_stations_properties()
    method, promoting the 'meshing' module as a central place for analysis-specific
    geometry generation.

    Args:
        rotor (Rotor): The Rotor object to discretize.
        num_stations (Optional[int]):
            Number of radial stations. Used if eta_values_override is None. Defaults to 21 if both are None.
        spacing_type (str):
            Type of spacing ("linear" or "cosine") if num_stations is used. Defaults to "cosine".
        eta_values_override (Optional[np.ndarray]):
            Specific eta values to use. Overrides num_stations and spacing_type.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each containing properties for a station.
                              See Rotor.get_radial_stations_properties() for dict structure.
    """
    if not isinstance(rotor, Rotor):
        raise TypeError("Input 'rotor' must be an instance of the Rotor class.")

    if num_stations is None and eta_values_override is None:
        num_stations = 21

    return rotor.get_radial_stations_properties(
        num_stations=num_stations,
        spacing_type=spacing_type,
        eta_values_override=eta_values_override
    )


def generate_duct_axisymmetric_panels(
        duct: Duct,
        num_profile_points: int = 51,
) -> np.ndarray:
    """
    Generates a series of linear panels representing the axisymmetric duct profile.

    Each panel is defined by its start and end points (axial_coord, radial_coord).
    This is a basic discretization suitable for simple 2D panel methods or visualization.

    Args:
        duct (Duct): The Duct object whose profile is to be panelized.
        num_profile_points (int): Number of points to use from the duct's profile curve.
                                  This will result in num_profile_points - 1 panels.

    Returns:
        np.ndarray: An array of panels. Each panel is represented as ((z1, r1), (z2, r2)).
                    Shape: (num_profile_points - 1, 2, 2).
                    The first dimension is the panel index.
                    The second dimension indicates start (0) or end (1) point of the panel.
                    The third dimension contains the (z, r) coordinates.

    Raises:
        TypeError: If duct is not a Duct instance.
        ValueError: If num_profile_points is too small.
    """
    if not isinstance(duct, Duct):
        raise TypeError("Input 'duct' must be an instance of the Duct class.")
    if num_profile_points < 2:
        raise ValueError("num_profile_points must be at least 2 to form any panels.")


    profile_points = duct.get_profile_points(num_points=num_profile_points)

    if profile_points.shape[0] < 2:
        raise ValueError("Not enough points from duct profile to generate panels.")

    num_panels = profile_points.shape[0] - 1
    panels = np.zeros((num_panels, 2, 2))

    for i in range(num_panels):
        panels[i, 0, :] = profile_points[i, :]
        panels[i, 1, :] = profile_points[i + 1, :]

    return panels

# TODO (Future - Advanced Meshing):
# def generate_3d_blade_surface_mesh(blade: Blade, num_spanwise_sections: int, num_chordwise_points: int) -> Any:
#     """ Generates a 3D surface mesh (e.g., quads or triangles) for a blade. """
#     # This would involve:
#     # 1. Getting airfoil coordinates at each spanwise section.
#     # 2. Scaling by chord, translating by pitch axis, rotating by twist.
#     # 3. Positioning along span, considering sweep and dihedral.
#     # 4. "Lofting" or "skinning" between sections to create surface elements.
#     raise NotImplementedError("3D blade surface meshing not yet implemented.")

# def generate_duct_panel_mesh_3d(duct: Duct, num_axial_panels: int, num_circumferential_panels: int) -> Any:
#     """ Generates a 3D panel mesh for an axisymmetric duct. """
#     # This would involve:
#     # 1. Getting the 2D profile (panels from generate_duct_axisymmetric_panels or similar).
#     # 2. Revolving these 2D panels around the axis of symmetry.
#     raise NotImplementedError("3D duct panel meshing not yet implemented.")