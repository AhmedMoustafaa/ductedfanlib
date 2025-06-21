"""
Defines the DuctedFan class, representing the complete assembly.
"""
from .rotor import Rotor
from .duct import Duct
from typing import Dict, Optional
import numpy as np


class DuctedFan:
    """
    Represents the complete ducted fan assembly, combining a rotor and a duct.

    Attributes:
        rotor (Rotor): The rotor component.
        duct (Duct): The duct component.
        tip_clearance (float):
            The radial gap between the blade tips and the duct inner surface (meters).
        rotor_axial_position (float):
            The axial position of the rotor's reference plane (e.g., propeller disk plane)
            within the duct's local coordinate system (meters). The duct's profile typically
            starts at its own axial coordinate z=0.
    """

    def __init__(
            self,
            rotor: Rotor,
            duct: Duct,
            tip_clearance: float,
            rotor_axial_position: float = 0.0,
    ):
        """
        Initializes a DuctedFan assembly.

        Args:
            rotor (Rotor): An instance of the Rotor class.
            duct (Duct): An instance of the Duct class.
            tip_clearance (float): Radial tip clearance in meters. Must be non-negative.
            rotor_axial_position (float): Axial position (meters) of the rotor reference plane
                                          within the duct's coordinate system.
        """
        if not isinstance(rotor, Rotor):
            raise TypeError("rotor must be an instance of the Rotor class.")
        if not isinstance(duct, Duct):
            raise TypeError("duct must be an instance of the Duct class.")
        if not (isinstance(tip_clearance, (int, float)) and tip_clearance >= 0):
            raise ValueError("Tip clearance must be a non-negative number.")
        if not isinstance(rotor_axial_position, (int, float)):
            raise ValueError("rotor_axial_position must be a number.")

        self.rotor: Rotor = rotor
        self.duct: Duct = duct
        self.tip_clearance: float = tip_clearance
        self.rotor_axial_position: float = float(rotor_axial_position)

        self._validate_geometry()

    def _validate_geometry(self, num_profile_points_for_check: int = 100) -> None:
        """
        Performs basic checks for geometric consistency.
        Ensures the rotor (considering tip clearance) fits within the duct
        at its designated axial position.

        Args:
            num_profile_points_for_check (int): Number of points to use for discretizing
                                                the duct profile if its interpolator needs
                                                to be (re)built for the check.
        Raises:
            ValueError: If the rotor does not fit within the duct.
        """
        try:
            duct_diameter_at_rotor_plane = self.duct.get_diameter_at_axial_location(
                self.rotor_axial_position,
                num_profile_points_for_interp=num_profile_points_for_check
            )
        except ValueError as e:  # Likely if rotor_axial_position is outside duct's defined z-range
            raise ValueError(
                f"Cannot validate geometry: Rotor axial position {self.rotor_axial_position:.4f}m "
                f"may be outside the duct's defined axial range. Duct error: {e}"
            ) from e
        except RuntimeError as e:  # If duct interpolator failed
            raise RuntimeError(
                f"Cannot validate geometry due to duct processing error: {e}"
            ) from e

        required_duct_diameter_for_rotor = self.rotor.diameter + 2 * self.tip_clearance

        # Add a small tolerance for floating point comparisons
        tolerance = 1e-6
        if required_duct_diameter_for_rotor > duct_diameter_at_rotor_plane + tolerance:
            raise ValueError(
                f"Rotor does not fit in duct at axial position {self.rotor_axial_position:.4f}m. "
                f"Rotor effective diameter (incl. tip clearance): {required_duct_diameter_for_rotor:.4f}m, "
                f"Duct inner diameter: {duct_diameter_at_rotor_plane:.4f}m."
            )
        # Optionally, print a success message or just pass silently
        # print("Geometry validation successful: Rotor fits within duct.")

    def get_overall_dimensions(self, num_duct_profile_points: int = 100) -> Dict[str, Optional[float]]:
        """
        Returns some overall characteristic dimensions of the ducted fan assembly.

        Args:
            num_duct_profile_points (int): Number of points to use for discretizing
                                           the duct profile to determine its dimensions.
        Returns:
            Dict[str, Optional[float]]: A dictionary with dimensions like
                                       'overall_length', 'max_outer_diameter'.
                                       Values can be None if not determinable.
        """
        dims = {
            "overall_length": None,
            "max_outer_diameter": None,  # Assuming duct profile defines inner diameter for now
            "rotor_diameter": self.rotor.diameter
        }

        try:
            # Duct length
            # Use explicit length if provided, otherwise try to derive from profile
            if self.duct.length is not None:
                dims["overall_length"] = self.duct.length
            else:
                derived_length = self.duct.derived_length  # Property now uses cached points or generates
                if derived_length is not None:
                    dims["overall_length"] = derived_length

            # Max duct inner diameter
            profile_points = self.duct.get_profile_points(num_points=num_duct_profile_points)
            if profile_points is not None and profile_points.shape[0] > 0:
                # profile_points are (z, r), so max radius is max of 2nd column
                max_radius = np.max(profile_points[:, 1])
                dims["max_outer_diameter"] = 2 * max_radius  # This is inner diameter, outer depends on thickness

            # For a true "outer" diameter, we'd need duct thickness information.
            # For now, "max_outer_diameter" will represent the max inner diameter of the duct.

        except Exception as e:
            print(f"Warning: Could not determine some overall dimensions due to an error: {e}")
            # Dimensions that couldn't be calculated will remain None

        return dims

    def get_diffusion_ratio(self):
        AR = self.rotor.get_disk_area()
        A4 = (np.pi * self.duct.outlet_diameter**2) / 4
        return A4 / AR

    def __repr__(self) -> str:
        return (
            f"<DuctedFan rotor={self.rotor!r}, \n"  # Rotor repr already ends with >
            f" duct={self.duct!r}, \n"  # Duct repr already ends with >
            f" tip_clearance={self.tip_clearance:.4f}m, rotor_axial_pos={self.rotor_axial_position:.3f}m>"
        )