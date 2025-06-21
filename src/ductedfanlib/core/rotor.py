"""
Defines the Rotor class for representing a ducted fan rotor assembly.
"""
from typing import List, Dict, Any, Union, Optional # Added Optional
from .blade import Blade
import numpy as np

class Rotor:
    """
    Represents the rotor assembly of a ducted fan.

    Attributes:
        num_blades (int): Number of blades in the rotor.
        tip_radius (float): Radius of the blade tips (meters).
        hub_radius (float): Radius of the hub (meters).
        blade_definition (Blade):
            An instance of the Blade class defining the properties of a single blade.
            It's assumed all blades are identical.
        collective_pitch_deg (float):
            A global pitch offset (in degrees) applied to all blades, added to the
            blade's intrinsic twist distribution. Defaults to 0.0.
    """

    def __init__(
        self,
        num_blades: int,
        tip_radius: float,
        hub_radius: float,
        blade_definition: Blade,
        collective_pitch_deg: float = 0.0,
    ):
        """
        Initializes a Rotor object.

        Args:
            num_blades (int): Number of blades. Must be positive.
            tip_radius (float): Tip radius in meters. Must be > hub_radius.
            hub_radius (float): Hub radius in meters. Must be >= 0 and < tip_radius.
            blade_definition (Blade): An instance of the Blade class.
            collective_pitch_deg (float): Global pitch offset in degrees for all blades.
        """
        if not isinstance(num_blades, int) or num_blades <= 0:
            raise ValueError("Number of blades must be a positive integer.")
        if not (isinstance(tip_radius, (int, float)) and tip_radius > 0):
            raise ValueError("Tip radius must be a positive number.")
        if not (isinstance(hub_radius, (int, float)) and hub_radius >= 0):
            raise ValueError("Hub radius must be a non-negative number.")
        if hub_radius >= tip_radius:
            raise ValueError("Hub radius must be less than tip radius.")
        if not isinstance(blade_definition, Blade):
            raise TypeError("blade_definition must be an instance of the Blade class.")

        self.num_blades: int = num_blades
        self.tip_radius: float = tip_radius
        self.hub_radius: float = hub_radius
        self.blade_definition: Blade = blade_definition
        self.collective_pitch_deg: float = float(collective_pitch_deg)

        rotor_blade_span = self.tip_radius - self.hub_radius
        if self.blade_definition.span is None:
            self.blade_definition.span = rotor_blade_span
        elif not np.isclose(self.blade_definition.span, rotor_blade_span):
            print(
                f"Warning: Provided Blade span ({self.blade_definition.span:.4f}m) differs from "
                f"Rotor-derived span ({rotor_blade_span:.4f}m). "
                f"Adjusting Blade span to match Rotor dimensions."
            )
            self.blade_definition.span = rotor_blade_span


    @property
    def diameter(self) -> float:
        """Returns the rotor diameter (meters)."""
        return 2 * self.tip_radius

    @property
    def blade_span(self) -> float:
        """Returns the effective span of a single blade in this rotor (meters)."""
        return self.tip_radius - self.hub_radius

    @property
    def hub_tip_ratio(self) -> float:
        """Returns the hub-to-tip radius ratio."""
        if self.tip_radius == 0:
            return 0.0
        return self.hub_radius / self.tip_radius
    @property
    def get_disk_area(self):
        "returns the rotor disk area"
        return np.pi * (self.tip_radius**2)

    def get_radial_stations_properties(
        self,
        num_stations: Optional[int] = None,
        spacing_type: str = "linear",
        eta_values_override: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        """
        Generates properties (radius, chord, twist, airfoil) at discrete radial stations
        along a blade.

        Users can specify either `num_stations` (and optionally `spacing_type`) to generate
        eta values, or provide `eta_values_override` directly. If `eta_values_override`
        is provided, `num_stations` and `spacing_type` are ignored for eta generation.

        Args:
            num_stations (Optional[int]):
                Number of radial stations to evaluate along the span. Must be >= 2 if provided
                and `eta_values_override` is None. Defaults to None.
            spacing_type (str):
                Type of spacing for eta values if `num_stations` is used.
                "linear" for evenly spaced eta.
                "cosine" for cosine spacing (denser near hub and tip). Defaults to "linear".
            eta_values_override (Optional[np.ndarray]):
                A NumPy array of specific eta values (0.0 to 1.0) to use for stations.
                If provided, these values are used directly. Defaults to None.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary
            contains properties for a station.

        Raises:
            ValueError: If inputs are invalid (e.g., neither num_stations nor eta_values_override
                        is provided, or num_stations < 2 when used, or invalid spacing_type,
                        or eta values out of [0,1] range).
        """
        eta_values: np.ndarray

        if eta_values_override is not None:
            if not isinstance(eta_values_override, np.ndarray) or eta_values_override.ndim != 1:
                raise ValueError("eta_values_override must be a 1D NumPy array.")
            if np.any((eta_values_override < 0.0) | (eta_values_override > 1.0)):
                raise ValueError("All values in eta_values_override must be between 0.0 and 1.0.")
            eta_values = np.sort(np.unique(eta_values_override)) # Ensure sorted and unique
            if len(eta_values) < 1:
                raise ValueError("eta_values_override cannot be empty if provided.")
        elif num_stations is not None:
            if not isinstance(num_stations, int) or num_stations < 2:
                raise ValueError("If num_stations is provided, it must be an integer >= 2.")
            if spacing_type.lower() == "linear":
                eta_values = np.linspace(0.0, 1.0, num_stations)
            elif spacing_type.lower() == "cosine":
                angle = np.linspace(0, np.pi, num_stations)
                eta_values = (1.0 - np.cos(angle)) / 2.0
                eta_values[0], eta_values[-1] = 0.0, 1.0 # Ensure exact ends
            else:
                raise ValueError(f"Unknown spacing_type: '{spacing_type}'. Choose 'linear' or 'cosine'.")
        else:
            raise ValueError("Either num_stations or eta_values_override must be provided.")

        stations_data = []
        blade_span = self.blade_span

        for eta in eta_values:
            current_radius = self.hub_radius + eta * blade_span

            try:
                chord = self.blade_definition.get_chord_at_span_eta(eta)
                base_twist = self.blade_definition.get_twist_at_span_eta(eta)
                total_twist = base_twist + self.collective_pitch_deg
                airfoil_obj = self.blade_definition.get_airfoil_at_span_eta(eta)
            except ValueError as ve: # Catch eta out of bounds from Blade methods
                raise ValueError(
                    f"Error getting blade properties at eta={eta:.4f} (radius={current_radius:.4f}m): {ve}"
                ) from ve
            except Exception as e:
                raise RuntimeError(
                    f"Unexpected error getting blade properties at eta={eta:.4f} (radius={current_radius:.4f}m): {e}"
                ) from e

            station_info = {
                "eta": eta,
                "radius_m": current_radius,
                "chord_m": chord,
                "twist_deg": total_twist,
                "base_twist_deg": base_twist,
                "collective_pitch_deg": self.collective_pitch_deg,
                "airfoil_object": airfoil_obj,
                "airfoil_name": airfoil_obj.name if hasattr(airfoil_obj, 'name') else str(airfoil_obj)
            }
            stations_data.append(station_info)

        return stations_data



    def __repr__(self) -> str:
        return (
            f"<Rotor num_blades={self.num_blades}, tip_radius={self.tip_radius:.3f}m, "
            f"hub_radius={self.hub_radius:.3f}m, coll_pitch={self.collective_pitch_deg:.1f}deg, "
            f"blade={self.blade_definition!r}>"
        )