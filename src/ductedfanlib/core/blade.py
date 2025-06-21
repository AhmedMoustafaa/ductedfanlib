"""
Defines the Blade class for representing a single ducted fan blade.
"""
from typing import Any, Callable, Union, List, Dict, TypeVar, OrderedDict
from collections import OrderedDict as PyOrderedDict # For sorted dictionary keys
import numpy as np
# Import specific types from the geometry module
from ductedfanlib.geometry.airfoils import Airfoil
from ductedfanlib.geometry.profiles import ConstantDistribution, LinearDistribution # Add more as needed

# Define a TypeVar for profile types to make type hints cleaner
ProfileCallable = Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]]
ProfileDefinition = Union[ConstantDistribution, LinearDistribution, ProfileCallable, Any]

# AirfoilDefinition can be a single Airfoil object, or a dictionary mapping float (eta) to Airfoil objects
AirfoilType = TypeVar('AirfoilType', bound=Airfoil)
AirfoilInputDefinition = Union[Airfoil, Dict[float, Airfoil]] # More specific for dict keys

import numpy as np


class Blade:
    """
    Represents a single fan blade with parametric definitions for its geometry.

    Attributes:
        airfoil_definition (AirfoilInputDefinition):
            The airfoil profile(s) used for this blade. Can be:
            - A single `Airfoil` object for the entire span.
            - A dictionary mapping sorted spanwise locations (eta, 0 to 1) to `Airfoil` objects.
              The airfoil at a key `eta_k` applies for `eta_k <= eta < eta_{k+1}`.
        chord_profile (ProfileDefinition):
            An object defining the chord length distribution (eta -> chord in meters).
        twist_profile (ProfileDefinition):
            An object defining the twist angle (in degrees) distribution (eta -> twist_angle).
        span (float, optional):
            Effective span of this blade section (meters).
    """

    def __init__(
        self,
        airfoil_definition: AirfoilInputDefinition,
        chord_profile: ProfileDefinition,
        twist_profile: ProfileDefinition,
        span: float = None,
    ):
        """
        Initializes a Blade object.
        """
        self.airfoil_definition: AirfoilInputDefinition
        if isinstance(airfoil_definition, dict):
            # Ensure dictionary keys (eta values) are floats and sorted for reliable lookup
            try:
                self.airfoil_definition = PyOrderedDict(
                    sorted({float(k): v for k, v in airfoil_definition.items()}.items())
                )
                if not self.airfoil_definition: # Empty dict
                    raise ValueError("Airfoil definition dictionary cannot be empty.")
                for k, v_af in self.airfoil_definition.items():
                    if not (0.0 <= k <= 1.0):
                        raise ValueError(f"Eta key {k} in airfoil_definition dict must be between 0 and 1.")
                    if not isinstance(v_af, Airfoil):
                        raise TypeError(f"Value for eta key {k} in airfoil_definition dict must be an Airfoil object.")
            except Exception as e:
                raise ValueError(f"Invalid airfoil_definition dictionary: {e}") from e
        elif isinstance(airfoil_definition, Airfoil):
            self.airfoil_definition = airfoil_definition
        else:
            raise TypeError(
                "airfoil_definition must be an Airfoil object or a Dict[float, Airfoil]."
            )

        self.chord_profile: ProfileDefinition = chord_profile
        self.twist_profile: ProfileDefinition = twist_profile
        self.span: Union[float, None] = span

        if not callable(self.chord_profile):
            raise TypeError("chord_profile must be a callable object (e.g., a distribution instance).")
        if not callable(self.twist_profile):
            raise TypeError("twist_profile must be a callable object (e.g., a distribution instance).")


    def get_chord_at_span_eta(self, eta: float) -> float:
        """
        Calculates the chord length at a given normalized spanwise location (eta).

        Args:
            eta (float): Normalized spanwise location (must be between 0 and 1, inclusive).

        Returns:
            float: Chord length in meters.

        Raises:
            ValueError: If eta is outside the [0, 1] range.
        """
        if not (0.0 <= eta <= 1.0):
            raise ValueError(f"eta ({eta}) must be between 0.0 and 1.0 (inclusive).")

        value = self.chord_profile(eta)
        if isinstance(value, np.ndarray):
            return float(value.item())
        return float(value)


    def get_twist_at_span_eta(self, eta: float) -> float:
        """
        Calculates the twist angle (in degrees) at a given normalized spanwise location (eta).

        Args:
            eta (float): Normalized spanwise location (must be between 0 and 1, inclusive).

        Returns:
            float: Twist angle in degrees.

        Raises:
            ValueError: If eta is outside the [0, 1] range.
        """
        if not (0.0 <= eta <= 1.0):
            raise ValueError(f"eta ({eta}) must be between 0.0 and 1.0 (inclusive).")

        value = self.twist_profile(eta)
        if isinstance(value, np.ndarray):
            return float(value.item())
        return float(value)

    def get_airfoil_at_span_eta(self, eta: float) -> Airfoil:
        """
        Gets the Airfoil object at a given normalized spanwise location (eta).
        If airfoil_definition is a dictionary, the airfoil at key `eta_k` applies for
        `eta_k <= eta < eta_{k+1}` (or for `eta >= largest_key` if eta is beyond the last key).

        Args:
            eta (float): Normalized spanwise location (must be between 0 and 1, inclusive).

        Returns:
            Airfoil: The Airfoil object active at that spanwise location.

        Raises:
            ValueError: If eta is outside the [0, 1] range.
            TypeError: If airfoil_definition is not a supported type.
            LookupError: If airfoil_definition is a dict and no suitable airfoil can be found for eta.
        """
        if not (0.0 <= eta <= 1.0):
            raise ValueError(f"eta ({eta}) must be between 0.0 and 1.0 (inclusive).")

        if isinstance(self.airfoil_definition, Airfoil):
            return self.airfoil_definition
        elif isinstance(self.airfoil_definition, PyOrderedDict): # It's an OrderedDict due to __init__
            # Find the airfoil section that applies to the given eta
            # The airfoil at key eta_k applies for eta_k <= query_eta < eta_{k+1}
            # Or if query_eta >= largest_key, use the airfoil at largest_key.

            selected_airfoil = None
            # Iterate through the sorted keys (eta station definitions)
            # The PyOrderedDict ensures keys are sorted.
            keys = list(self.airfoil_definition.keys())

            if not keys: # Should have been caught by __init__ but as a safeguard
                raise LookupError("Airfoil definition dictionary is empty.")

            # If eta is less than the first defined station, use the first station's airfoil
            if eta < keys[0]:
                # This case should ideally not happen if users define from eta=0.0
                # Or, one could argue it should raise an error if eta < min_key.
                # For now, let's be lenient and use the first one.
                # A stricter approach: raise ValueError(f"eta {eta} is below the first defined airfoil station {keys[0]}")
                selected_airfoil = self.airfoil_definition[keys[0]]
            else:
                for i in range(len(keys)):
                    current_key = keys[i]
                    if current_key <= eta:
                        selected_airfoil = self.airfoil_definition[current_key]
                        # If this is the last key, or if eta is less than the next key, this is our section
                        if (i + 1) == len(keys) or eta < keys[i+1]:
                            break
                    else: # Should not be reached if eta >= keys[0] and keys are sorted
                        break

            if selected_airfoil is None:
                # This should ideally not be reached if logic is correct and dict is not empty
                raise LookupError(f"Could not determine airfoil for eta={eta}. Available stations: {keys}")
            return selected_airfoil
        else:
            # This case should be caught by __init__'s type checking for airfoil_definition
            raise TypeError(
                f"Unsupported airfoil_definition type: {type(self.airfoil_definition)}. "
            )

    def __repr__(self) -> str:
        chord_profile_repr = self.chord_profile.__class__.__name__ \
            if hasattr(self.chord_profile, '__class__') else str(self.chord_profile)
        twist_profile_repr = self.twist_profile.__class__.__name__ \
            if hasattr(self.twist_profile, '__class__') else str(self.twist_profile)

        airfoil_repr = ""
        if isinstance(self.airfoil_definition, Airfoil):
            airfoil_repr = self.airfoil_definition.name
        elif isinstance(self.airfoil_definition, dict): # PyOrderedDict is a dict
            airfoil_repr = f"Dict[{len(self.airfoil_definition)} sections]"
        else:
            airfoil_repr = str(self.airfoil_definition)

        return (
            f"<Blade airfoil='{airfoil_repr}', "
            f"chord_profile='{chord_profile_repr}', "
            f"twist_profile='{twist_profile_repr}', "
            f"span={self.span if self.span is not None else 'N/A'}>"
        )