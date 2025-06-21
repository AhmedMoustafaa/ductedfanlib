"""
Defines the Duct class for representing the shroud of a ducted fan.
"""
from typing import Any, List, Tuple, Dict, Union, Optional, Callable
from ductedfanlib.geometry.curves import BezierCurve, SplineCurve  # Import curve types
import numpy as np
from scipy.interpolate import interp1d  # For get_diameter_at_axial_location

# Define a type hint for curve objects we expect
CurveObjectType = Union[BezierCurve, SplineCurve, Any]  # 'Any' for future custom curve types


class Duct:
    """
    Represents the duct (shroud) of a ducted fan.

    The duct's internal profile is defined by a curve object (e.g., BezierCurve, SplineCurve)
    which generates (axial_coord, radial_coord) points.
    The axial coordinates typically start from 0 at the duct inlet.

    Attributes:
        profile_curve (CurveObjectType):
            An instance of a curve class (e.g., BezierCurve, SplineCurve) from
            `geometry.curves` that defines the duct's internal profile.
            The curve should generate 2D points, interpreted as (axial_coordinate, radial_coordinate).
        reference_axial_position (float):
            The axial position (z-coordinate, in meters, within the duct's local coordinate system)
            that is considered a key reference, often where the rotor plane is intended to align.
            For example, if the duct profile starts at z=0 (inlet), and this is 0.1, then the
            rotor plane is 0.1m downstream from the duct inlet lip.
        # Explicit dimensional parameters are for convenience or high-level definition.
        # The true shape is dictated by profile_curve.
        length (Optional[float]): Overall axial length of the duct (meters).
                                  If None, may be derived from profile_curve.
        inlet_diameter (Optional[float]): Diameter at the duct inlet (meters).
        throat_diameter (Optional[float]): Diameter at the duct throat (meters).
        outlet_diameter (Optional[float]): Diameter at the duct outlet (meters).
    """

    def __init__(
            self,
            profile_curve: CurveObjectType,
            reference_axial_position: float = 0.0,
            length: Optional[float] = None,
            inlet_diameter: Optional[float] = None,
            # throat_diameter: Optional[float] = None, # Throat is complex to define parametrically without more info
            outlet_diameter: Optional[float] = None,
    ):
        """
        Initializes a Duct object.

        Args:
            profile_curve: A curve object (e.g., BezierCurve, SplineCurve) defining
                           the duct's internal profile as (axial, radial) coordinates.
            reference_axial_position (float): Axial location (m) of a reference point
                                              (e.g., rotor plane) within the duct's local coords.
            length (Optional[float]): Explicit overall axial length in meters.
            inlet_diameter (Optional[float]): Explicit inlet diameter in meters.
            outlet_diameter (Optional[float]): Explicit outlet diameter in meters.
        """
        if not hasattr(profile_curve, '__call__') or not hasattr(profile_curve, 'get_points'):
            raise TypeError(
                "profile_curve must be a callable object with a get_points method (e.g., BezierCurve, SplineCurve).")

        self.profile_curve: CurveObjectType = profile_curve
        self.reference_axial_position: float = float(reference_axial_position)

        # Store explicit dimensions if provided; they might be used for high-level checks
        # or to guide generation if profile_curve was to be generated internally.
        self.length: Optional[float] = float(length) if length is not None else None
        self.inlet_diameter: Optional[float] = float(inlet_diameter) if inlet_diameter is not None else None
        self.outlet_diameter: Optional[float] = float(outlet_diameter) if outlet_diameter is not None else None

        # Cached profile points and interpolator
        self._cached_profile_points: Optional[np.ndarray] = None
        self._radius_interpolator: Optional[Callable[[float], float]] = None

    def _generate_and_cache_profile(self, num_points: int = 100) -> None:
        """Internal method to generate and cache profile points and interpolator."""
        if self._cached_profile_points is None or self._cached_profile_points.shape[0] != num_points:
            points = self.profile_curve.get_points(num_points=num_points)
            if points.shape[1] != 2:
                raise ValueError("Profile curve must generate 2D points (axial_coord, radial_coord).")

            # Ensure points are sorted by axial coordinate for interpolation
            sorted_indices = np.argsort(points[:, 0])
            self._cached_profile_points = points[sorted_indices]

            # Create interpolator: axial_coord -> radial_coord
            # Use bounds_error=False and fill_value=np.nan to handle extrapolation gracefully
            # or catch it. For now, let's be strict and assume axial_location is within bounds.
            try:
                self._radius_interpolator = interp1d(
                    self._cached_profile_points[:, 0],  # Axial coordinates
                    self._cached_profile_points[:, 1],  # Radial coordinates
                    kind='linear',  # Could be 'cubic' for smoother, but linear is robust
                    bounds_error=True  # Raise error if outside interpolation range
                )
            except ValueError as e:
                # This can happen if axial coordinates are not strictly increasing
                # (should be handled by sorting, but good to catch)
                # Or if too few points for certain interpolation kinds.
                self._radius_interpolator = None  # Invalidate
                print(f"Warning: Could not create radius interpolator for duct '{self!r}'. {e}")

    def get_profile_points(self, num_points: int = 100) -> np.ndarray:
        """
        Generates or retrieves a cached list of (axial_coord, radial_coord) points
        representing the discretized internal profile of the duct.
        Points are sorted by axial coordinate.

        Args:
            num_points (int): Number of points to discretize the profile into.
                              If different from cached, regenerates.

        Returns:
            np.ndarray: Array of shape (num_points, 2) with (z, r) coordinates in meters.
        """
        self._generate_and_cache_profile(num_points=num_points)
        if self._cached_profile_points is None:
            # This should not happen if _generate_and_cache_profile worked
            raise RuntimeError("Failed to generate duct profile points.")
        return self._cached_profile_points.copy()  # Return a copy

    def get_radius_at_axial_location(self, axial_location: float, num_profile_points_for_interp: int = 100) -> float:
        """
        Calculates the internal radius of the duct at a given axial location.
        The axial_location is relative to the duct's own coordinate system (where z=0 is
        typically the start of the profile_curve).

        Args:
            axial_location (float): The z-coordinate (meters).
            num_profile_points_for_interp (int): Number of points to use for discretizing
                                                 the profile curve if an interpolator needs
                                                 to be (re)built.

        Returns:
            float: The internal radius (meters) at that location.

        Raises:
            ValueError: If axial_location is outside the defined duct profile range
                        or if interpolation fails.
        """
        if self._radius_interpolator is None or \
                (self._cached_profile_points is not None and self._cached_profile_points.shape[
                    0] != num_profile_points_for_interp):
            self._generate_and_cache_profile(num_points=num_profile_points_for_interp)

        if self._radius_interpolator is None:
            raise RuntimeError("Duct radius interpolator is not available.")

        try:
            radius = self._radius_interpolator(axial_location)
            return float(radius.item() if isinstance(radius, np.ndarray) else radius)
        except ValueError as e:  # Typically from bounds_error=True in interp1d
            min_z = self._cached_profile_points[:, 0].min()
            max_z = self._cached_profile_points[:, 0].max()
            raise ValueError(
                f"Axial location {axial_location:.4f}m is outside the duct profile's defined "
                f"axial range [{min_z:.4f}m, {max_z:.4f}m]. Error: {e}"
            ) from e

    def get_diameter_at_axial_location(self, axial_location: float, num_profile_points_for_interp: int = 100) -> float:
        """
        Calculates the internal diameter of the duct at a given axial location.

        Args:
            axial_location (float): The z-coordinate (meters).
            num_profile_points_for_interp (int): Number of points for profile discretization
                                                 if interpolator needs (re)building.
        Returns:
            float: The internal diameter (meters) at that location.
        """
        return 2.0 * self.get_radius_at_axial_location(axial_location, num_profile_points_for_interp)

    @property
    def derived_length(self) -> Optional[float]:
        """Attempts to derive duct length from its profile curve points."""
        if self._cached_profile_points is None:
            self._generate_and_cache_profile()  # Use default num_points
        if self._cached_profile_points is not None and self._cached_profile_points.shape[0] > 0:
            return self._cached_profile_points[:, 0].max() - self._cached_profile_points[:, 0].min()
        return None

    @property
    def derived_inlet_diameter(self) -> Optional[float]:
        """Attempts to derive inlet diameter from profile (assumes inlet at min axial coord)."""
        # This assumes the profile starts at the inlet.
        try:
            # Ensure profile is generated to get radius at min_z
            min_z = self.get_profile_points(num_points=2)[:, 0].min()  # Get min_z from at least 2 points
            return 2.0 * self.get_radius_at_axial_location(min_z)
        except (RuntimeError, ValueError):  # If profile/interpolator fails or min_z is out of bounds
            return None

    def __repr__(self) -> str:
        profile_type = self.profile_curve.__class__.__name__ \
            if hasattr(self.profile_curve, '__class__') else "CustomCurve"

        length_info = f"L={self.length if self.length is not None else self.derived_length:.3f}m" \
            if (self.length is not None or self.derived_length is not None) else "L=N/A"

        return (
            f"<Duct profile_type='{profile_type}', {length_info}, "
            f"ref_pos_z={self.reference_axial_position:.3f}m>"
        )