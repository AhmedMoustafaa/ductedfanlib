"""
Defines classes for generating parametric curves like Bezier and Splines.
These are useful for defining duct profiles, blade camber lines, etc.
"""
from typing import Union, List, Optional
import numpy as np
from scipy.special import comb  # For binomial coefficients C(n, k) for Bezier


class BezierCurve:
    """
    Represents a Bezier curve defined by a set of control points.
    The curve can be of any degree (linear, quadratic, cubic, etc.) depending
    on the number of control points (degree = num_control_points - 1).
    The points can be 2D or 3D.
    """

    def __init__(self, control_points: Union[List[List[float]], np.ndarray]):
        """
        Initializes a BezierCurve object.

        Args:
            control_points (Union[List[List[float]], np.ndarray]):
                A list of control points or a NumPy array of shape (N, D),
                where N is the number of control points and D is the dimension (2 for 2D, 3 for 3D).
                Example for a 2D cubic Bezier: [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]
        """
        self.control_points = np.asarray(control_points, dtype=float)
        if self.control_points.ndim != 2:
            raise ValueError("Control points must be a 2D array of shape (N, D).")

        self.num_control_points = self.control_points.shape[0]
        self.degree = self.num_control_points - 1
        self.dimension = self.control_points.shape[1]

        if self.degree < 1:  # Linear Bezier needs at least 2 points
            raise ValueError("A Bezier curve must have at least 2 control points (degree >= 1).")

    def _bernstein_polynomial(self, i: int, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculates the i-th Bernstein basis polynomial of degree n at parameter t.
        B(i,n)(t) = C(n,i) * t^i * (1-t)^(n-i)
        where n is the degree of the Bezier curve.
        """
        n = self.degree
        # scipy.special.comb(N,k) computes "N choose k"
        binomial_coeff = comb(n, i, exact=False)  # exact=False returns float
        return binomial_coeff * (t ** i) * ((1 - t) ** (n - i))

    def __call__(self, t: Union[float, np.ndarray]) -> np.ndarray:
        """
        Evaluates the Bezier curve at parameter value(s) t.
        The parameter t ranges from 0 to 1.

        Args:
            t (Union[float, np.ndarray]):
                A scalar or NumPy array of parameter values (between 0 and 1).

        Returns:
            np.ndarray:
                The point(s) on the curve. If t is a scalar, returns a 1D array of shape (D,).
                If t is an array of M values, returns a 2D array of shape (M, D).
        """
        t_arr = np.asarray(t, dtype=float)
        # Ensure t is within or close to [0, 1] range for typical use, though formula works outside.
        # if np.any(t_arr < 0) or np.any(t_arr > 1):
        # print("Warning: Bezier parameter t is outside the typical [0, 1] range.")

        if t_arr.ndim == 0:  # Scalar t
            point_on_curve = np.zeros(self.dimension)
            for i in range(self.num_control_points):
                point_on_curve += self._bernstein_polynomial(i, t_arr) * self.control_points[i]
            return point_on_curve
        else:  # Array of t values
            points_on_curve = np.zeros((len(t_arr), self.dimension))
            for i in range(self.num_control_points):
                bernstein_values = self._bernstein_polynomial(i, t_arr)
                # Expand dims for broadcasting: bernstein_values[:, np.newaxis] * self.control_points[i]
                points_on_curve += bernstein_values[:, np.newaxis] * self.control_points[i]
            return points_on_curve

    def get_points(self, num_points: int = 100) -> np.ndarray:
        """
        Generates a sequence of `num_points` along the Bezier curve.
        Points are linearly spaced in the parameter `t` from 0 to 1.

        Args:
            num_points (int): The number of points to generate along the curve.

        Returns:
            np.ndarray: An array of shape (num_points, D) representing the points.
        """
        if num_points < 2:
            raise ValueError("num_points must be at least 2 to define a curve segment.")
        t_values = np.linspace(0.0, 1.0, num_points)
        return self(t_values)

    def __repr__(self) -> str:
        return (f"<BezierCurve degree={self.degree}, "
                f"num_control_points={self.num_control_points}, "
                f"dimension={self.dimension}>")


class SplineCurve:
    """
    Represents a spline curve (B-spline) defined by a set of data points.
    Uses scipy.interpolate.splprep for fitting and scipy.interpolate.splev for evaluation.
    The curve passes through or near the provided data points.
    """

    def __init__(self, points: Union[List[List[float]], np.ndarray],
                 degree: int = 3,
                 smooth_factor: Optional[float] = 0,
                 periodic: bool = False):
        """
        Initializes a SplineCurve object.

        Args:
            points (Union[List[List[float]], np.ndarray]):
                Array of shape (N, D) defining the data points the spline should pass through or near.
                N is the number of points, D is the dimension (2 for 2D, 3 for 3D).
            degree (int): Degree of the spline (e.g., 1 for linear, 2 for quadratic, 3 for cubic).
                          Must be 1 <= degree <= 5.
            smooth_factor (Optional[float]):
                Smoothing factor for splprep (s parameter).
                s=0: spline passes through all data points (interpolation).
                s>0: spline is smoothed (approximation).
                If None, scipy attempts to choose a value (not recommended for predictable results).
                Default is 0 for interpolation.
            periodic (bool):
                If True, creates a periodic spline. The first and last points should be the same
                if you want a closed curve that is C^(k-1) continuous at the connection.
                `splprep` handles this with the `per` flag.
        """
        self.points = np.asarray(points, dtype=float)
        if self.points.ndim != 2:
            raise ValueError("Points must be a 2D array of shape (N, D).")

        self.num_data_points = self.points.shape[0]
        self.dimension = self.points.shape[1]
        self.degree = degree

        if not (1 <= self.degree <= 5):
            raise ValueError(f"Spline degree k must be 1 <= k <= 5, got {self.degree}.")
        if self.num_data_points <= self.degree:
            raise ValueError(
                f"Number of data points ({self.num_data_points}) must be greater than spline degree ({self.degree})."
            )

        # splprep needs coordinates as a list of 1D arrays (e.g., [x_coords, y_coords])
        coords_for_splprep = [self.points[:, d] for d in range(self.dimension)]

        # tck is a tuple (t,c,k) containing the vector of knots, B-spline coefficients, and degree.
        # u is the array of parameter values corresponding to the input data points.
        try:
            self.tck, self.u = splprep(
                coords_for_splprep,
                s=smooth_factor,
                k=self.degree,
                per=int(periodic),  # splprep expects 0 or 1 for per
                quiet=2  # Suppress Fortran messages
            )
        except Exception as e:
            # splprep can raise various errors (e.g., TypeError if too few points,
            # _fitpack. διαδικασίαข้อผิดพลาด if smoothing factor is too large for noisy data)
            raise RuntimeError(f"SciPy splprep failed: {e}") from e

    def __call__(self, t_norm: Union[float, np.ndarray]) -> np.ndarray:
        """
        Evaluates the spline at normalized parameter value(s) t_norm (0 to 1).
        This t_norm is mapped to the internal parameter range of the spline
        (typically 0 to 1 for open splines, or the range of self.u).

        Args:
            t_norm (Union[float, np.ndarray]): Normalized parameter value(s) from 0 to 1.

        Returns:
            np.ndarray: Point(s) on the spline. Shape (D,) if t_norm is scalar,
                        or (M, D) if t_norm is an array of M values.
        """
        t_norm_arr = np.asarray(t_norm, dtype=float)

        # The parameter 'u' returned by splprep usually ranges from 0 to 1 for open curves.
        # For periodic curves, it also often spans 0 to 1, representing one full period.
        # We evaluate splev with these 'u' values.
        # If t_norm is intended to represent this 0-1 range directly:
        u_eval = t_norm_arr
        # An alternative mapping if self.u had a different range:
        # u_eval = self.u.min() + t_norm_arr * (self.u.max() - self.u.min())

        # splev returns a list of arrays (one for each dimension)
        evaluated_coords_list = splev(u_eval, self.tck)

        # Stack them into (D, M) and transpose to (M, D) or (D,)
        if t_norm_arr.ndim == 0:  # Scalar input
            return np.array([coords[0] if hasattr(coords, '__getitem__') and len(coords) > 0 else coords for coords in
                             evaluated_coords_list])
        else:  # Array input
            return np.vstack(evaluated_coords_list).T

    def get_points(self, num_points: int = 100) -> np.ndarray:
        """
        Generates a sequence of `num_points` along the spline.
        Points are linearly spaced in the normalized parameter `t_norm` from 0 to 1.

        Args:
            num_points (int): The number of points to generate along the curve.

        Returns:
            np.ndarray: An array of shape (num_points, D) representing the points.
        """
        if num_points < 2:
            raise ValueError("num_points must be at least 2.")
        t_norm_values = np.linspace(0.0, 1.0, num_points)
        return self(t_norm_values)

    def __repr__(self) -> str:
        return (f"<SplineCurve degree={self.degree}, "
                f"num_data_points={self.num_data_points}, "
                f"dimension={self.dimension}>")
