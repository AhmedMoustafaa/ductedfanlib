"""
Defines classes and functions for creating parametric distribution profiles.

These profiles describe how a scalar value (like chord, twist, thickness)
varies along a normalized dimension 'eta' (typically from 0 to 1).
"""
from typing import Callable, Union, Tuple, List
import numpy as np

class ConstantDistribution:
    """
    Represents a constant value distribution.
    """
    def __init__(self, value: float):
        """
        Args:
            value (float): The constant value for this distribution.
        """
        self.value = value

    def __call__(self, eta: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Returns the constant value, regardless of eta.

        Args:
            eta (Union[float, np.ndarray]): Normalized position(s) (0 to 1). Unused.

        Returns:
            Union[float, np.ndarray]: The constant value. If eta is a numpy array,
                                      an array of the same shape filled with the value is returned.
        """
        if isinstance(eta, np.ndarray):
            return np.full_like(eta, self.value)
        return self.value

    def __repr__(self) -> str:
        return f"<ConstantDistribution value={self.value}>"


class LinearDistribution:
    """
    Represents a linear distribution between a start and end value.
    Value(eta) = start_value * (1 - eta) + end_value * eta
    """
    def __init__(self, start_value: float, end_value: float):
        """
        Args:
            start_value (float): The value at eta = 0.
            end_value (float): The value at eta = 1.
        """
        self.start_value = start_value
        self.end_value = end_value

    def __call__(self, eta: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculates the value at the given normalized position(s) eta.

        Args:
            eta (Union[float, np.ndarray]): Normalized position(s) (0 to 1).

        Returns:
            Union[float, np.ndarray]: The interpolated value(s) at eta.
        """
        if isinstance(eta, (float, int)):
            if not (0.0 <= eta <= 1.0):
                # print(f"Warning: eta ({eta}) is outside the typical [0, 1] range.")
                pass # Allow extrapolation for now

        return self.start_value * (1.0 - eta) + self.end_value * eta

    def __repr__(self) -> str:
        return f"<LinearDistribution start={self.start_value}, end={self.end_value}>"


class PolynomialDistribution:
    """
    Represents a polynomial distribution defined by coefficients.
    Value(eta) = c0 + c1*eta + c2*eta^2 + ...
    """
    def __init__(self, coefficients: List[float]):
        # coefficients[0] is c0, coefficients[1] is c1, etc.
        self.coefficients = np.array(coefficients)
#
    def __call__(self, eta: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        if isinstance(eta, (float, int)):
            val = 0.0
            for i, c in enumerate(self.coefficients):
                val += c * (eta ** i)
            return val
        # For numpy array eta:
        elif isinstance(eta, (List, np.ndarray)):
            eta_arr = np.asarray(eta)
            val_arr = np.zeros_like(eta_arr)
            for i, c in enumerate(self.coefficients):  val_arr += c * (eta_arr ** i)
            return val_arr
    def __repr__(self) -> str:
        return f"<PolynomialDistribution coefficients={self.coefficients.tolist()}>"


class CustomDistribution:
    def __init__(self,Xs: Union[List, np.ndarray], Ys: Union[List, np.ndarray], func:Callable=None):
        self.func = func
        self.Xs = Xs
        self.Ys = Ys
    def __call__(self,eta:Union[float,int,List,np.ndarray]) -> Union[float,List,np.ndarray]:
        """
        interpolates to find the Y value at X=eta
        :param eta: Single or List of Values to calculate Y(eta) at
        :return: Single or List of Y(eta)
        """
        if self.func: #if th e distribution is defined as a function
            try:
                return self.func(eta) #try to just pass the list of vales to the function
            except:
                if isinstance(eta, (int, float)):
                    vals = self.func(eta)
                elif isinstance(eta, (List, np.ndarray)):
                    vals = np.zeros(np.shape(eta))
                    for i,eta_i in enumerate(eta):
                        vals[i] = self.func(eta_i)
                return vals

        else:
            if not self.Xs or not self.Ys:
                raise  AttributeError("Please Provide list of X & Y values or a callable function")
            else:
                pass # TODO
                

def discretize_distribution(
    distribution: Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]],
    num_points: int,
    eta_start: float = 0.0,
    eta_end: float = 1.0,
    spacing: str = "linear" # "linear" or "cosine"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluates a given distribution callable at a specified number of points
    between eta_start and eta_end using a specified spacing.

    Args:
        distribution (Callable): A callable object (like instances of the classes above)
                                 that accepts eta (float or np.ndarray) and returns a value.
        num_points (int): The number of points to evaluate.
        eta_start (float): The starting eta value for discretization.
        eta_end (float): The ending eta value for discretization.
        spacing (str): Type of point spacing for eta values.
                       "linear" or "cosine".

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - eta_values (np.ndarray): The array of eta values at which the distribution was evaluated.
            - evaluated_values (np.ndarray): The array of corresponding evaluated values.

    Raises:
        ValueError: If num_points < 1 or unknown spacing type.
    """
    if num_points < 1:
        raise ValueError("num_points must be at least 1.")

    if spacing == "linear":
        eta_values = np.linspace(eta_start, eta_end, num_points)
    elif spacing == "cosine":
        angle = np.linspace(0, np.pi, num_points)
        normalized_cosine = (1 - np.cos(angle)) / 2.0 # 0 to 1
        eta_values = eta_start + (eta_end - eta_start) * normalized_cosine
    else:
        raise ValueError(f"Unknown spacing type: '{spacing}'. Choose 'linear' or 'cosine'.")

    evaluated_values = distribution(eta_values)
    return eta_values, evaluated_values