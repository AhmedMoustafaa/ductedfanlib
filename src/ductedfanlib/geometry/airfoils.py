"""
Defines the Airfoil class and related utility functions for
airfoil geometry representation, generation, and aerodynamic coefficient lookup.
"""
from typing import Optional, Tuple, List, Any, Dict, Union
import numpy as np
from scipy.interpolate import interp1d, PchipInterpolator
from pathlib import Path

try:
    import neuralfoil as nf
    NEURALFOIL_AVAILABLE = True
except ImportError:
    NEURALFOIL_AVAILABLE = False
try:
    from aerosandbox.aerodynamics.aero_2D.xfoil import XFoil as ASBXFoil
    from aerosandbox.geometry import Airfoil as ASBAirfoil
    AEROSANDBOX_XFOIL_AVAILABLE = True
except ImportError:
    AEROSANDBOX_XFOIL_AVAILABLE = False

def _cosine_spacing(x_start: float, x_end: float, num_points: int) -> np.ndarray:
    if num_points == 1: return np.array([(x_start + x_end) / 2])
    if num_points < 1: return np.array([])
    angle = np.linspace(0, np.pi, num_points)
    return x_start + (x_end - x_start) * (1 - np.cos(angle)) / 2.0

def _linear_spacing(x_start: float, x_end: float, num_points: int) -> np.ndarray:
    if num_points < 1: return np.array([])
    return np.linspace(x_start, x_end, num_points)

DEFAULT_CLA_2PI = 2 * np.pi

class Airfoil:
    """
    Represents an airfoil profile, its geometry, and its aerodynamic coefficients.
    """
    def __init__(self, name: str,
                 coordinates: Optional[np.ndarray] = None,
                 upper_coords: Optional[np.ndarray] = None,
                 lower_coords: Optional[np.ndarray] = None,
                 thickness_to_chord_ratio: Optional[float] = None,
                 cd_max_poststall: float = 1.5,
                 default_analysis_method: str = "neuralfoil"):

        self.name: str = name
        self._upper_surface: Optional[np.ndarray] = None
        self._lower_surface: Optional[np.ndarray] = None
        self._coordinates: Optional[np.ndarray] = None

        if upper_coords is not None and lower_coords is not None:
            self._validate_and_set_surface_coords(upper_coords, lower_coords)
        elif coordinates is not None:
            self._parse_combined_coordinates(coordinates)
        else:
            raise ValueError("Must provide either 'coordinates' or both 'upper_coords' and 'lower_coords'.")

        self._normalize_and_align_surfaces()

        self.polars: Dict[float, Dict[str, Any]] = {}
        self.stall_angle_deg: Optional[float] = None
        self.cl_stall: Optional[float] = None
        self.cd_stall: Optional[float] = None
        self.cd_max_poststall: float = float(cd_max_poststall)

        self._thickness_to_chord_ratio: Optional[float] = thickness_to_chord_ratio
        self._lift_curve_slope_cache: Dict[Tuple, float] = {}

        # ... (rest of __init__ as before) ...
        valid_methods = ["polar", "neuralfoil", "xfoil"]
        if default_analysis_method not in valid_methods:
            raise ValueError(f"default_analysis_method must be one of {valid_methods}.")
        self.default_analysis_method = default_analysis_method


    # Geometry Methods
    @staticmethod
    def _sort_surface(coords: np.ndarray) -> np.ndarray:
        if coords.shape[0] > 1 and coords[0, 0] > coords[-1, 0]:
            coords = np.flipud(coords)
        return coords

    def _validate_and_set_surface_coords(self, upper_coords: np.ndarray, lower_coords: np.ndarray):
        if not isinstance(upper_coords, np.ndarray) or upper_coords.ndim != 2 or upper_coords.shape[1] != 2:
            raise ValueError("Upper coords must be a NumPy array of shape (Nu, 2).")
        if not isinstance(lower_coords, np.ndarray) or lower_coords.ndim != 2 or lower_coords.shape[1] != 2:
            raise ValueError("Lower coords must be a NumPy array of shape (Nl, 2).")
        self._upper_surface = self._sort_surface(upper_coords.copy())
        self._lower_surface = self._sort_surface(lower_coords.copy())

    def _parse_combined_coordinates(self, coordinates: np.ndarray):
        if not isinstance(coordinates, np.ndarray) or coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("Combined coords must be a NumPy array of shape (N, 2).")
        le_index = np.argmin(coordinates[:, 0])
        if le_index == 0:
             raise ValueError(f"Ambiguous coordinate format for '{self.name}' with LE at index 0. Provide separate upper/lower surfaces.")
        self._upper_surface = np.flipud(coordinates[:le_index+1, :].copy())
        self._lower_surface = coordinates[le_index:, :].copy()

    def _normalize_and_align_surfaces(self):
        if self._upper_surface is None or self._lower_surface is None: return
        le_x_offset = self._upper_surface[0, 0]
        self._upper_surface[:, 0] -= le_x_offset
        self._lower_surface[:, 0] -= le_x_offset
        le_y_avg = (self._upper_surface[0, 1] + self._lower_surface[0, 1]) / 2.0
        self._upper_surface[0, 1] = le_y_avg; self._lower_surface[0, 1] = le_y_avg
        chord_length = self._upper_surface[-1, 0]
        if np.isclose(chord_length, 0.0) or chord_length < 0:
            print(f"Warning for airfoil '{self.name}': Chord length is {chord_length:.4f}. Skipping normalization by chord.")
            self._coordinates = None; return
        self._upper_surface /= chord_length; self._lower_surface /= chord_length
        self._upper_surface[-1, 0] = 1.0; self._lower_surface[-1, 0] = 1.0
        avg_te_y = (self._upper_surface[-1,1] + self._lower_surface[-1,1]) / 2.0
        self._upper_surface[-1,1] = avg_te_y; self._lower_surface[-1,1] = avg_te_y
        self._coordinates = None

    @property
    def coordinates(self) -> np.ndarray:
        if self._coordinates is None:
            if self._upper_surface is None or self._lower_surface is None: raise ValueError("Surfaces not defined.")
            self._coordinates = np.vstack((np.flipud(self._upper_surface), self._lower_surface[1:, :]))
        return self._coordinates

    @property
    def upper_surface_coords(self) -> np.ndarray:
        if self._upper_surface is None: raise ValueError("Upper surface not defined.")
        return self._upper_surface

    @property
    def lower_surface_coords(self) -> np.ndarray:
        if self._lower_surface is None: raise ValueError("Lower surface not defined.")
        return self._lower_surface

    def get_interpolated_coordinates(self, num_points_per_surface: int = 50, distribution: str = "cosine") -> np.ndarray:
        if num_points_per_surface < 2: raise ValueError("num_points_per_surface must be at least 2.")
        x_new = _cosine_spacing(0.0, 1.0, num_points_per_surface) if distribution == "cosine" else _linear_spacing(0.0, 1.0, num_points_per_surface)
        f_upper = interp1d(self.upper_surface_coords[:, 0], self.upper_surface_coords[:, 1], kind='cubic', fill_value="extrapolate")
        upper_new = np.vstack((x_new, f_upper(x_new))).T
        f_lower = interp1d(self.lower_surface_coords[:, 0], self.lower_surface_coords[:, 1], kind='cubic', fill_value="extrapolate")
        lower_new = np.vstack((x_new, f_lower(x_new))).T
        return np.vstack((np.flipud(upper_new), lower_new[1:, :]))

    def plot(self, ax: Optional[Any] = None, show: bool = True, title: Optional[str] = None, plot_interpolated: bool = False, num_points_interp: int =50) -> None:
        try: import matplotlib.pyplot as plt
        except ImportError: print("Matplotlib required for plotting."); return
        if ax is None: fig, ax = plt.subplots()
        coords_to_plot = self.get_interpolated_coordinates(num_points_interp) if plot_interpolated else self.coordinates
        label = f"{self.name} ({'Interp.' if plot_interpolated else 'Orig.'})"
        ax.plot(coords_to_plot[:, 0], coords_to_plot[:, 1], marker='.', markersize=3, linestyle='-', label=label)
        ax.set_aspect('equal', adjustable='box'); ax.set_xlabel("x/c"); ax.set_ylabel("y/c")
        ax.set_title(title if title is not None else self.name); ax.grid(True); ax.legend()
        if show: plt.show()


    def get_thickness_to_chord_ratio(self) -> float:
        """
        Calculates or retrieves the maximum thickness to chord ratio (t/c).

        Returns the value provided at initialization if available. Otherwise,
        calculates it geometrically from the surface coordinates. The geometric
        calculation may fail or be inaccurate for highly cambered airfoils with
        non-monotonic x-coordinates.
        """
        if self._thickness_to_chord_ratio is not None:
            return self._thickness_to_chord_ratio

        print(f"Warning: t/c for airfoil '{self.name}' not provided. Attempting geometric calculation.")

        try:
            upper_x_sorted_indices = np.argsort(self.upper_surface_coords[:, 0])
            upper_x = self.upper_surface_coords[upper_x_sorted_indices, 0]
            upper_y = self.upper_surface_coords[upper_x_sorted_indices, 1]

            lower_x_sorted_indices = np.argsort(self.lower_surface_coords[:, 0])
            lower_x = self.lower_surface_coords[lower_x_sorted_indices, 0]
            lower_y = self.lower_surface_coords[lower_x_sorted_indices, 1]

            if not np.all(np.diff(upper_x) > -1e-9) or not np.all(np.diff(lower_x) > -1e-9):
                 raise ValueError("X-coordinates are non-monotonic even after sorting.")

            common_x_min = max(np.min(upper_x), np.min(lower_x))
            common_x_max = min(np.max(upper_x), np.max(lower_x))
            if common_x_max <= common_x_min: return 0.0

            x_common = np.linspace(common_x_min, common_x_max, 201)

            f_upper = PchipInterpolator(upper_x, upper_y)
            y_upper_common = f_upper(x_common)
            f_lower = PchipInterpolator(lower_x, lower_y)
            y_lower_common = f_lower(x_common)

            thickness_dist = y_upper_common - y_lower_common
            self._thickness_to_chord_ratio = np.max(thickness_dist) if len(thickness_dist) > 0 else 0.0
            return self._thickness_to_chord_ratio
        except Exception as e:
            print(f"ERROR: Geometric t/c calculation for {self.name} failed: {e}. Returning 0.0. "
                  "This can happen with highly cambered airfoils. Consider providing t/c at initialization.")
            self._thickness_to_chord_ratio = 0.0
            return 0.0

    def add_polar(self, Re: float, alpha_deg: np.ndarray, cl: np.ndarray, cd: np.ndarray, cm: Optional[np.ndarray] = None):
        if not all(isinstance(arr, np.ndarray) for arr in [alpha_deg, cl, cd]): raise TypeError("alpha_deg, cl, cd must be NumPy arrays.")
        if not (alpha_deg.shape == cl.shape == cd.shape): raise ValueError("alpha_deg, cl, cd must have the same shape.")
        if cm is not None and cm.shape != alpha_deg.shape: raise ValueError("cm must have the same shape as alpha_deg if provided.")
        sort_indices = np.argsort(alpha_deg)
        self.polars[float(Re)] = {"alpha_deg": alpha_deg[sort_indices], "cl": cl[sort_indices], "cd": cd[sort_indices], "cm": cm[sort_indices] if cm is not None else np.zeros_like(alpha_deg)}
        self.stall_angle_deg = None; self.cl_stall = None; self.cd_stall = None; self._lift_curve_slope_cache = {}


    def _get_closest_Re_polar_data(self, target_Re: float) -> Optional[Dict[str, np.ndarray]]:
        if not self.polars: return None
        available_Res = np.array(list(self.polars.keys())); closest_Re_idx = np.argmin(np.abs(available_Res - target_Re)); closest_Re = available_Res[closest_Re_idx]
        if abs(closest_Re - target_Re) / target_Re > 0.2 and len(available_Res) > 1 : print(f"Warning for {self.name}: Target Re {target_Re:.2e} is >20% different from closest available polar Re {closest_Re:.2e}.")
        return self.polars[closest_Re]
    def _analyze_neuralfoil(self, alpha_deg: Union[float, np.ndarray], Re: float) -> Tuple[np.ndarray, np.ndarray]:
        if not NEURALFOIL_AVAILABLE: raise RuntimeError("NeuralFoil not installed.")
        results = nf.get_aero_from_coordinates(coordinates=self.coordinates, alpha=alpha_deg, Re=Re)
        return np.asarray(results['CL']), np.asarray(results['CD'])
    def _analyze_xfoil(self, alpha_deg: float, Re: float, timeout: float = 10.0) -> Tuple[float, float]:
        if not AEROSANDBOX_XFOIL_AVAILABLE: raise RuntimeError("AeroSandbox XFoil not installed.")
        asb_airfoil = ASBAirfoil(name=self.name, coordinates=self.coordinates)
        xf = ASBXFoil(airfoil=asb_airfoil, Re=Re, timeout=timeout, verbose=False, max_iter=30)
        try: result_dict = xf.alpha(alpha_deg); cl = result_dict.get('cl',0.0); cd = result_dict.get('cd',0.01)
        except Exception: cl, cd = 0.0, 0.01
        return float(cl), float(cd)


    def characterize_stall_properties(self, Re: float, analysis_method: Optional[str]=None, alpha_range_deg: Tuple[float, float]=(-5.0, 25.0), num_alpha_points: int=61) -> bool:
        method = analysis_method if analysis_method is not None else self.default_analysis_method; alphas = np.linspace(alpha_range_deg[0], alpha_range_deg[1], num_alpha_points); cls, cds = np.zeros_like(alphas), np.zeros_like(alphas)
        if method == "polar":
            polar = self._get_closest_Re_polar_data(Re)
            if not polar: print(f"Warning: No polar data for Re={Re:.2e} to characterize stall for {self.name}."); return False
            cl_interp = PchipInterpolator(polar["alpha_deg"], polar["cl"], extrapolate=True); cd_interp = PchipInterpolator(polar["alpha_deg"], polar["cd"], extrapolate=True)
            cls, cds = cl_interp(alphas), cd_interp(alphas)
        elif method == "neuralfoil":
            if not NEURALFOIL_AVAILABLE: print("NeuralFoil not available."); return False
            cls, cds = self._analyze_neuralfoil(alphas, Re)
        elif method == "xfoil":
            if not AEROSANDBOX_XFOIL_AVAILABLE: print("XFoil not available."); return False
            asb_af = ASBAirfoil(name=self.name, coordinates=self.coordinates); xf = ASBXFoil(airfoil=asb_af, Re=Re, verbose=False, max_iter=30)
            try: results = xf.alpha(alphas); cls = np.asarray(results['cl']); cds = np.asarray(results['cd'])
            except Exception as e: print(f"Warning: XFoil sweep failed: {e}"); return False
        else: raise ValueError(f"Unknown analysis_method: {method}")
        if len(cls) > 0 and not np.all(np.isnan(cls)):
            idx = np.nanargmax(cls); self.stall_angle_deg = float(alphas[idx]); self.cl_stall = float(cls[idx]); self.cd_stall = float(cds[idx]); return True
        return False


    def get_lift_curve_slope(self, Re: float, alpha0_deg: float=0.0, delta_alpha_deg: float=0.5, analysis_method: Optional[str]=None) -> float:
        method = analysis_method if analysis_method is not None else self.default_analysis_method
        cache_key = (Re, method, alpha0_deg, delta_alpha_deg)
        if cache_key in self._lift_curve_slope_cache: return self._lift_curve_slope_cache[cache_key]
        cl1, _ = self.get_lift_drag_coeffs(alpha0_deg-delta_alpha_deg, Re, analysis_method=method, apply_viterna_poststall=False)
        cl2, _ = self.get_lift_drag_coeffs(alpha0_deg+delta_alpha_deg, Re, analysis_method=method, apply_viterna_poststall=False)
        if abs(2*delta_alpha_deg) < 1e-6: return DEFAULT_CLA_2PI
        cl_alpha_per_deg = (cl2-cl1) / (2*delta_alpha_deg)
        cl_alpha_per_rad = cl_alpha_per_deg * (180.0/np.pi)
        self._lift_curve_slope_cache[cache_key] = cl_alpha_per_rad
        return cl_alpha_per_rad

    def get_lift_drag_coeffs(self, alpha_deg: float, Re: float, analysis_method: Optional[str]=None, apply_viterna_poststall: bool=True, fixed_cd_max_for_poststall: Optional[float]=None) -> Tuple[float,float]:
        method = analysis_method if analysis_method is not None else self.default_analysis_method; cl_direct, cd_direct = 0.0, 0.01
        if method=="polar":
            polar=self._get_closest_Re_polar_data(Re)
            if polar: cl_direct=float(PchipInterpolator(polar["alpha_deg"], polar["cl"], extrapolate=True)(alpha_deg)); cd_direct=float(PchipInterpolator(polar["alpha_deg"], polar["cd"], extrapolate=True)(alpha_deg))
            else: return self.get_lift_drag_coeffs(alpha_deg, Re, analysis_method=self.default_analysis_method if self.default_analysis_method!="polar" else "neuralfoil", apply_viterna_poststall=apply_viterna_poststall, fixed_cd_max_for_poststall=fixed_cd_max_for_poststall)
        elif method=="neuralfoil":
            if not NEURALFOIL_AVAILABLE: raise RuntimeError("NeuralFoil not installed."); cl_arr,cd_arr=self._analyze_neuralfoil(alpha_deg, Re); cl_direct=float(cl_arr.item(0)if isinstance(cl_arr,np.ndarray)and cl_arr.size==1 else cl_arr); cd_direct=float(cd_arr.item(0)if isinstance(cd_arr,np.ndarray)and cd_arr.size==1 else cd_arr)
        elif method=="xfoil":
            if not AEROSANDBOX_XFOIL_AVAILABLE: raise RuntimeError("XFoil not installed."); cl_direct,cd_direct=self._analyze_xfoil(alpha_deg, Re)
        else: raise ValueError(f"Unknown analysis_method: {method}")
        if not apply_viterna_poststall: return cl_direct, cd_direct
        if self.stall_angle_deg is None: self.characterize_stall_properties(Re, analysis_method=method)
        alpha_s,cl_s,cd_s=self.stall_angle_deg,self.cl_stall,self.cd_stall
        if alpha_s is None or cl_s is None or cd_s is None: return cl_direct, cd_direct
        if (alpha_deg*alpha_s>=0)and(abs(alpha_deg)>=abs(alpha_s)):
            tc_ratio=self.get_thickness_to_chord_ratio(); cd_max_eff=fixed_cd_max_for_poststall if fixed_cd_max_for_poststall is not None else self.cd_max_poststall; B1=cd_max_eff; A1=B1/2.0
            alpha_s_rad=np.radians(alpha_s); sin_alpha_s=np.sin(alpha_s_rad); cos_alpha_s=np.cos(alpha_s_rad)
            if np.isclose(sin_alpha_s,0.0):sin_alpha_s=1e-6*np.sign(alpha_s_rad if alpha_s_rad else 1.0)
            if np.isclose(cos_alpha_s,0.0):cos_alpha_s=1e-6*np.sign(np.pi/2.0-abs(alpha_s_rad))
            A2=(cl_s-cd_max_eff*sin_alpha_s*cos_alpha_s)*sin_alpha_s/(cos_alpha_s**2); B2=cd_s-(B1*sin_alpha_s**2)/cos_alpha_s
            alpha_rad=np.radians(alpha_deg); sin_alpha=np.sin(alpha_rad); cos_alpha=np.cos(alpha_rad); sin_alpha_safe=sin_alpha if not np.isclose(sin_alpha,0.0)else 1e-6*np.sign(alpha_rad if alpha_rad else 1.0)
            cl_post=A1*np.sin(2*alpha_rad)+A2*(cos_alpha**2)/sin_alpha_safe; cd_post=B1*sin_alpha**2+B2*cos_alpha; return cl_post,max(0.0001,cd_post)
        else: return cl_direct, cd_direct
    def __repr__(self) -> str:
        tc_val = "N/A";
        try: tc_val = f"{self.get_thickness_to_chord_ratio():.3f}"
        except Exception: pass
        stall_info = f"alpha_s={self.stall_angle_deg:.1f}deg" if self.stall_angle_deg is not None else "stall_not_characterized"
        return f"<Airfoil name='{self.name}', t/c={tc_val}, {stall_info}, polars_loaded={len(self.polars)}, method='{self.default_analysis_method}'>"

def generate_naca4_coordinates(naca_code: str, num_points_per_surface: int = 81,
                                 finite_te: bool = False, te_thickness_fraction: float = 0.001,
                                 **kwargs_for_airfoil_init) -> Airfoil:
    if not (isinstance(naca_code, str) and len(naca_code) == 4 and naca_code.isdigit()):
        raise ValueError("NACA code must be a 4-digit string (e.g., '2412').")

    m_param = int(naca_code[0]) / 100.0
    p_param = int(naca_code[1]) / 10.0
    t_param = int(naca_code[2:]) / 100.0

    # ** MODIFIED/FIXED PART **
    # Pass the known thickness-to-chord ratio to the constructor
    kwargs_for_airfoil_init['thickness_to_chord_ratio'] = t_param

    if num_points_per_surface < 2: raise ValueError("num_points_per_surface must be at least 2.")
    x = _cosine_spacing(0.0, 1.0, num_points_per_surface)
    a0=0.2969; a1=-0.1260; a2=-0.3516; a3=0.2843
    a4 = (te_thickness_fraction / 2.0 - (a0 + a1 + a2 + a3)) if finite_te else -0.1015
    yt = 5 * t_param * (a0*np.sqrt(x) + a1*x + a2*x**2 + a3*x**3 + a4*x**4)
    xu, xl, yu, yl = x.copy(), x.copy(), np.zeros_like(x), np.zeros_like(x)
    if np.isclose(p_param, 0.0) or np.isclose(m_param, 0.0):
        xu = x; xl = x; yu = yt; yl = -yt
    else:
        yc, dyc_dx = np.zeros_like(x), np.zeros_like(x)
        mask_front = x <= p_param; mask_rear = x > p_param
        if p_param > 0:
            xc_f = x[mask_front]; yc[mask_front] = (m_param/p_param**2)*(2*p_param*xc_f - xc_f**2)
            dyc_dx[mask_front] = (2*m_param/p_param**2)*(p_param - xc_f)
        if (1-p_param) > 0:
            xc_r = x[mask_rear]; yc[mask_rear] = (m_param/(1-p_param)**2)*((1-2*p_param) + 2*p_param*xc_r - xc_r**2)
            dyc_dx[mask_rear] = (2*m_param/(1-p_param)**2)*(p_param - xc_r)
        theta = np.arctan(dyc_dx)
        xu = x - yt * np.sin(theta); yu = yc + yt * np.cos(theta)
        xl = x + yt * np.sin(theta); yl = yc - yt * np.cos(theta)
    upper_coords = np.vstack((xu, yu)).T
    lower_coords = np.vstack((xl, yl)).T
    return Airfoil(name=f"NACA {naca_code}", upper_coords=upper_coords, lower_coords=lower_coords, **kwargs_for_airfoil_init)

def load_airfoil_from_file(filepath: Union[str, Path], airfoil_name: Optional[str] = None, **kwargs_for_airfoil_init) -> Airfoil:
    # ... (implementation as before) ...
    file_path = Path(filepath);
    if not file_path.exists(): raise FileNotFoundError(f"File not found: {file_path}")
    name_to_use = airfoil_name if airfoil_name is not None else file_path.stem
    lines = file_path.read_text().splitlines(); coords_list = []
    if lines:
        first_line_parts = lines[0].strip().split()
        is_coords = False
        if len(first_line_parts) >= 2:
            try: float(first_line_parts[0]); float(first_line_parts[1]); is_coords = True
            except ValueError: pass
        if not is_coords and airfoil_name is None: name_to_use = lines[0].strip(); lines = lines[1:]
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            try: coords_list.append((float(parts[0]), float(parts[1])))
            except ValueError: pass
    if not coords_list: raise ValueError(f"No valid coordinates in file: {file_path}")
    all_coords = np.array(coords_list)
    potential_split_idx = -1
    for i in range(1, len(all_coords) - 1):
        if (all_coords[i,0] > 0.9*np.max(all_coords[:,0]) and all_coords[i+1,0] < 0.1*np.max(all_coords[:,0]) and all_coords[i,0] > all_coords[i-1,0]):
            potential_split_idx = i + 1; break
    if potential_split_idx != -1:
        return Airfoil(name=name_to_use, upper_coords=all_coords[:potential_split_idx,:], lower_coords=all_coords[potential_split_idx:,:], **kwargs_for_airfoil_init)
    else:
        return Airfoil(name=name_to_use, coordinates=all_coords, **kwargs_for_airfoil_init)
