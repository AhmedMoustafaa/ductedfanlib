"""
Single authoritative definition of OperatingConditions.
The duplicate @dataclass in study.py is removed and imports from here instead.
"""
from dataclasses import dataclass, field


@dataclass
class OperatingConditions:
    """Operating conditions for a BEMT or ADT analysis run."""
    axial_velocity_ms: float          # freestream axial velocity (m/s); 0 for hover
    rpm: float                        # rotational speed (RPM)
    rho_kgm3: float = 1.225           # air density (kg/m³)
    mu_Pas: float   = 1.81e-5         # dynamic viscosity (Pa·s)
    altitude_m: float = 0.0           # informational only (m)

    def __post_init__(self):
        if self.rpm <= 0:
            raise ValueError("rpm must be positive.")
        if self.rho_kgm3 <= 0:
            raise ValueError("rho_kgm3 must be positive.")
        if self.mu_Pas <= 0:
            raise ValueError("mu_Pas must be positive.")
        if self.axial_velocity_ms < 0:
            raise ValueError("axial_velocity_ms must be >= 0.")

    @property
    def omega_rads(self) -> float:
        """Rotational speed in rad/s."""
        import math
        return self.rpm * 2.0 * math.pi / 60.0

    @property
    def is_hover(self) -> bool:
        return self.axial_velocity_ms < 0.5
