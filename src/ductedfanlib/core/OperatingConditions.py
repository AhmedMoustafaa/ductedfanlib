from dataclasses import dataclass

@dataclass
class OperatingConditions:
    """A data class to hold operating conditions."""
    axial_velocity_ms: float
    rpm: float
    rho_kgm3: float = 1.225
    mu_Pas: float = 1.81e-5
