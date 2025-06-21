# In a test script
from ductedfanlib.core import Blade, Rotor
from ductedfanlib.geometry.airfoils import generate_naca4_coordinates
from ductedfanlib.geometry.profiles import LinearDistribution
import numpy as np

# Create an airfoil
naca0012 = generate_naca4_coordinates("0012")

# Define blade profiles
chord_dist = LinearDistribution(start_value=0.1, end_value=0.05)
twist_dist = LinearDistribution(start_value=10.0, end_value=0.0) # Base twist

my_blade = Blade(
    airfoil_definition=naca0012,
    chord_profile=chord_dist,
    twist_profile=twist_dist
)

# Create a rotor with collective pitch
my_rotor = Rotor(
    num_blades=3,
    tip_radius=0.25,
    hub_radius=0.075,
    blade_definition=my_blade,
    collective_pitch_deg=5.0 # Apply +5 degrees collective pitch
)
print(my_rotor)

# --- Test get_radial_stations_properties ---

# 1. Using num_stations and spacing_type
print("\n--- Stations using num_stations (cosine) ---")
stations_auto_eta = my_rotor.get_radial_stations_properties(num_stations=5, spacing_type="cosine")
for i, s in enumerate(stations_auto_eta):
    print(f"  S{i+1}: eta={s['eta']:.3f}, r={s['radius_m']:.3f}m, base_twist={s['base_twist_deg']:.2f}deg, total_twist={s['twist_deg']:.2f}deg")

# 2. Using eta_values_override
custom_eta_points = np.array([0.0, 0.33, 0.66, 1.0])
print(f"\n--- Stations using eta_values_override: {custom_eta_points} ---")
stations_custom_eta = my_rotor.get_radial_stations_properties(eta_values_override=custom_eta_points)
for i, s in enumerate(stations_custom_eta):
    print(f"  S{i+1}: eta={s['eta']:.3f}, r={s['radius_m']:.3f}m, base_twist={s['base_twist_deg']:.2f}deg, total_twist={s['twist_deg']:.2f}deg")

# Example of an invalid eta (should be caught by Blade class now if eta_values_override contains it)
# invalid_eta_points = np.array([-0.1, 0.5, 1.1])
# try:
#     my_rotor.get_radial_stations_properties(eta_values_override=invalid_eta_points)
# except ValueError as e:
#     print(f"\nCaught expected error for invalid eta: {e}")

# Example: No station definition provided
# try:
#     my_rotor.get_radial_stations_properties()
# except ValueError as e:
#     print(f"\nCaught expected error for no station def: {e}")