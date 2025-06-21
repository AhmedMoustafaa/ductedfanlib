# In a test script
from ductedfanlib.core import Blade
from ductedfanlib.geometry.airfoils import Airfoil, generate_naca4_coordinates
from ductedfanlib.geometry.profiles import LinearDistribution
from collections import OrderedDict

# Create some airfoils
af_root = generate_naca4_coordinates("4415", 61) # Thicker root airfoil
af_mid = generate_naca4_coordinates("4412", 61)
af_tip = generate_naca4_coordinates("4410", 61)  # Thinner tip airfoil

# Define airfoil sections along the span (eta values as keys)
# Using OrderedDict to show intent, though Blade __init__ will sort it
airfoil_sections = {
    0.0: af_root,
    0.6: af_mid,  # Airfoil 'af_mid' applies from eta=0.6 up to (but not including) eta=0.9
    0.9: af_tip   # Airfoil 'af_tip' applies from eta=0.9 up to eta=1.0
}
# If you define it with unsorted keys, the Blade __init__ sorts them.
# airfoil_sections = {0.6: af_mid, 0.0: af_root, 1.0: af_tip} # Will be sorted internally

chord_dist = LinearDistribution(start_value=0.2, end_value=0.1)
twist_dist = LinearDistribution(start_value=25.0, end_value=10.0)

blade_with_sections = Blade(
    airfoil_definition=airfoil_sections,
    chord_profile=chord_dist,
    twist_profile=twist_dist,
    span=0.25
)
print(blade_with_sections)

print(f"\nAirfoil at eta=0.0: {blade_with_sections.get_airfoil_at_span_eta(0.0).name}")   # Expected: NACA 4415
print(f"Airfoil at eta=0.3: {blade_with_sections.get_airfoil_at_span_eta(0.3).name}")   # Expected: NACA 4415
print(f"Airfoil at eta=0.59: {blade_with_sections.get_airfoil_at_span_eta(0.59).name}") # Expected: NACA 4415
print(f"Airfoil at eta=0.6: {blade_with_sections.get_airfoil_at_span_eta(0.6).name}")   # Expected: NACA 4412
print(f"Airfoil at eta=0.8: {blade_with_sections.get_airfoil_at_span_eta(0.8).name}")   # Expected: NACA 4412
print(f"Airfoil at eta=0.89: {blade_with_sections.get_airfoil_at_span_eta(0.89).name}")# Expected: NACA 4412
print(f"Airfoil at eta=0.9: {blade_with_sections.get_airfoil_at_span_eta(0.9).name}")   # Expected: NACA 4410
print(f"Airfoil at eta=1.0: {blade_with_sections.get_airfoil_at_span_eta(1.0).name}")   # Expected: NACA 4410

# try:
#     blade_with_sections.get_chord_at_span_eta(1.1) # Should raise ValueError
# except ValueError as e:
#     print(f"\nCaught expected error: {e}")