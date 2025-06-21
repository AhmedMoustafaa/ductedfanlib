from ductedfanlib.geometry.airfoils import Airfoil, generate_naca4_coordinates, load_airfoil_from_file
# Test NACA generator
naca2412 = generate_naca4_coordinates("2412", num_points_per_surface=200)
print(naca2412)
naca2412.plot() # Will show the plot
naca2412.plot(plot_interpolated=True, num_points_interp=41)
print(naca2412.get_lift_drag_coeffs(10,1e6))
print(naca2412.get_lift_curve_slope(10,1e6))
print(naca2412.get_thickness_to_chord_ratio())