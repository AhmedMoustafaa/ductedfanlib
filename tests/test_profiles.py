# In a test script
from ductedfanlib.geometry.profiles import ConstantDistribution, LinearDistribution, discretize_distribution
import numpy as np

# Test Constant
cd = ConstantDistribution(value=5.5)
print(f"Constant at eta=0.5: {cd(0.5)}")
print(f"Constant with eta array: {cd(np.array([0.1, 0.5, 0.9]))}")

# Test Linear
ld = LinearDistribution(start_value=10, end_value=20)
print(f"Linear at eta=0.0: {ld(0.0)}")
print(f"Linear at eta=0.5: {ld(0.5)}")
print(f"Linear at eta=1.0: {ld(1.0)}")
print(f"Linear with eta array: {ld(np.array([0.0, 0.25, 0.5, 0.75, 1.0]))}")

# Test Discretization
eta_vals, prop_vals = discretize_distribution(ld, num_points=5, spacing="linear")
print(f"Discretized Linear (linear spacing): eta={eta_vals}, values={prop_vals}")

eta_vals_cos, prop_vals_cos = discretize_distribution(ld, num_points=5, spacing="cosine")
print(f"Discretized Linear (cosine spacing): eta={eta_vals_cos}, values={prop_vals_cos}")