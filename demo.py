import numpy as np
from cti_model import run_cdm_parallel_cumul, run_cdm_parallel_cumul_radial_poly
import time

def print_verification(name, input_arr, output_arr):
    """Helper to print results consistently."""
    print(f"\n--- Verification: {name} ---")
    diff = input_arr - output_arr
    max_loss = np.max(diff)
    total_loss = np.sum(diff)
    
    if max_loss > 0:
        print(f"SUCCESS: CTI applied.")
        print(f"  Max pixel loss:   {max_loss:.4f} e-")
        print(f"  Total signal loss: {total_loss:.4f} e-")
    else:
        print(f"WARNING: Output identical to input. Check parameters.")
    
    # Sanity check
    assert not np.array_equal(input_arr, output_arr), f"Error: {name} returned exact input!"

def test_standard_cdm():
    print("1. Running Standard CTI model...")
    # Parameters
    image = np.ones((100, 100)) * 1000.0
    y0 = 0
    beta = 0.3
    vg = 1.0e-10
    t = 0.005
    fwc = 200000.0
    vth = 1.0e7
    tr = np.array([1.0e-3])
    sigma = np.array([1.0e-15])
    
    # Create a gradient trap map
    cnt = np.zeros((4510, 100, 1))
    for i in range(4510):
        cnt[i, :, 0] = i * 10.0

    # Run
    start = time.time()
    result = run_cdm_parallel_cumul(image, y0, beta, vg, t, fwc, vth, tr, cnt, sigma)
    end = time.time()
    print(f"   Execution time: {end - start:.4f}s")
    
    print_verification("Standard CDM", image, result)

def test_radial_cdm():
    print("\n2. Running Radial Polynomial CTI model...")
    # Parameters
    rows, cols = 100, 100
    image = np.ones((rows, cols)) * 1000.0
    colindex = np.arange(cols)
    y0 = 0
    beta = 0.3
    vg = 1.0e-10
    t = 0.005
    fwc = 200000.0
    vth = 1.0e7
    tr = np.array([1.0e-3])
    sigma = np.array([1.0e-15])
    
    # Radial parameters
    a = np.array([[0.0, 100.0]]) # Linear increase: 0 + 100*R
    xc, yc = 2255, 2255

    # Run
    start = time.time()
    result = run_cdm_parallel_cumul_radial_poly(
        image, colindex, y0, beta, vg, t, fwc, vth, tr, a, xc, yc, sigma
    )
    end = time.time()
    print(f"   Execution time: {end - start:.4f}s")
    
    print_verification("Radial CDM", image, result)

if __name__ == "__main__":
    print("Starting combined verification...\n")
    test_standard_cdm()
    test_radial_cdm()
    print("\nAll tests passed successfully.")
