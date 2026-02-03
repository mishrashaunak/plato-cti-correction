import numpy as np
from cti_model import run_cdm_parallel_cumul

# 1. Setup Parameters
# -------------------
image = np.ones((100, 100)) * 1000.0  # 100x100 image with 1000 electrons flat field
y0 = 0
beta = 0.3
vg = 1.0e-10
t = 0.005        # Reduced time to make transfer faster/more realistic for testing
fwc = 200000.0
vth = 1.0e7
tr = np.array([1.0e-3])      # 1 trap species (release time)
sigma = np.array([1.0e-15])  # 1 trap species (cross section)

# 2. Populate Trap Density
# ------------------------
# We add a gradient of traps so the effect varies across the image
# cnt shape: (4510 rows, 100 cols, 1 species)
cnt = np.zeros((4510, 100, 1))
for i in range(4510):
    cnt[i, :, 0] = i * 10.0  # Increasing trap count down the rows

print("Running CTI model (Compiling with Numba, this may take a moment)...")

# 3. Run Model
# ------------
result = run_cdm_parallel_cumul(image, y0, beta, vg, t, fwc, vth, tr, cnt, sigma)

# 4. Verification
# ---------------
print("\n--- Verification Results ---")
print(f"Input Shape:  {image.shape}")
print(f"Output Shape: {result.shape}")

# Check if the image changed
diff = image - result
max_loss = np.max(diff)
total_loss = np.sum(diff)

if max_loss > 0:
    print(f"SUCCESS: CTI applied. Max pixel signal loss: {max_loss:.4f} e-")
    print(f"Total signal loss across image: {total_loss:.4f} e-")
else:
    print("WARNING: Output identical to input. Check trap density (cnt) or parameters.")

# Simple sanity check: First row should have less trailing (less traps traversed) 
# than the last row in a parallel readout context if y0=0 implies readout at y=0? 
# Actually, usually y=0 is readout, so higher rows see more traps.
# Let's just check that pixels are not identical.
assert not np.array_equal(image, result), "Error: The model returned the exact input image!"
