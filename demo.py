import numpy as np
from cti_model import run_cdm_parallel_cumul

# Dummy parameters to verify the code runs
image = np.ones((100, 100)) * 1000.0  # 100x100 image with 1000 electrons
y0 = 0
beta = 0.3
vg = 1.0e-10
t = 25.0
fwc = 200000.0
vth = 1.0e7
tr = np.array([1.0e-3])     # 1 trap species
sigma = np.array([1.0e-15]) # 1 trap species
# Dummy cumulated trap density map (4510 rows, 100 cols, 1 species)
cnt = np.zeros((4510, 100, 1)) 

print("Running CTI model...")
result = run_cdm_parallel_cumul(image, y0, beta, vg, t, fwc, vth, tr, cnt, sigma)
print("Done! Result shape:", result.shape)
