# CTI Correction Pipeline for PLATO

This repository contains the Python implementation of the Charge Transfer Inefficiency (CTI) calibration and correction strategy for the **ESA PLATO Mission**, as described in the paper:

> **Impact of Charge Transfer Inefficiency on transit light-curves: A correction strategy for PLATO** > *S. Mishra, R. Samadi, D. Bérard (2025)* > Astronomy & Astrophysics (A&A)  
> [Link to ArXiv](https://arxiv.org/abs/2510.22092) | [DOI](https://doi.org/10.48550/arXiv.2510.22092)

## Overview
This pipeline models the spatial variation of trap density across the CCD and corrects CTI-induced biases in photometric transit measurements. The routines are adapted from the **Pyxel** framework and optimized for the specific radiation environment expected for PLATO.

## Repository Structure
* `cti_model.py`: Contains the core Numba-accelerated functions for CTI simulation and correction.
* `demo.py`: A simple script demonstrating how to run the model with dummy data.
* `requirements.txt`: List of dependencies.

## Installation
You can install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

## Usage
The core logic is contained in cti_model.py. You can import the functions directly into your Python scripts:
```from cti_model import run_cdm_parallel_cumul

# See demo.py for a complete example of input parameters
```
## Citation

If you use this code in your research, please cite the paper:

```bibtex
@article{mishra2025,
  title   = {Impact of Charge Transfer Inefficiency on transit light-curves: A correction strategy for PLATO},
  author  = {Mishra, S. and Samadi, R. and Bérard, D.},
  journal = {Astronomy \& Astrophysics},
  eprint={2510.22092},
  archivePrefix={arXiv},
  primaryClass= [astro-ph.EP]
  year    = {2025},
  note    = {In press}
}
```
