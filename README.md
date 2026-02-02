# CTI Correction Pipeline for PLATO

This repository contains the Python implementation of the Charge Transfer Inefficiency (CTI) calibration and correction strategy for the **ESA PLATO Mission**, as described in the paper:

> **Impact of Charge Transfer Inefficiency on transit light-curves: A correction strategy for PLATO** > *S. Mishra, R. Samadi, D. Bérard (2025)* > Astronomy & Astrophysics (A&A)  
> [Link to ArXiv](https://arxiv.org/abs/2510.22092) | [DOI](Insert_DOI_Once_Published)

## Overview
This pipeline models the spatial variation of trap density across the CCD and corrects CTI-induced biases in photometric transit measurements. The routines are adapted from the **Pyxel** framework and optimized for the specific radiation environment expected for PLATO.

## Dependencies
* Python 3.x
* Numpy
* Numba (for JIT compilation and parallelization)

## Citation
If you use this code in your research, please cite the paper:

@article{mishra2025cti,
  title={Impact of Charge Transfer Inefficiency on transit light-curves: A correction strategy for PLATO},
  author={Mishra, S. and Samadi, R. and Bérard, D.},
  journal={Astronomy \& Astrophysics},
  year={2025}
}
