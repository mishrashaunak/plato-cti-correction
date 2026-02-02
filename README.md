# CTI Correction Pipeline for PLATO

This repository contains the Python implementation of the Charge Transfer Inefficiency (CTI) calibration and correction strategy for the **ESA PLATO Mission**, as described in the paper:

> **Impact of Charge Transfer Inefficiency on transit light-curves: A correction strategy for PLATO** > *S. Mishra, R. Samadi, D. Bérard (2025)* > Astronomy & Astrophysics (A&A)  
> [Link to ArXiv](https://arxiv.org/abs/2510.22092) | [DOI](https://doi.org/10.1051/0004-6361/202451554)

*(Note: The DOI link above is a placeholder. Once A&A assigns the final DOI, update it here).*

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
