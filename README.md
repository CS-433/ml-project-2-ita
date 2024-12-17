# ML4Science: Extending the POD-DL-ROM paradigm to space-time in hemodynamics
Project 2 of **EPFL CS-433 Machine Learning**.

**Team name**: ITA  
**Team members**: Manuel Curnis, Alessandro Di Maria, Vincenzo Pio Scurria  
**Supervisors**: Riccardo Tenderini, Simone Deparis


---

## Overview
This project aims to develop a space-time Proper Orthogonal
Decomposition – Deep Learning - Reduced Order Model (POD-DL-ROM) for deriving a data-driven reduced-order solution to the time-dependent Navier-Stokes equations in a blood vessel.
architecture includes an autoencoder with dense layers that efficiently compresses POD coefficients of velocity (spatial and temporal) into a lowerdimensional latent space and a fully connected neural network
that in parallel maps physical parameters to these latent variables.

## usage
- Clone the repository.
- Create a Conda environment: conda create --name project2 python=3.12.8 -y (PyTorch may not be compatible with the latest Python  versions )
- Activate the environment: conda activate project2
- Install the required dependencies: pip install -r requirements.txt
- Launch Jupyter Lab: jupyter lab
- Run the file "run.ipynb" (already executed )

## Files

### 1. run.ipynb

This Jupyter notebook  performs the following tasks: data loading, preprocessing, model training, hyperparameter optimization, testing, and cross-validation.

### 2. data_handler.py
Functions provided by: Riccardo Tenderini (SCI SB SD Group, EPFL) to load the data, project solution and visualize solutions in 3D

### 2. models.py
This file contains the deep learning models that will be trained in run.py

### 3. ML_handler.py
This file contains the functions used to train and validate the models

### 4. plot_handler.py
contains a functions to plot train loss and validation loss






