# MCMESA - MESA Model Comparator

A Python tool for statistical comparison of stellar evolution models from MESA using observational data. MCMESA employs perturbative MCMC analysis to evaluate model differences and estimate parameters.

## Features

- Perturbative analysis of MESA model differences
- Adaptive MCMC sampling with convergence diagnostics
- YAML-based configuration
- Multiple perturbation parameters
- Automated visualization and result saving

## Installation

Requirements: Python 3.8+, MESA output files

```bash
pip install numpy pandas scipy emcee corner matplotlib pyyaml h5py
```

## Usage

1. Create a configuration file (`config.yaml`):

```yaml
models:
  - name: "model1"
    file_path: "data/mesa_model1.data"
  - name: "model2" 
    file_path: "data/mesa_model2.data"
observations:
  file_path: "obs_data.csv"
observables:
  - model_column: "log_L"
    data_column: "luminosity"
    error_column: "luminosity_err"
    perturbation_group: "group1"
```

2. Run:
```bash
python MCMESA.py
```

## Configuration

- `models`: List of MESA models to compare
- `observables`: Observable parameters and their mappings
- `mcmc`: Sampling parameters (walkers, steps, burn-in)
- `output`: Output directory and chain file location

## Output

- Trace plots for MCMC convergence
- Corner plots for parameter distributions
- Posterior predictive checks
- Statistical summary in YAML format
- MCMC chain data in HDF5 format

## Algorithm

MCMESA interpolates models over stellar age and uses perturbation parameters (α) to blend models: M(α) = M₁ + α(M₂ - M₁). MCMC sampling explores the parameter space and automatically monitors convergence.

## License

Open source. Please include attribution when using or modifying.
