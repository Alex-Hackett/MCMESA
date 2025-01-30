"""
MESA Model Comparator with Perturbative MCMC Analysis

A tool to statistically compare two stellar evolution models from MESA code using
observational data within a perturbative framework. Supports multiple perturbation
parameters, adaptive MCMC sampling, and comprehensive diagnostics.

Key features:
- YAML configuration for flexible setup
- Multiple perturbation parameters grouped by observable
- Adaptive sampling with convergence diagnostics
- Automatic result saving and visualization
- Comprehensive input validation
"""

import os
import yaml
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import emcee
import corner
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple

def load_config(config_path: str) -> Dict:
    """Load and validate configuration from YAML file
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing validated configuration
    
    Raises:
        FileNotFoundError: If config file is missing
        ValueError: For invalid configuration parameters
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Validate required fields
    required_fields = ['models', 'observables', 'mcmc']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required config field: {field}")
            
    return config

def load_mesa_data(file_path: str) -> pd.DataFrame:
    """Load and validate MESA model data
    
    Args:
        file_path: Path to MESA history file
        
    Returns:
        Cleaned DataFrame with model data
    
    Raises:
        FileNotFoundError: If data file is missing
        ValueError: For invalid/missing columns
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"MESA data file not found: {file_path}")
        
    try:
        data = pd.read_csv(file_path, skiprows=5, delim_whitespace=True)
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing MESA file {file_path}: {str(e)}")
    
    required_columns = ['star_age']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Missing required column in MESA data: {col}")
            
    return data.dropna()

def create_interpolators(model_data: pd.DataFrame, observables: List[Dict]) -> Dict[str, interp1d]:
    """Create age-based interpolators for model observables
    
    Args:
        model_data: Loaded MESA model data
        observables: List of observable configurations
        
    Returns:
        Dictionary mapping column names to interpolation functions
    """
    age = model_data['star_age'].values
    interps = {}
    
    for obs in observables:
        col = obs['model_column']
        if col not in model_data.columns:
            raise ValueError(f"Model column {col} not found in MESA data")
            
        interps[col] = interp1d(
            age, 
            model_data[col].values,
            bounds_error=False,
            fill_value='extrapolate',
            assume_sorted=False
        )
        
    return interps


class ModelComparator:
"""Main comparison engine handling MCMC analysis for N models"""

def __init__(self, config: Dict):
    """Initialize with loaded configuration
    
    Args:
        config: Configuration dictionary from YAML
    Raises:
        ValueError: If invalid configuration or insufficient models
    """
    self.config = config
    self.obs_data = None
    self.model_interps = {}
    self.perturbation_groups = {}
    self.sampler = None
    self.n_models = len(config['models'])
    
    if self.n_models < 2:
        raise ValueError("At least two models required for comparison")
        
    self._prepare_data()
    self._setup_perturbation_groups()

def _model_prediction(self, age: float, params: np.ndarray) -> Dict[str, float]:
    """Calculate combined model prediction using multiple perturbation parameters
    
    Args:
        age: Stellar age to evaluate
        params: Array of perturbation parameters (alphas)
        
    Returns:
        Dictionary of observable predictions
        
    Raises:
        ValueError: For invalid model predictions
    """
    predictions = {}
    param_idx = 0
    base_model = self.config['models'][0]['name']
    base_interps = self.model_interps[base_model]
    
    for group_name, observables in self.perturbation_groups.items():
        # Get perturbation parameters for this group
        n_params = self.n_models - 1
        alphas = params[param_idx:param_idx+n_params]
        param_idx += n_params
        
        for obs in observables:
            col = obs['model_column']
            try:
                base_val = base_interps[col](age)
            except ValueError:
                raise ValueError(f"Invalid prediction for {col} at age {age}")
            
            # Start with base model prediction
            prediction = base_val
            
            # Add perturbations from other models
            for i, alpha in enumerate(alphas):
                model_name = self.config['models'][i+1]['name']
                model_val = self.model_interps[model_name][col](age)
                prediction += alpha * (model_val - base_val)
            
            predictions[col] = prediction
            
    return predictions

def _setup_perturbation_groups(self):
    """Organize observables into perturbation groups with parameter tracking"""
    groups = {}
    for obs in self.config['observables']:
        group = obs.get('perturbation_group', 'global')
        groups.setdefault(group, []).append(obs)
        
    self.perturbation_groups = groups
    self.n_params_per_group = self.n_models - 1

def run_mcmc(self):
    """Perform adaptive MCMC sampling with parameter space scaling"""
    # Calculate MCMC dimensions based on model count
    ndim = len(self.perturbation_groups) * (self.n_models - 1)
    nwalkers = self.config['mcmc']['n_walkers']
    
    # Initialize backend and sampler
    backend = emcee.backends.HDFBackend(self.config['output']['chain_file'])
    backend.reset(nwalkers, ndim)
    
    self.sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        lambda params: self.log_prior(params) + self.log_likelihood(params),
        backend=backend
    )
    
    # Initial burn-in phase
    print("Running burn-in...")
    initial_state = np.random.uniform(-0.1, 0.1, size=(nwalkers, ndim))
    max_n = self.config['mcmc'].get('max_steps', 10000)
    
    state = self.sampler.run_mcmc(
        initial_state, 
        self.config['mcmc']['burn_in_steps'],
        progress=True
    )
    self.sampler.reset()
    
    # Main sampling with convergence check
    print("\nRunning main chain...")
    autocorr_est = []
    converged = False
    
    for sample in self.sampler.sample(state, iterations=max_n, progress=True):
        if self.sampler.iteration % 100 == 0:
            try:
                autocorr = self.sampler.get_autocorr_time(tol=0)
                autocorr_est.append(autocorr)
                
                # Check convergence criteria
                if np.all(autocorr / autocorr_est[0] < 1.1) and len(autocorr_est) > 10:
                    converged = True
                    break
                    
            except emcee.autocorr.AutocorrError:
                continue
                
    if not converged:
        print(f"Warning: Chain did not converge within {max_n} steps")
        
def analyze_results(self):
    """Generate diagnostic plots and summary statistics"""
    # Create output directory
    output_dir = self.config['output'].get('directory', 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Trace plot
    fig, axes = plt.subplots(len(self.perturbation_groups), 1, figsize=(10, 7))
    samples = self.sampler.get_chain()
    
    for i, group in enumerate(self.perturbation_groups.keys()):
        axes[i].plot(samples[:, :, i], alpha=0.4)
        axes[i].set_ylabel(f"α ({group})")
        
    plt.xlabel("MCMC Step")
    plt.savefig(f"{output_dir}/trace_plot.png")
    plt.close()
    
    # Corner plot
    flat_samples = self.sampler.get_chain(discard=100, thin=15, flat=True)
    labels = [f"α ({group})" for group in self.perturbation_groups.keys()]
    
    fig = corner.corner(
        flat_samples,
        labels=labels,
        show_titles=True,
        title_kwargs={"fontsize": 12}
    )
    plt.savefig(f"{output_dir}/corner_plot.png")
    plt.close()
    
    # Posterior predictive check
    plt.figure(figsize=(10, 6))
    n_samples = 100
    
    for obs in self.config['observables']:
        data_col = obs['data_column']
        ages = self.obs_data['age'].values
        obs_vals = self.obs_data[data_col].values
        errors = self.obs_data[obs['error_column']].values
        
        # Plot observed data
        plt.errorbar(ages, obs_vals, yerr=errors, fmt='o', label=data_col)
        
        # Plot model predictions
        for idx in np.random.choice(len(flat_samples), n_samples):
            params = flat_samples[idx]
            preds = [self._model_prediction(age, params)[obs['model_column']] 
                    for age in ages]
            plt.plot(ages, preds, alpha=0.05, color='gray')
            
    plt.xlabel("Age (yr)")
    plt.ylabel("Observable Value")
    plt.legend()
    plt.savefig(f"{output_dir}/posterior_predictive.png")
    plt.close()
    
    # Save summary statistics
    summary = {
        "timestamp": datetime.now().isoformat(),
        "mean_alpha": list(np.mean(flat_samples, axis=0)),
        "median_alpha": list(np.median(flat_samples, axis=0)),
        "std_alpha": list(np.std(flat_samples, axis=0)),
        "autocorr_time": list(self.sampler.get_autocorr_time(tol=0)),
    }
    
    with open(f"{output_dir}/summary.yaml", 'w') as f:
        yaml.dump(summary, f)

def main():
    """Main execution flow"""
    try:
        config = load_config("config.yaml")
        comparator = ModelComparator(config)
        comparator.run_mcmc()
        comparator.analyze_results()
        print("\nAnalysis complete!")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
