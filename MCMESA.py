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
    """Main comparison engine handling MCMC analysis"""
    
    def __init__(self, config: Dict):
        """Initialize with loaded configuration"""
        self.config = config
        self.obs_data = None
        self.model_interps = {}
        self.perturbation_groups = {}
        self.sampler = None
        
        self._prepare_data()
        self._setup_perturbation_groups()
        
    def _prepare_data(self):
        """Load and validate all input data"""
        # Load observed data
        obs_path = self.config['observations']['file_path']
        self.obs_data = pd.read_csv(obs_path)
        
        # Validate observations
        required_cols = ['age'] + [obs['data_column'] for obs in self.config['observables']]
        for col in required_cols:
            if col not in self.obs_data.columns:
                raise ValueError(f"Missing column in observed data: {col}")
        
        # Load MESA models
        for model in self.config['models']:
            data = load_mesa_data(model['file_path'])
            self.model_interps[model['name']] = create_interpolators(
                data, self.config['observables']
            )
            
    def _setup_perturbation_groups(self):
        """Group observables by perturbation parameter"""
        groups = {}
        for idx, obs in enumerate(self.config['observables']):
            group = obs.get('perturbation_group', 'global')
            groups.setdefault(group, []).append(obs)
            
        self.perturbation_groups = groups
        
    def _model_prediction(self, age: float, params: np.ndarray) -> Dict[str, float]:
        """Calculate perturbed model prediction for given age and parameters"""
        predictions = {}
        param_idx = 0
        
        for group_name, observables in self.perturbation_groups.items():
            alpha = params[param_idx]
            param_idx += 1
            
            for obs in observables:
                col = obs['model_column']
                m1 = self.model_interps[self.config['models'][0]['name']][col](age)
                m2 = self.model_interps[self.config['models'][1]['name']][col](age)
                predictions[col] = m1 + alpha * (m2 - m1)
                
        return predictions
        
    def log_prior(self, params: np.ndarray) -> float:
        """Uniform priors for perturbation parameters"""
        if np.any(params < -1) or np.any(params > 1):
            return -np.inf
        return 0.0
        
    def log_likelihood(self, params: np.ndarray) -> float:
        """Calculate log likelihood for given parameters"""
        chi2 = 0.0
        valid_points = 0
        
        for _, row in self.obs_data.iterrows():
            try:
                pred = self._model_prediction(row['age'], params)
            except ValueError:
                return -np.inf
                
            for obs in self.config['observables']:
                data_col = obs['data_column']
                err_col = obs['error_column']
                model_col = obs['model_column']
                
                if np.isnan(row[err_col]) or row[err_col] <= 0:
                    continue
                    
                residual = (row[data_col] - pred[model_col]) / row[err_col]
                chi2 += residual**2
                valid_points += 1
                
        if valid_points == 0:
            return -np.inf
            
        return -0.5 * chi2 - 0.5 * valid_points * np.log(2*np.pi)
        
    def run_mcmc(self):
        """Perform adaptive MCMC sampling with convergence checking"""
        # Setup sampler
        ndim = len(self.perturbation_groups)
        nwalkers = self.config['mcmc']['n_walkers']
        
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
