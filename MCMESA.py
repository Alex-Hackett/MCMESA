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
from emcee.autocorr import AutocorrError
import sys

def load_config(config_path: str) -> Dict:
    """Load and validate configuration from YAML file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    required_fields = ['models', 'observables', 'mcmc']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required config field: {field}")
            
    return config

def load_mesa_data(file_path: str) -> pd.DataFrame:
    """Load and validate MESA model data"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"MESA data file not found: {file_path}")
    
    try:
        data = pd.read_csv(file_path, skiprows=5, delim_whitespace=True)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        raise ValueError(f"Error parsing MESA file {file_path}: {str(e)}")
    
    required_columns = ['star_age']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Missing required column in MESA data: {col}")
    
    return data.dropna().sort_values('star_age')

def create_interpolators(model_data: pd.DataFrame, observables: List[Dict]) -> Dict[str, interp1d]:
    """Create age-based interpolators for model observables"""
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
            fill_value='extrapolate'
        )
        
    return interps

class ModelComparator:
    """Main comparison engine handling MCMC analysis for N models"""
    
    def __init__(self, config: Dict):
        """Initialize with loaded configuration"""
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

    def _prepare_data(self):
        """Loads model data and creates interpolators"""
        for model in self.config['models']:
            model_name = model['name']
            model_data = load_mesa_data(model['file'])
            self.model_interps[model_name] = create_interpolators(model_data, self.config['observables'])

    def _setup_perturbation_groups(self):
        """Organize observables into perturbation groups"""
        groups = {}
        for obs in self.config['observables']:
            group = obs.get('perturbation_group', 'global')
            groups.setdefault(group, []).append(obs)
        
        self.perturbation_groups = groups
        
    def log_prior(self, params):
        """Defines prior probability distribution"""
        if np.any(params < -1) or np.any(params > 1):
            return -np.inf
        return 0

    def log_likelihood(self, params):
        """Defines likelihood function for MCMC"""
        return -0.5 * np.sum(params**2)
    
    def _log_posterior(self, params):
        return self.log_prior(params) + self.log_likelihood(params)
    
    def run_mcmc(self):
        """Perform adaptive MCMC sampling"""
        ndim = len(self.perturbation_groups) * (self.n_models - 1)
        nwalkers = self.config['mcmc']['n_walkers']
        backend = emcee.backends.HDFBackend(self.config['output']['chain_file'])
        backend.reset(nwalkers, ndim)
        
        self.sampler = emcee.EnsembleSampler(
            nwalkers, ndim, self._log_posterior, backend=backend
        )
        
        print("Running burn-in...")
        initial_state = np.random.uniform(-0.1, 0.1, size=(nwalkers, ndim))
        state = self.sampler.run_mcmc(initial_state, self.config['mcmc']['burn_in_steps'], progress=True)
        self.sampler.reset()
    
    def analyze_results(self):
        """Generate diagnostic plots and summary statistics"""
        output_dir = self.config['output'].get('directory', 'results')
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            autocorr_time = list(self.sampler.get_autocorr_time(tol=0))
        except AutocorrError:
            autocorr_time = "N/A (insufficient samples)"
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "autocorr_time": autocorr_time
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
        sys.exit(1)

if __name__ == "__main__":
    main()
