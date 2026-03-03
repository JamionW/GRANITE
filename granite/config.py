"""
GRANITE Configuration Management

Provides centralized configuration loading with CLI override support.
"""
import os
import yaml
from typing import Dict, Any, Optional


class GRANITEConfig:
    """
    Configuration manager that loads from YAML and applies CLI overrides.
    
    Priority: CLI arguments > config file > defaults
    """
    
    DEFAULT_CONFIG_PATH = os.path.join(
        os.path.dirname(__file__), '..', 'config.yaml'
    )
    
    DEFAULTS = {
        'data': {
            'state_fips': '47',
            'county_fips': '065',
            'census_year': 2020,
        },
        'processing': {
            'verbose': False,
            'random_seed': 42,
            'enable_caching': True,
            'cache_dir': './granite_cache',
        },
        'training': {
            'epochs': 150,
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'enforce_constraints': True,
        },
        'model': {
            'hidden_dim': 64,
            'dropout': 0.3,
        },
    }
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config = dict(self.DEFAULTS)
        
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                file_config = yaml.safe_load(f) or {}
            config = self._deep_merge(config, file_config)
        else:
            raise FileNotFoundError(
                f"Configuration file not found at {self.config_path}. "
                f"Copy config.yaml.example to config.yaml and adjust settings."
            )
        
        return config
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = dict(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def apply_cli_overrides(self, args) -> None:
        """Apply command-line argument overrides."""
        if hasattr(args, 'fips') and args.fips:
            self._config['data']['target_fips'] = args.fips
            self._config['data']['state_fips'] = args.fips[:2]
            self._config['data']['county_fips'] = args.fips[2:5]
        
        if hasattr(args, 'epochs') and args.epochs:
            self._config['training']['epochs'] = args.epochs
        
        if hasattr(args, 'verbose') and args.verbose:
            self._config['processing']['verbose'] = True
        
        if hasattr(args, 'no_cache') and args.no_cache:
            self._config['processing']['enable_caching'] = False
        
        if hasattr(args, 'no_constraints') and args.no_constraints:
            self._config['training']['enforce_constraints'] = False
    
    def get(self, *keys, default=None):
        """Get nested configuration value."""
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def __getitem__(self, key):
        return self._config[key]
    
    def to_dict(self) -> Dict[str, Any]:
        """Return full configuration as dictionary."""
        return dict(self._config)


def load_config(config_path: Optional[str] = None, args=None) -> GRANITEConfig:
    """
    Load configuration with optional CLI overrides.
    
    Args:
        config_path: Path to YAML config file
        args: argparse namespace with CLI arguments
        
    Returns:
        GRANITEConfig instance
    """
    config = GRANITEConfig(config_path)
    if args:
        config.apply_cli_overrides(args)
    return config