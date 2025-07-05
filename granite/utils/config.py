"""
Configuration management for GRANITE framework
"""
import os
import yaml
import json
from typing import Dict, Any


def load_config(config_path='config/config.yaml'):
    """
    Load configuration from YAML file
    
    Parameters:
    -----------
    config_path : str
        Path to configuration file
        
    Returns:
    --------
    dict
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        # Try looking in package directory
        package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(package_dir, config_path)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Process special values
    config = _process_config_values(config)
    
    return config


def save_config(config, output_path):
    """
    Save configuration to file
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
    output_path : str
        Output file path (.yaml or .json)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if output_path.endswith('.json'):
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
    else:
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def merge_configs(base_config, override_config):
    """
    Recursively merge configuration dictionaries
    
    Parameters:
    -----------
    base_config : dict
        Base configuration
    override_config : dict
        Override configuration
        
    Returns:
    --------
    dict
        Merged configuration
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def _process_config_values(config):
    """
    Process special configuration values
    
    Parameters:
    -----------
    config : dict
        Raw configuration
        
    Returns:
    --------
    dict
        Processed configuration
    """
    if isinstance(config, dict):
        processed = {}
        for key, value in config.items():
            if isinstance(value, str):
                # Environment variable substitution
                if value.startswith('${') and value.endswith('}'):
                    env_var = value[2:-1]
                    processed[key] = os.environ.get(env_var, value)
                else:
                    processed[key] = value
            else:
                processed[key] = _process_config_values(value)
        return processed
    elif isinstance(config, list):
        return [_process_config_values(item) for item in config]
    else:
        return config


def create_experiment_config(base_config_path, experiment_name, **overrides):
    """
    Create experiment-specific configuration
    
    Parameters:
    -----------
    base_config_path : str
        Path to base configuration
    experiment_name : str
        Name of experiment
    **overrides : dict
        Configuration overrides
        
    Returns:
    --------
    dict
        Experiment configuration
    """
    # Load base config
    base_config = load_config(base_config_path)
    
    # Create experiment config
    exp_config = {
        'experiment': {
            'name': experiment_name,
            'timestamp': pd.Timestamp.now().isoformat()
        }
    }
    
    # Apply overrides
    for key, value in overrides.items():
        keys = key.split('.')
        current = exp_config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    # Merge with base
    final_config = merge_configs(base_config, exp_config)
    
    return final_config


class ConfigManager:
    """Manager class for configuration handling"""
    
    def __init__(self, config_path='config/config.yaml'):
        self.config_path = config_path
        self.config = load_config(config_path)
        self._original_config = self.config.copy()
    
    def get(self, key_path, default=None):
        """
        Get configuration value using dot notation
        
        Parameters:
        -----------
        key_path : str
            Dot-separated key path (e.g., 'model.epochs')
        default : Any
            Default value if key not found
            
        Returns:
        --------
        Any
            Configuration value
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path, value):
        """
        Set configuration value using dot notation
        
        Parameters:
        -----------
        key_path : str
            Dot-separated key path
        value : Any
            Value to set
        """
        keys = key_path.split('.')
        current = self.config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def update(self, updates):
        """
        Update multiple configuration values
        
        Parameters:
        -----------
        updates : dict
            Dictionary of updates
        """
        self.config = merge_configs(self.config, updates)
    
    def reset(self):
        """Reset to original configuration"""
        self.config = self._original_config.copy()
    
    def save(self, output_path=None):
        """Save current configuration"""
        if output_path is None:
            output_path = self.config_path
        
        save_config(self.config, output_path)
    
    def to_dict(self):
        """Get configuration as dictionary"""
        return self.config.copy()
    
    def __getitem__(self, key):
        """Dictionary-style access"""
        return self.config[key]
    
    def __setitem__(self, key, value):
        """Dictionary-style setting"""
        self.config[key] = value