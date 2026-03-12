"""Configuration loading utilities."""

from pathlib import Path
import yaml
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load experiment configuration.
    
    Args:
        config_path: Path to experiment config file
    
    Returns:
        Complete configuration dictionary
    """
    config_path = Path(config_path).resolve()
    
    # Find project root (directory containing 'configs' folder)
    # Start from config file and walk up until we find 'configs' directory
    project_root = config_path.parent
    while project_root != project_root.parent:  # Stop at filesystem root
        if (project_root / 'configs').exists():
            break
        project_root = project_root.parent
    else:
        # Fallback: use config file's parent if we can't find project root
        project_root = config_path.parent
    
    # Load experiment config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load referenced dataset config (resolve relative to project root)
    if 'dataset' in config and isinstance(config['dataset'], str):
        dataset_path = Path(config['dataset'])
        if not dataset_path.is_absolute():
            # Resolve relative to project root (where 'configs' directory is)
            dataset_path = (project_root / dataset_path).resolve()
        with open(dataset_path, 'r') as f:
            dataset_config = yaml.safe_load(f)
        config['dataset'] = dataset_config
    
    return config 