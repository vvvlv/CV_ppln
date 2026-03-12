"""Model module."""

from .registry import register_model, get_model, list_models

# Object detection (MMDetection / CFINet)
from .architectures.cfinet import CFINetWrapper


def create_model(config: dict):
    """Create model from experiment configuration."""
    model_config = config['model']
    model_type = model_config['type']
    return get_model(model_type, model_config) 