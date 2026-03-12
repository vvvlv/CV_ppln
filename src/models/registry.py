"""Model registration system."""

from typing import Dict, Callable
import torch.nn as nn

# Global registry
_MODEL_REGISTRY: Dict[str, Callable] = {}


def register_model(name: str):
    """Decorator to register a model class."""
    def decorator(cls):
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_model(name: str, config: dict) -> nn.Module:
    """Get model instance from registry."""
    if name not in _MODEL_REGISTRY:
        available = list(_MODEL_REGISTRY.keys())
        raise ValueError(f"Model '{name}' not found. Available: {available}")
    
    model_class = _MODEL_REGISTRY[name]
    # Create model instance - pass config dict directly
    # Model classes should extract their own parameters from config
    return model_class(**config)


def list_models() -> list:
    """List all registered model names."""
    return list(_MODEL_REGISTRY.keys()) 