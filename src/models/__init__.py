"""Model module."""

from .registry import register_model, get_model, list_models
from .architectures.unet import UNet
from .architectures.unet_no_bottleneck import UNetNoBottleneck
from .architectures.unet_deep import UNetDeep
from .architectures.unet_shallow import UNetShallow
from .architectures.unet_heavy_bottleneck import UNetHeavyBottleneck
from .architectures.unet_limited_skip import UNetLimitedSkip
from .architectures.unet_simple_encoder import UNetSimpleEncoder
from .architectures.unet_simple_bottleneck import UNetSimpleBottleneck
from .architectures.unet_simple_decoder import UNetSimpleDecoder
from .architectures.unet_simple_encoder_decoder import UNetSimpleEncoderDecoder
from .architectures.unet_two_down import UNetTwoDown
from .architectures.unet_no_skip import UNetNoSkip
from .architectures.unet_kernel5 import UNetKernel5
from .architectures.unet_kernel7 import UNetKernel7
from .architectures.roinet import RoiNet
from .architectures.transunet import TransUNet
from .architectures.transroinet import TransRoiNet
from .architectures.tiny_swin import TinySwinUNet
from .architectures.att_unet_aspp import ASPPAttUNet
from .architectures.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSysWrapper
from .architectures.cfinet import CFINetWrapper


def create_model(config: dict):
    """Create model from experiment configuration."""
    model_config = config['model']
    model_type = model_config['type']
    return get_model(model_type, model_config) 