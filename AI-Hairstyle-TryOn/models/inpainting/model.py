"""
LaMa-style FFC ResNet generator (big-lama architecture) for image inpainting.
Input: concat(masked RGB, mask) = 4 channels. Output: 3-channel prediction in [0,1] with sigmoid.
"""

import torch.nn as nn

from .lama.ffc import FFCResNetGenerator


def build_big_lama_generator() -> nn.Module:
    """
    Build generator matching LaMa 'big-lama' training config (WACV 2022).
    Pretrained weights must be loaded separately (see inference.load_lama_state_dict).
    """
    return FFCResNetGenerator(
        input_nc=4,
        output_nc=3,
        ngf=64,
        n_downsampling=3,
        n_blocks=18,
        norm_layer=nn.BatchNorm2d,
        padding_type="reflect",
        activation_layer=nn.ReLU,
        up_norm_layer=nn.BatchNorm2d,
        up_activation=nn.ReLU(True),
        init_conv_kwargs=dict(ratio_gin=0, ratio_gout=0, enable_lfu=False),
        downsample_conv_kwargs=dict(ratio_gin=0, ratio_gout=0, enable_lfu=False),
        resnet_conv_kwargs=dict(ratio_gin=0.75, ratio_gout=0.75, enable_lfu=False),
        add_out_act="sigmoid",
        max_features=1024,
    )
