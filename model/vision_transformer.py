import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torchvision._internally_replaced_utils import load_state_dict_from_url

from utils.utils import load_weights_from_state_dict

__all__ = [
    "VisionTransformer",
    "vit_b_16",
    "vit_b_32",
    "vit_l_16",
    "vit_l_32",
    "vit_h_14",
]


model_urls = {
    'vit_b_16': 'https://download.pytorch.org/models/vit_b_16-c867db91.pth',
    'vit_b_32': 'https://download.pytorch.org/models/vit_b_32-d86f8d99.pth',
    'vit_l_16': 'https://download.pytorch.org/models/vit_l_16-852ce7e3.pth',
    'vit_l_32': 'https://download.pytorch.org/models/vit_l_32-c7638314.pth',
    'vit_h_14': 'https://download.pytorch.org/models/vit_h_14_swag-80465313.pth',
}


class MLPBlock(nn.Module):
    """Transformer MLP block."""

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, mlp_dim)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(mlp_dim, in_dim)
        self.dropout2 = nn.Dropout(dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear_1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.linear_2(x)
        x = self.dropout2(x)
        return x


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, x: Tensor) -> Tensor:
        x_norm = self.ln_1(x)
        attn_out, _ = self.self_attention(x_norm, x_norm, x_norm, need_weights=False)
        x = x + self.dropout(attn_out)
        x = x + self.mlp(self.ln_2(x))
        return x


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pos_embedding
        return self.ln(self.layers(self.dropout(x)))



class VisionTransformer(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1000,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.norm_layer = norm_layer

        self.conv_proj = nn.Conv2d(
            in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
        )

        seq_length = (image_size // patch_size) ** 2

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length += 1

        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )
        self.seq_length = seq_length

        self.heads = nn.Sequential(
            nn.Identity(),  # placeholder for pre_logits if needed
        )
        self.heads.head = nn.Linear(hidden_dim, num_classes)

        # Initialize weights
        if isinstance(self.conv_proj, nn.Conv2d):
            fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
            nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)

        nn.init.zeros_(self.heads.head.weight)
        nn.init.zeros_(self.heads.head.bias)

    def _process_input(self, x: Tensor) -> Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)
        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        x = x.permute(0, 2, 1)

        return x

    def forward_features(self, x: Tensor, need_fea: bool = False):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        cls_token = x[:, 0]

        if need_fea:
            # Return patch tokens as features (excluding class token)
            patch_tokens = x[:, 1:]
            # Reshape to spatial format for compatibility
            h = w = int(math.sqrt(patch_tokens.shape[1]))
            features = patch_tokens.permute(0, 2, 1).reshape(n, self.hidden_dim, h, w)
            return [features], cls_token
        else:
            return cls_token

    def forward(self, x: Tensor, need_fea: bool = False) -> Tensor:
        if need_fea:
            features, features_fc = self.forward_features(x, need_fea)
            out = self.heads.head(features_fc)
            return features, features_fc, out
        else:
            x = self.forward_features(x)
            x = self.heads.head(x)
            return x

    def cam_layer(self):
        return self.encoder.layers[-1]


def _vision_transformer(
    arch: str,
    image_size: int,
    patch_size: int,
    num_layers: int,
    num_heads: int,
    hidden_dim: int,
    mlp_dim: int,
    pretrained: bool,
    progress: bool,
    **kwargs: Any,
) -> VisionTransformer:
    model = VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        **kwargs,
    )

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        load_weights_from_state_dict(model, state_dict)

    return model


def vit_b_16(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VisionTransformer:
    """
    Constructs a ViT-B/16 architecture from
    `"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" <https://arxiv.org/abs/2010.11929>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vision_transformer(
        arch='vit_b_16',
        image_size=224,
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        pretrained=pretrained,
        progress=progress,
        **kwargs,
    )


def vit_b_32(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VisionTransformer:
    """
    Constructs a ViT-B/32 architecture from
    `"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" <https://arxiv.org/abs/2010.11929>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vision_transformer(
        arch='vit_b_32',
        image_size=224,
        patch_size=32,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        pretrained=pretrained,
        progress=progress,
        **kwargs,
    )


def vit_l_16(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VisionTransformer:
    """
    Constructs a ViT-L/16 architecture from
    `"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" <https://arxiv.org/abs/2010.11929>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vision_transformer(
        arch='vit_l_16',
        image_size=224,
        patch_size=16,
        num_layers=24,
        num_heads=16,
        hidden_dim=1024,
        mlp_dim=4096,
        pretrained=pretrained,
        progress=progress,
        **kwargs,
    )


def vit_l_32(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VisionTransformer:
    """
    Constructs a ViT-L/32 architecture from
    `"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" <https://arxiv.org/abs/2010.11929>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vision_transformer(
        arch='vit_l_32',
        image_size=224,
        patch_size=32,
        num_layers=24,
        num_heads=16,
        hidden_dim=1024,
        mlp_dim=4096,
        pretrained=pretrained,
        progress=progress,
        **kwargs,
    )


def vit_h_14(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VisionTransformer:
    """
    Constructs a ViT-H/14 architecture from
    `"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" <https://arxiv.org/abs/2010.11929>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vision_transformer(
        arch='vit_h_14',
        image_size=518,
        patch_size=14,
        num_layers=32,
        num_heads=16,
        hidden_dim=1280,
        mlp_dim=5120,
        pretrained=pretrained,
        progress=progress,
        **kwargs,
    )


if __name__ == '__main__':
    inputs = torch.rand((1, 3, 224, 224))
    model = vit_b_16(pretrained=True)
    model.eval()
    out = model(inputs)
    print('out shape:{}'.format(out.size()))
    feas, fea_fc, out = model(inputs, True)
    for idx, fea in enumerate(feas):
        print('feature {} shape:{}'.format(idx + 1, fea.size()))
    print('fc shape:{}'.format(fea_fc.size()))
    print('out shape:{}'.format(out.size()))
