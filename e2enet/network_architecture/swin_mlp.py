# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from typing import Any, Callable, Optional, Sequence, Tuple, Union, cast
import numpy as np
from e2enet.network_architecture.blocks import UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from e2enet.network_architecture.layers import DropPath, trunc_normal_
from e2enet.network_architecture.neural_network import SegmentationNetwork

from collections.abc import Iterable
from e2enet.network_architecture.layers import Conv
from e2enet.network_architecture.utils import ensure_tuple_rep, optional_import
rearrange, _ = optional_import("einops", name="rearrange")


def issequenceiterable(obj: Any) -> bool:
    """
    Determine if the object is an iterable sequence and is not a string.
    """
    try:
        if hasattr(obj, "ndim") and obj.ndim == 0:
            return False  # a 0-d tensor is not iterable
    except Exception:
        return False
    return isinstance(obj, Iterable) and not isinstance(obj, (str, bytes))


def ensure_tuple_rep(tup: Any, dim: int) -> Tuple[Any, ...]:
    """
    Returns a copy of `tup` with `dim` values by either shortened or duplicated input.

    Raises:
        ValueError: When ``tup`` is a sequence and ``tup`` length is not ``dim``.

    Examples::

        >>> ensure_tuple_rep(1, 3)
        (1, 1, 1)
        >>> ensure_tuple_rep(None, 3)
        (None, None, None)
        >>> ensure_tuple_rep('test', 3)
        ('test', 'test', 'test')
        >>> ensure_tuple_rep([1, 2, 3], 3)
        (1, 2, 3)
        >>> ensure_tuple_rep(range(3), 3)
        (0, 1, 2)
        >>> ensure_tuple_rep([1, 2], 3)
        ValueError: Sequence must have length 3, got length 2.

    """
    if isinstance(tup, torch.Tensor):
        tup = tup.detach().cpu().numpy()
    if isinstance(tup, np.ndarray):
        tup = tup.tolist()
    if not issequenceiterable(tup):
        return (tup,) * dim
    if len(tup) == dim:
        return tuple(tup)

    raise ValueError(f"Sequence must have length {dim}, got {len(tup)}.")

class SwinUNETMLP(SegmentationNetwork):
    """
    Swin UNETR based on: "Hatamizadeh et al.,
    Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images
    <https://arxiv.org/abs/2201.01266>"
    """

    def __init__(
        self,
        img_size=None,
        in_channels=None,
        out_channels=None,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        feature_size=24,
        norm_name = "instance",
        drop_rate = 0.0,
        attn_drop_rate = 0.0,
        dropout_path_rate = 0.0,
        normalize = True,
        use_checkpoint = False,
        spatial_dims = 3,
        deep_supervision = False,
        pool_strides = None # [[2, 2, 2], [2, 2, 2], [2, 2, 2]]
    ):
        """
        Args:
            img_size: dimension of input image.
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            feature_size: dimension of network feature size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            norm_name: feature normalization type and arguments.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            dropout_path_rate: drop path rate.
            normalize: normalize output intermediate features in each stage.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: number of spatial dims.

        Examples::

            # for 3D single channel input with size (96,96,96), 4-channel output and feature size of 48.
            >>> net = SwinUNETMLP(img_size=(96,96,96), in_channels=1, out_channels=4, feature_size=48)

            # for 3D 4-channel input with size (128,128,128), 3-channel output and (2,4,2,2) layers in each stage.
            >>> net = SwinUNETMLP(img_size=(128,128,128), in_channels=4, out_channels=3, depths=(2,4,2,2))

            # for 2D single channel input with size (96,96), 2-channel output and gradient checkpointing.
            >>> net = SwinUNETMLP(img_size=(96,96), in_channels=3, out_channels=2, use_checkpoint=True, spatial_dims=2)

        """

        super().__init__()

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(2, spatial_dims)
        window_size = 7
        #window_size = ensure_tuple_rep(7, spatial_dims)
        # depths = (2 for i in range(len(pool_strides)-1))
        # num_heads = (3*(i+1) for i in range(len(pool_strides)-1))


        if not (spatial_dims == 2 or spatial_dims == 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        for m, p in zip(img_size, patch_size):
            for i in range(len(depths)):
                if m % np.power(p, i + 1) != 0:
                    raise ValueError("input image size (img_size) should be divisible by stage-wise image resolution.")

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        if feature_size % 12 != 0:
            raise ValueError("feature_size should be divisible by 12.")

        self.normalize = normalize
        self.do_ds = deep_supervision

        self.swinMLP = SwinMLP(
            img_size = img_size,
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            pool_strides=pool_strides,
        )

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        #
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder10 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.out1 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)
        self.out2 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)
        self.out3 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size * 2, out_channels=out_channels)
        self.out4 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size * 4, out_channels=out_channels)
        self.out5 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size * 8, out_channels=out_channels)

    def load_from(self, weights):

        with torch.no_grad():
            self.swinViT.patch_embed.proj.weight.copy_(weights["state_dict"]["module.patch_embed.proj.weight"])
            self.swinViT.patch_embed.proj.bias.copy_(weights["state_dict"]["module.patch_embed.proj.bias"])
            for bname, block in self.swinViT.layers1[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers1")
            self.swinViT.layers1[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers1.0.downsample.reduction.weight"]
            )
            self.swinViT.layers1[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers1.0.downsample.norm.weight"]
            )
            self.swinViT.layers1[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers1.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers2[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers2")
            self.swinViT.layers2[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers2.0.downsample.reduction.weight"]
            )
            self.swinViT.layers2[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers2.0.downsample.norm.weight"]
            )
            self.swinViT.layers2[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers2.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers3[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers3")
            self.swinViT.layers3[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers3.0.downsample.reduction.weight"]
            )
            self.swinViT.layers3[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers3.0.downsample.norm.weight"]
            )
            self.swinViT.layers3[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers3.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers4[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers4")
            self.swinViT.layers4[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers4.0.downsample.reduction.weight"]
            )
            self.swinViT.layers4[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers4.0.downsample.norm.weight"]
            )
            self.swinViT.layers4[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers4.0.downsample.norm.bias"]
            )


    def forward(self, x_in):
        """
        hidden_states_out = self.swinMLP(x_in, self.normalize)
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        #enc2 = self.encoder3(hidden_states_out[1])
        #enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[2])
        #dec3 = self.decoder5(dec4, hidden_states_out[1])
        #dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec4, hidden_states_out[1])
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        """
        hidden_states_out = self.swinMLP(x_in, self.normalize)
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])
        dec3 = self.decoder5(dec4, hidden_states_out[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)

        logits5 = self.out5(dec3)
        logits4 = self.out4(dec2)
        logits3 = self.out3(dec1)
        logits2 = self.out2(dec0)
        logits1 = self.out1(out)

        return [logits1, logits2, logits3, logits4, logits5]
        #return [logits1, logits2, logits3]




class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    x_shape = x.size()
    if len(x_shape) == 4:
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    if len(x_shape) == 5:
        B, D, H, W, C = x.shape
        x = x.view(B, D // window_size, window_size, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size, window_size, window_size, C)

    return windows


def window_reverse(windows, window_size, x_shape):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    if len(x_shape) == 4:
        _, H, W, _ = x_shape
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)

    if len(x_shape) == 5:
        _, D, H, W, _ = x_shape
        B = int(windows.shape[0] / (D * H * W / window_size / window_size / window_size))
        x = windows.view(B, D // window_size, H // window_size, W // window_size, window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)

    return x


class SwinMLPBlock(nn.Module):
    r""" Swin MLP Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        #self.padding = [self.window_size - self.shift_size, self.shift_size,
        #                self.window_size - self.shift_size, self.shift_size]  # P_l,P_r,P_t,P_b

        self.norm1 = norm_layer(dim)
        # use group convolution to implement multi-head MLP
        if len(self.input_resolution) == 2:
            self.spatial_mlp = nn.Conv1d(self.num_heads * self.window_size ** 2,
                                         self.num_heads * self.window_size ** 2,
                                         kernel_size=1,
                                         groups=self.num_heads)
        elif len(self.input_resolution) == 3:
            self.spatial_mlp = nn.Conv1d(self.num_heads * self.window_size ** 3,
                                         self.num_heads * self.window_size ** 3,
                                         kernel_size=1,
                                         groups=self.num_heads)


        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        if len(self.input_resolution) == 2:
            #H, W = self.input_resolution
            B, C, H, W = x.shape
            x = rearrange(x, "B C H W -> B H W C")
            #assert L == H * W, "input feature has wrong size"
            x = x.view(B, H*W, C)
            shortcut = x
            x = self.norm1(x)
            x = x.view(B, H, W, C)


        elif len(self.input_resolution) == 3:
            #D, H, W = self.input_resolution
            B, C, D, H, W = x.shape
            x = rearrange(x, "B C D H W -> B D H W C")
            #assert L == D * H * W, "input feature has wrong size"
            x = x.view(B, D * H * W, C)
            shortcut = x
            x = self.norm1(x)
            x = x.view(B, D, H, W, C)


        # # shift
        #x_shape = x.size()
        # if len(x_shape) == 5:
        #     x = rearrange(x, "n c d h w -> n d h w c")
        #     B, D, H, W, C = x.shape
        #     #shortcut = x
        # elif len(x_shape) == 4:
        #     x = rearrange(x, "n c h w -> n h w c")
        #     B, H, W, C = x.shape
        #     #shortcut = x

        if len(self.input_resolution) == 3:
            b, d, h, w, c = x.shape
            #window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
            pad_l = pad_t = pad_d0 = 0
            pad_d1 = (self.window_size - d % self.window_size) % self.window_size
            pad_b = (self.window_size - h % self.window_size) % self.window_size
            pad_r = (self.window_size - w % self.window_size) % self.window_size
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))


        elif len(self.input_resolution) == 2:
            b, h, w, c = x.shape
            #window_size, shift_size = get_window_size((h, w), self.window_size, self.shift_size)
            pad_l = pad_t = 0
            pad_b = (self.window_size - h % self.window_size) % self.window_size
            pad_r = (self.window_size - w % self.window_size) % self.window_size
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))


        if self.shift_size > 0:
            if len(self.input_resolution) == 2:
                P_l, P_r, P_t, P_b = [self.window_size - self.shift_size, self.shift_size,
                                      self.window_size - self.shift_size, self.shift_size]  # P_l,P_r,P_t,P_b
                shifted_x = F.pad(x, [0, 0, P_l, P_r, P_t, P_b], "constant", 0)
            if len(self.input_resolution) == 3:
                P_d1, P_d2, P_l, P_r, P_t, P_b = [self.window_size - self.shift_size, self.shift_size, self.window_size - self.shift_size, self.shift_size,
                                      self.window_size - self.shift_size, self.shift_size]  # P_l,P_r,P_t,P_b
                shifted_x = F.pad(x, [0, 0, P_d1, P_d2, P_l, P_r, P_t, P_b], "constant", 0)

        else:
            shifted_x = x

        if len(self.input_resolution) == 2:
            _, _H, _W, _ = shifted_x.shape

            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
            x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

            # Window/Shifted-Window Spatial MLP
            x_windows_heads = x_windows.view(-1, self.window_size * self.window_size, self.num_heads, C // self.num_heads)
            x_windows_heads = x_windows_heads.transpose(1, 2)  # nW*B, nH, window_size*window_size, C//nH
            x_windows_heads = x_windows_heads.reshape(-1, self.num_heads * self.window_size * self.window_size,
                                                  C // self.num_heads)
            spatial_mlp_windows = self.spatial_mlp(x_windows_heads)  # nW*B, nH*window_size*window_size, C//nH
            spatial_mlp_windows = spatial_mlp_windows.view(-1, self.num_heads, self.window_size * self.window_size,
                                                       C // self.num_heads).transpose(1, 2)
            spatial_mlp_windows = spatial_mlp_windows.reshape(-1, self.window_size * self.window_size, C)

            # merge windows
            #spatial_mlp_windows = spatial_mlp_windows.reshape(-1, self.window_size, self.window_size, C)
            shifted_x = window_reverse(spatial_mlp_windows, self.window_size, shifted_x.shape)  # B H' W' C

            # reverse shift
            if self.shift_size > 0:
                #P_l, P_r, P_t, P_b = self.padding
                x = shifted_x[:, P_t:-P_b, P_l:-P_r, :].contiguous()
            else:
                x = shifted_x

            x = x.view(B, H * W, C)

        if len(self.input_resolution) == 3:
            _, _D, _H, _W, _ = shifted_x.shape

            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, window_size, C
            x_windows = x_windows.view(-1, self.window_size * self.window_size * self.window_size,
                                           C)  # nW*B, window_size*window_size*window_size, C

            # Window/Shifted-Window Spatial MLP
            x_windows_heads = x_windows.view(-1, self.window_size * self.window_size * self.window_size, self.num_heads,
                                                 C // self.num_heads)
            x_windows_heads = x_windows_heads.transpose(1, 2)  # nW*B, nH, window_size*window_size*window_size, C//nH
            x_windows_heads = x_windows_heads.reshape(-1, self.num_heads * self.window_size * self.window_size * self.window_size,
                                                          C // self.num_heads)
            spatial_mlp_windows = self.spatial_mlp(x_windows_heads)  # nW*B, nH*window_size*window_size, C//nH
            spatial_mlp_windows = spatial_mlp_windows.view(-1, self.num_heads, self.window_size * self.window_size * self.window_size,
                                                               C // self.num_heads).transpose(1, 2)
            spatial_mlp_windows = spatial_mlp_windows.reshape(-1, self.window_size * self.window_size * self.window_size, C)

            # merge windows
            spatial_mlp_windows = spatial_mlp_windows.reshape(-1, self.window_size, self.window_size, self.window_size, C)
            shifted_x = window_reverse(spatial_mlp_windows, self.window_size, shifted_x.shape)  # B H' W' C

            # reverse shift
            if self.shift_size > 0:
                #P_l, P_r, P_t, P_b = self.padding
                x = shifted_x[:, P_d1:-P_d2, P_t:-P_b, P_l:-P_r, :].contiguous()
            else:
                x = shifted_x

            if len(self.input_resolution) == 3:
                if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
                    x = x[:, :D, :H, :W, :].contiguous()
            elif len(self.input_resolution) == 2:
                if pad_r > 0 or pad_b > 0:
                    x = x[:, :H, :W, :].contiguous()

            x = x.view(B, D * H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if len(self.input_resolution) == 2:
            x = x.view(B, H, W, C)
            x = rearrange(x, "B H W C -> B C H W")
        elif len(self.input_resolution) == 3:
            x = x.view(B, D, H, W, C)
            x = rearrange(x, "B D H W C -> B C D H W")

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W

        # Window/Shifted-Window Spatial MLP
        if self.shift_size > 0:
            nW = (H / self.window_size + 1) * (W / self.window_size + 1)
        else:
            nW = H * W / self.window_size / self.window_size
        flops += nW * self.dim * (self.window_size * self.window_size) * (self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


# class PatchMerging(nn.Module):
#     r""" Patch Merging Layer.
#
#     Args:
#         input_resolution (tuple[int]): Resolution of input feature.
#         dim (int): Number of input channels.
#         norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
#     """
#
#     def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.input_resolution = input_resolution
#         self.dim = dim
#         if len(self.input_resolution) == 3:
#             self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
#             self.norm = norm_layer(8 * dim)
#         elif len(self.input_resolution) == 2:
#             self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
#             self.norm = norm_layer(4 * dim)
#         #self.norm = norm_layer(4 * dim)
#
#     def forward(self, x):
#         """
#         x: B, H*W, C
#         """
#         if len(self.input_resolution) == 2:
#
#             H, W = self.input_resolution
#             B, L, C = x.shape
#             assert L == H * W, "input feature has wrong size"
#             assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
#
#             x = x.view(B, H, W, C)
#
#             x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
#             x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
#             x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
#             x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
#             x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
#             x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
#
#             x = self.norm(x)
#             x = self.reduction(x)
#         if len(self.input_resolution) == 3:
#             D, H, W = self.input_resolution
#             B, L, C = x.shape
#             assert L == D * H * W, "input feature has wrong size"
#             assert D % 2 == 0 and H % 2 == 0 and W % 2 == 0, f"x size ({D}*{H}*{W}) are not even."
#
#             x = x.view(B, D, H, W, C)
#
#             x0 = x[:, 0::2, 0::2, 0::2, :]
#             x1 = x[:, 1::2, 0::2, 0::2, :]
#             x2 = x[:, 0::2, 1::2, 0::2, :]
#             x3 = x[:, 0::2, 0::2, 1::2, :]
#             x4 = x[:, 1::2, 0::2, 1::2, :]
#             x5 = x[:, 0::2, 1::2, 0::2, :]
#             x6 = x[:, 0::2, 0::2, 1::2, :]
#             x7 = x[:, 1::2, 1::2, 1::2, :]
#             x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)
#             x = x.view(B, -1, 8 * C)  # B H/2*W/2 4*C
#
#             x = self.norm(x)
#             x = self.reduction(x)
#
#         return x

class PatchMerging(nn.Module):
    """
    Patch merging layer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm, pool_stride=None) -> None:
        """
        Args:
            dim: number of feature channels.
            norm_layer: normalization layer.
            spatial_dims: number of spatial dims.
        """

        super().__init__()
        self.dim = dim
        if len(input_resolution) == 3:
            self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(8 * dim)
        elif len(input_resolution) == 2:
            self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(4 * dim)
        self.pool_stride = pool_stride

    def forward(self, x):

        x_shape = x.size()
        if len(x_shape) == 5:
            b, c, d, h, w = x_shape
            x = rearrange(x, "b c d h w -> b d h w c")
            # pad_input = (d % self.pool_stride[0] == 1) or (h % self.pool_stride[1] == 1) or (w % self.pool_stride[2] == 1)
            pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1)
            if pad_input:
                x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
            x0 = x[:, 0::2, 0::2, 0::2, :]
            x1 = x[:, 1::2, 0::2, 0::2, :]
            x2 = x[:, 0::2, 1::2, 0::2, :]
            x3 = x[:, 0::2, 0::2, 1::2, :]
            x4 = x[:, 1::2, 0::2, 1::2, :]
            x5 = x[:, 1::2, 1::2, 0::2, :]
            x6 = x[:, 1::2, 0::2, 1::2, :]
            x7 = x[:, 1::2, 1::2, 1::2, :]
            x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)

        elif len(x_shape) == 4:
            b, h, w, c = x_shape
            pad_input = (h % 2 == 1) or (w % 2 == 1)
            if pad_input:
                x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
            x0 = x[:, 0::2, 0::2, :]
            x1 = x[:, 1::2, 0::2, :]
            x2 = x[:, 0::2, 1::2, :]
            x3 = x[:, 1::2, 1::2, :]
            x = torch.cat([x0, x1, x2, x3], -1)

        x = self.norm(x)
        x = self.reduction(x)
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin MLP layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        #self.pool_stride = pool_stride

        # build blocks
        self.blocks = nn.ModuleList([
            SwinMLPBlock(dim=dim, input_resolution=input_resolution,
                         num_heads=num_heads, window_size=window_size,
                         shift_size=0 if (i % 2 == 0) else window_size // 2,
                         mlp_ratio=mlp_ratio,
                         drop=drop,
                         drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                         norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        b, d, h, w, c = x.size()
        if len(self.input_resolution) == 3:
            x = rearrange(x, "b d h w c -> b c d h w")
        elif len(self.input_resolution) == 2:
            x = rearrange(x, "b h w c -> b c h w")
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=[56,56,56], patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, spatial_dims=3):
        super().__init__()
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        patches_resolution = [im_d // p_d for im_d, p_d in zip(img_size, patch_size)]
        self.img_size = img_size
        self.spatial_dims = spatial_dims
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = np.prod([im_d // p_d for im_d, p_d in zip(img_size, patch_size)])

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = Conv[Conv.CONV, spatial_dims](
            in_channels=in_chans, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        #B, C, H, W = x.shape
        x_shape = x.size()
        if len(x_shape) == 5:
            _, _, d, h, w = x_shape
            if w % self.patch_size[2] != 0:
                x = F.pad(x, (0, self.patch_size[2] - w % self.patch_size[2]))
            if h % self.patch_size[1] != 0:
                x = F.pad(x, (0, 0, 0, self.patch_size[1] - h % self.patch_size[1]))
            if d % self.patch_size[0] != 0:
                x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - d % self.patch_size[0]))
        elif len(x_shape) == 4:
            _, _, h, w = x_shape
            if w % self.patch_size[1] != 0:
                x = F.pad(x, (0, self.patch_size[1] - w % self.patch_size[1]))
            if h % self.patch_size[0] != 0:
                x = F.pad(x, (0, 0, 0, self.patch_size[0] - h % self.patch_size[0]))

        x = self.proj(x)
        if self.norm is not None:
            x_shape = x.size()
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)

            if len(x_shape) == 5:
                d, wh, ww = x_shape[2], x_shape[3], x_shape[4]
                x = x.transpose(1, 2).view(-1, self.embed_dim, d, wh, ww)
            elif len(x_shape) == 4:
                wh, ww = x_shape[2], x_shape[3]
                x = x.transpose(1, 2).view(-1, self.embed_dim, wh, ww)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class SwinMLP(nn.Module):
    r""" Swin MLP

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin MLP layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        drop_rate (float): Dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size: Union[Sequence[int], int], patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio


        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution


        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            if len(list(patches_resolution) == 2):
                H, W = patches_resolution
                self.absolute_pos_embed = nn.Parameter(torch.zeros(1, H, W, embed_dim))
            elif len(list(patches_resolution) == 3):
                D, H, W = patches_resolution
                self.absolute_pos_embed = nn.Parameter(torch.zeros(1, D, H, W, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()
        self.layers3 = nn.ModuleList()
        self.layers4 = nn.ModuleList()
        # self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=tuple((i // (2 ** i_layer) for i in self.patches_resolution)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               drop=drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging,
                               use_checkpoint=use_checkpoint)
            if i_layer == 0:
                self.layers1.append(layer)
            elif i_layer == 1:
                self.layers2.append(layer)
            elif i_layer == 2:
                self.layers3.append(layer)
            elif i_layer == 3:
                self.layers4.append(layer)
            # self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def proj_out(self, x, normalize=False):
        if normalize:
            x_shape = x.size()
            if len(x_shape) == 5:
                n, ch, d, h, w = x_shape
                x = rearrange(x, "n ch d h w -> n d h w ch")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n d h w ch -> n ch d h w")
            elif len(x_shape) == 4:
                n, ch, h, w = x_shape
                x = rearrange(x, "n ch h w -> n h w ch")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n h w ch -> n ch h w")
        return x

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    # def forward(self, x):
    #     x = self.forward_features(x)
    #     x = self.head(x)
    #     return x

    def forward(self, x, normalize=True):
        x0 = self.patch_embed(x)   # B, D, H, W, C
        if self.ape:
            x = x + self.absolute_pos_embed
        x0 = self.pos_drop(x0)
        x_list = []
        x_list.append(x0)
        x0_out = self.proj_out(x0, normalize)
        x_out = []
        x_out.append(x0_out)

        x1 = self.layers1[0](x0.contiguous())
        x1_out = self.proj_out(x1, normalize)
        x2 = self.layers2[0](x1.contiguous())
        x2_out = self.proj_out(x2, normalize)
        x3 = self.layers3[0](x2.contiguous())
        x3_out = self.proj_out(x3, normalize)
        x4 = self.layers4[0](x3.contiguous())
        x4_out = self.proj_out(x4, normalize)
        # for i in range(self.num_layers):
        #     x_tmp = self.layers[i](x_list[-1].contiguous())
        #     x_list.append(x_tmp)
        #     x_tmpout = self.proj_out(x_tmp, normalize)
        #     x_out.append(x_tmpout)
        # return x_out
        return [x0_out, x1_out, x2_out, x3_out, x4_out]

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
