import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
#from einops import rearrange
from nnunet.network_architecture.utils import ensure_tuple_rep, optional_import
from nnunet.network_architecture.layers import Conv
rearrange, _ = optional_import("einops", name="rearrange")
from nnunet.network_architecture.layers import DropPath, trunc_normal_
import torch.nn.functional as F
import numpy as np
from typing import Any, Callable, Optional, Sequence, Tuple, Union, cast
from collections.abc import Iterable
from nnunet.network_architecture.neural_network import SegmentationNetwork
#from nnunet.network_architecture.utils.shift_cuda import Shift
from nnunet.network_architecture.blocks.dynunet_block import UnetBasicBlock, UnetResBlock, get_conv_layer

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


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x

def MyNorm(dim):
    return nn.GroupNorm(1, dim)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., dim_num=3):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        if dim_num == 2:
            self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1)
            self.act = act_layer()
            self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1)
            self.drop = nn.Dropout(drop)
        elif dim_num == 3:
            self.fc1 = nn.Conv3d(in_features, hidden_features, 1, 1)
            self.act = act_layer()
            self.fc2 = nn.Conv3d(hidden_features, out_features, 1, 1)
            self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MixShiftBlock(nn.Module):
    r""" Mix-Shifting Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        shift_size (int): Shift size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, shift_size, shift_dist, mix_size, layer_scale_init_value=1e-6,
                 mlp_ratio=4, drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.mlp_ratio = mlp_ratio

        self.shift_size = shift_size
        self.shift_dist = shift_dist
        self.chunk_size = [i.shape[0] for i in torch.chunk(torch.zeros(dim), self.shift_size)]

        self.kernel_size = [(ms, ms // 2) for ms in mix_size]

        if len(self.input_resolution) == 2:
            self.dwconv_lr = nn.ModuleList(
                [nn.Conv2d(chunk_dim, chunk_dim, kernel_size=kernel_size[0], padding=kernel_size[1], groups=chunk_dim) for
                 chunk_dim, kernel_size in zip(self.chunk_size, self.kernel_size)])
            self.dwconv_td = nn.ModuleList(
                [nn.Conv2d(chunk_dim, chunk_dim, kernel_size=kernel_size[0], padding=kernel_size[1], groups=chunk_dim) for
                 chunk_dim, kernel_size in zip(self.chunk_size, self.kernel_size)])

        if len(self.input_resolution) == 3:
            self.dwconv_lr = nn.ModuleList(
                [nn.Conv3d(chunk_dim, chunk_dim, kernel_size=kernel_size[0], padding=kernel_size[1], groups=1) for
                 chunk_dim, kernel_size in zip(self.chunk_size, self.kernel_size)])
            self.dwconv_td = nn.ModuleList(
                [nn.Conv3d(chunk_dim, chunk_dim, kernel_size=kernel_size[0], padding=kernel_size[1], groups=1) for
                 chunk_dim, kernel_size in zip(self.chunk_size, self.kernel_size)])
            self.dwconv_dd = nn.ModuleList(
                [nn.Conv3d(chunk_dim, chunk_dim, kernel_size=kernel_size[0], padding=kernel_size[1], groups=1)
                 for chunk_dim, kernel_size in zip(self.chunk_size, self.kernel_size)])

        self.norm = norm_layer(dim)
        self.pwconv1 = nn.Linear(dim, int(mlp_ratio * dim))  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(int(mlp_ratio * dim), dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x

        # split groups
        xs = torch.chunk(x, self.shift_size, 1)

        if len(self.input_resolution) == 2:
            B_, C, H, W = x.shape

            # shift with pre-defined relative distance
            x_shift_lr = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, self.shift_dist)]
            x_shift_td = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, self.shift_dist)]

            # regional mixing
            for i in range(self.shift_size):
                x_shift_lr[i] = self.dwconv_lr[i](x_shift_lr[i])
                x_shift_td[i] = self.dwconv_td[i](x_shift_td[i])

            x_lr = torch.cat(x_shift_lr, 1)
            x_td = torch.cat(x_shift_td, 1)

            x = x_lr + x_td
            x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
            x = self.norm(x)
            x = self.pwconv1(x)
            x = self.act(x)
            x = self.pwconv2(x)
            if self.gamma is not None:
                x = self.gamma * x
            x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        if len(self.input_resolution) == 3:
            B_, C, D, H, W = x.shape
            # shift with pre-defined relative distance
            x_shift_lr = [torch.roll(x_c, shift, 4) for x_c, shift in zip(xs, self.shift_dist)]
            x_shift_td = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, self.shift_dist)]
            x_shift_dd = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, self.shift_dist)]

            # regional mixing
            for i in range(self.shift_size):
                x_shift_lr[i] = self.dwconv_lr[i](x_shift_lr[i])
                x_shift_td[i] = self.dwconv_td[i](x_shift_td[i])
                x_shift_dd[i] = self.dwconv_dd[i](x_shift_dd[i])

            x_lr = torch.cat(x_shift_lr, 1)
            x_td = torch.cat(x_shift_td, 1)
            x_dd = torch.cat(x_shift_dd, 1)

            x = x_lr + x_td + x_dd
            x = self.norm(x)
            x = x.permute(0, 2, 3, 4, 1)  # (N, C, D, H, W) -> (N, D, H, W, C)
            x = self.pwconv1(x)
            x = self.act(x)
            x = self.pwconv2(x)
            if self.gamma is not None:
                x = self.gamma * x
            x = x.permute(0, 4, 1, 2, 3)  # (N, D, H, W, C) -> (N, C, D, H, W)

        x = input + self.drop_path(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, " \
               f"shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        N = H * W
        # dwconv_1 dwconv_2
        for i in range(self.shift_size):
            flops += 2 * (N * self.chunk_size[i] * self.kernel_size[i][0])
        # x_lr + x_td
        flops += N * self.dim
        # norm
        flops += self.dim * H * W
        # pwconv
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm, pool_strides=None):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        #if self.input_resolution == 3:
        #self.reduction = nn.Linear(np.prod(pool_strides) * dim, 2 * dim, bias=False)
        if len(self.input_resolution) == 2:
            self.reduction = nn.Conv2d(np.prod(pool_strides) * dim, 2 * dim, 1, 1, bias=False)
        elif len(self.input_resolution) == 3:
            self.reduction = nn.Conv3d(np.prod(pool_strides) * dim, 2 * dim, 1, 1, bias=False)
        self.norm = norm_layer(np.prod(pool_strides) * dim)
        # self.norm = LayerNorm(np.prod(pool_strides) * dim, eps=1e-6, data_format="channels_first")
        self.pool_strides = pool_strides

    def forward(self, x):
        """
        x: B, H*W, C
        """
        if len(self.input_resolution) == 2:
            #H, W = self.input_resolution
            B, H, W, C = x.shape
            #assert L == H * W, "input feature has wrong size"
            assert H % self.pool_strides[0] == 0 and W % self.pool_strides[1] == 0, f"x size ({H}*{W}) are not even."

            #x = x.view(B, H, W, C)

            x_concat = []

            for m in range(self.pool_strides[1]):
                for n in range(self.pool_strides[0]):
                    x_concat.append(x[:, :, n::self.pool_strides[0], m::self.pool_strides[1]])

            x = torch.cat(x_concat, 1)  # B H/2 W/2 4*C
            #x = x.view(B, -1, np.prod(self.pool_strides) * C)  # B H/2*W/2 4*C
            x = self.norm(x)
            x = self.reduction(x)

        if len(self.input_resolution) == 3:
            #D, H, W = self.input_resolution
            B, C, D, H, W = x.shape
            #assert L == D * H * W, "input feature has wrong size"
            assert D % self.pool_strides[0] == 0 and H % self.pool_strides[1] == 0 and W % self.pool_strides[2] == 0, f"x size ({D}*{H}*{W}) are not even."

            #x = x.view(B, D, H, W, C)

            x_concat = []
            for i in range(self.pool_strides[2]):
                for m in range(self.pool_strides[1]):
                    for n in range(self.pool_strides[0]):
                        x_concat.append(x[:, :, n::self.pool_strides[0], m::self.pool_strides[1], i::self.pool_strides[2]])

            x = torch.cat(x_concat, 1)
            # x = x.view(B, -1, np.prod(self.pool_strides) * C)  # B H/2*W/2 4*C

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


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm, pool_stride=None):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.pool_stride = pool_stride
        if len(self.input_resolution) == 2:
            self.expand = nn.Linear(dim, np.prod(np.array(self.pool_stride))/2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        if len(self.input_resolution) == 3:
            a = np.prod(np.array(self.pool_stride))/2
            self.expand = nn.Linear(dim, int(np.prod(np.array(self.pool_stride))/2) * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        if len(self.input_resolution) == 2:
            H, W = self.input_resolution
            x = self.expand(x)
            B, L, C = x.shape
            assert L == H * W, "input feature has wrong size"

            x = x.view(B, H, W, C)
            x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c',
                          p1=self.pool_stride[0], p2=self.pool_stride[1], c=C // np.prod(self.pool_stride))
            x = x.view(B, -1, C // np.prod(self.pool_stride))

        if len(self.input_resolution) == 3:
            D, H, W = self.input_resolution
            x = self.expand(x)

            B, L, C = x.shape
            assert L ==  D * H * W, "input feature has wrong size"

            x = x.view(B, D, H, W, C)
            x = rearrange(x, 'b d h w (p1 p2 p3 c)-> b (d p1) (h p2) (w p3) c',
                          p1=self.pool_stride[0], p2=self.pool_stride[1], p3=self.pool_stride[2], c=C // np.prod(self.pool_stride))
            x = x.view(B, -1, C // np.prod(self.pool_stride))
            #print(x)
        x = self.norm(x)

        return x



class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, shift_dist, mix_size, shift_size=3, as_bias=True,
                 mlp_ratio=4., drop=0., drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False, pool_strides=None):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            # AxialShiftedBlock(dim=dim, input_resolution=input_resolution,
            #                   shift_size=shift_size,
            #                   mlp_ratio=mlp_ratio,
            #                   as_bias=as_bias,
            #                   drop=drop,
            #                   drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
            #                   norm_layer=norm_layer)
            MixShiftBlock(dim=dim, input_resolution=input_resolution,
                          shift_size=shift_size,
                          shift_dist=shift_dist,
                          mix_size=mix_size,
                          mlp_ratio=mlp_ratio,
                          drop=drop,
                          drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                          norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer, pool_strides=pool_strides)
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




class BasicLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, shift_dist, mix_size, shift_size=3, as_bias=True,
                 mlp_ratio=4., drop=0.,drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False, pool_stride=None, last_layer=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            # AxialShiftedBlock(dim=dim, input_resolution=input_resolution,
            #                   shift_size=shift_size,
            #                   mlp_ratio=mlp_ratio,
            #                   as_bias=as_bias,
            #                   drop=drop,
            #                   drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
            #                   norm_layer=norm_layer)
            MixShiftBlock(dim=dim, input_resolution=input_resolution,
                          shift_size=shift_size,
                          shift_dist=shift_dist,
                          mix_size=mix_size,
                          mlp_ratio=mlp_ratio,
                          drop=drop,
                          drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                          norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            #self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer, pool_stride=pool_stride)
            if last_layer:
                self.upsample = get_conv_layer(len(input_resolution), dim, dim, kernel_size=3,
                                           stride=2, conv_only=True, is_transposed=True,)
            else:
                self.upsample = get_conv_layer(len(input_resolution), dim, dim // 2, kernel_size=3,
                                               stride=2, conv_only=True, is_transposed=True, )


        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x_upsample = self.upsample(x)
        else:
            x_upsample = x
        return x, x_upsample


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, spatial_dims=3):
        super().__init__()
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        patches_resolution = list([im_d // p_d for im_d, p_d in zip(img_size, patch_size)])
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
        x_shape = x.size()
        if len(x_shape) == 4:
            _, _, h, w = x_shape
            # FIXME look at relaxing size constraints
            assert h == self.img_size[0] and w == self.img_size[1], \
                f"Input image size ({h}*{w}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
            if w % self.patch_size[1] != 0:
                x = F.pad(x, (0, self.patch_size[1] - w % self.patch_size[1]))
            if h % self.patch_size[0] != 0:
                x = F.pad(x, (0, 0, 0, self.patch_size[0] - h % self.patch_size[0]))

        elif len(x_shape) == 5:
            _, _, d, h, w = x_shape
            # FIXME look at relaxing size constraints
            assert d == self.img_size[0] and h == self.img_size[1] and w == self.img_size[2],\
                f"Input image size ({d}*{h}*{w}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}*{self.img_size[2]})."
            if w % self.patch_size[2] != 0:
                x = F.pad(x, (0, self.patch_size[2] - w % self.patch_size[2]))
            if h % self.patch_size[1] != 0:
                x = F.pad(x, (0, 0, 0, self.patch_size[1] - h % self.patch_size[1]))
            if d % self.patch_size[0] != 0:
                x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - d % self.patch_size[0]))

        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        # if self.norm is not None:
        #     x_shape = x.size()
        #     x = x.flatten(2).transpose(1, 2)
        #     x = self.norm(x)
        #
        #     if len(x_shape) == 5:
        #         d, wh, ww = x_shape[2], x_shape[3], x_shape[4]
        #         x = x.transpose(1, 2).view(-1, self.embed_dim, d, wh, ww)
        #     elif len(x_shape) == 4:
        #         wh, ww = x_shape[2], x_shape[3]
        #         x = x.transpose(1, 2).view(-1, self.embed_dim, wh, ww)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class as_mlp_unet(SegmentationNetwork):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=2, in_chans=3, num_classes=1000, shift_size=1, as_bias=True,
                 embed_dim=96, depths=None, shift_dist=[-1,0,1],
                 mix_size=[[1,1,1], [1,1,1], [1,1,1], [1,1,1]], depths_decoder=None, mlp_ratio=4., drop_rate=0., drop_path_rate=0.1,
                 norm_layer=MyNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", spatial_dims=3, deep_supervision = False, pool_strides=None, **kwargs):
        super().__init__()



        self.num_classes = num_classes
        self.num_layers = len(pool_strides)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.final_upsample = final_upsample
        self.pool_strides = pool_strides
        self.spatial_dims = len(pool_strides[0])

        self.do_ds = deep_supervision
        self.in_chans = in_chans

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=self.in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None, spatial_dims=self.spatial_dims)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        print(depths)
        if depths == None:
            depths = [2 for i in range(len(pool_strides))]
            depths[-2] = 2
            depths = tuple(depths)
            #depths = (2,6,2)
        if depths_decoder == None:
            depths_decoder = [2 for i in range(len(pool_strides))]
            depths_decoder[1] = 2
            depths_decoder = tuple(depths_decoder)
            #depths_decoder = (2,6,2)


        print(
            "SwinMLPrSys expand initial----depths:{};depths_decoder:{};drop_path_rate:{};num_classes:{}".format(
                depths,
                depths_decoder, drop_path_rate, num_classes))

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build encoder and bottleneck layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            if i_layer == 0:
                layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                   #input_resolution=(i // (2 ** i_layer) for i in self.patches_resolution),
                                   input_resolution=self.patches_resolution,
                                   depth=depths[i_layer],
                                   shift_size=shift_size,
                                   shift_dist=shift_dist,
                                   mix_size=mix_size[i_layer],
                                   mlp_ratio=self.mlp_ratio,
                                   as_bias=as_bias,
                                   drop=drop_rate,
                                   drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                   norm_layer=norm_layer,
                                   downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                   use_checkpoint=use_checkpoint,
                                   pool_strides=pool_strides[i_layer+1])
            else:

                layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                   # input_resolution=(i // (2 ** i_layer) for i in self.patches_resolution),
                                   input_resolution=list(self.patches_resolution[i] // np.prod(np.array(pool_strides)[1:i_layer+1, i]) for i in range(len(self.patches_resolution))),
                                   depth=depths[i_layer],
                                   shift_size=shift_size,
                                   shift_dist=shift_dist,
                                   mix_size=mix_size[i_layer],
                                   mlp_ratio=self.mlp_ratio,
                                   as_bias=as_bias,
                                   drop=drop_rate,
                                   drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                   norm_layer=norm_layer,
                                   downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                   use_checkpoint=use_checkpoint,
                                   pool_strides=pool_strides[i_layer+1] if (i_layer < self.num_layers - 1) else None)
            self.layers.append(layer)

        # build decoder layers
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            #concat_linear = nn.Linear(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                      #int(embed_dim * 2 ** (self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()
            concat_linear = get_conv_layer(len(patches_resolution), 2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), kernel_size = 1,
                                           stride = 1, ) if i_layer > 0 else nn.Identity()
            if i_layer == 0:
                layer_up = get_conv_layer(len(patches_resolution), int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), int(embed_dim * 2 ** (self.num_layers - 2 - i_layer)), kernel_size=3,
                                          stride=pool_strides[(self.num_layers - 1)], conv_only=True, is_transposed=True,)

            else:

                layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                         input_resolution=list(
                                         patches_resolution[i] // np.prod(np.array(self.pool_strides)[1:-i_layer, i]) for i in range(len(patches_resolution))),
                                         depth=depths[i_layer],
                                         shift_size=shift_size,
                                         shift_dist=shift_dist,
                                         mix_size=mix_size[i_layer],
                                         mlp_ratio=self.mlp_ratio,
                                         as_bias=as_bias,
                                         drop=drop_rate,
                                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                         norm_layer=norm_layer,
                                         upsample=PatchExpand, #if (i_layer < self.num_layers - 1) else None,
                                         use_checkpoint=use_checkpoint, pool_stride=pool_strides[(self.num_layers - 1 - i_layer)], last_layer = (i_layer==self.num_layers-1))

            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm = norm_layer(self.num_features)
        # self.norm = LayerNorm(self.num_features, eps=1e-6, data_format="channels_first")

        self.norm_up = nn.ModuleList()
        #self.norm_up.append(norm_layer(self.embed_dim))
        for inx, i_layer in enumerate(self.layers_up):
            if inx == 0:
                norm_up = norm_layer(self.embed_dim)
                # norm_up = LayerNorm(self.embed_dim, eps=1e-6, data_format="channels_first")

            else:
                norm_up = norm_layer(self.embed_dim * 2**(inx-1))
                # norm_up = LayerNorm(self.embed_dim * 2**(inx-1), eps=1e-6, data_format="channels_first")

            self.norm_up.append(norm_up)
        #self.norm_up.append(norm_layer(self.embed_dim * 2**(inx+1)))

        self.output_layer = nn.Conv3d(in_channels=embed_dim, out_channels=self.num_classes, kernel_size=1, bias=False)
        self.output_layer_2 = nn.Conv3d(in_channels=embed_dim, out_channels=self.num_classes, kernel_size=1, bias=False)
        self.output_layer_4 = nn.Conv3d(in_channels=2*embed_dim, out_channels=self.num_classes, kernel_size=1,
                                        bias=False)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
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

    # Encoder and Bottleneck
    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x_downsample = []

        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)

        x = self.norm(x)  # B L C
        return x, x_downsample


    # Dencoder and Skip connection
    def forward_up_features(self, x, x_downsample):
        #out_put = [self.norm_up[-1](x), ]
        x_upsample = []
        # a = len(self.layers_up)
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                upsample = layer_up(x)
                x_upsample.append(self.norm_up[-1](upsample))
            else:
                #upsample = torch.cat([upsample, x_downsample[len(x_downsample) - 1 - inx]], -1)
                upsample =  torch.cat((upsample, x_downsample[len(x_downsample) - 1 - inx]), dim=1)
                upsample = self.concat_back_dim[inx](upsample)
                x, upsample = layer_up(upsample)
                #out_put.append(self.norm_up[-inx - 1](x))

                x_upsample.append(self.norm_up[-inx -1](upsample))

        return x_upsample

    def forward(self, x):
        x, x_downsample = self.forward_features(x)
        x_upsample = self.forward_up_features(x, x_downsample)
        a = self.output_layer(x_upsample[-1])
        b = self.output_layer_2(x_upsample[-2])
        c = self.output_layer_4(x_upsample[-3])

        return [a, b, c]

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (
                    2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops



