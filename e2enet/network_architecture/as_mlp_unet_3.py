import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
#from einops import rearrange
from e2enet.network_architecture.utils import ensure_tuple_rep, optional_import
from e2enet.network_architecture.layers import Conv
rearrange, _ = optional_import("einops", name="rearrange")
from e2enet.network_architecture.layers import DropPath, trunc_normal_
import torch.nn.functional as F
import numpy as np
from typing import Any, Callable, Optional, Sequence, Tuple, Union, cast
from collections.abc import Iterable
from e2enet.network_architecture.neural_network import SegmentationNetwork
#from e2enet.network_architecture.utils.shift_cuda import Shift
from e2enet.network_architecture.blocks.dynunet_block import UnetBasicBlock, UnetResBlock, get_conv_layer

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

class torch_shift(nn.Module):
    def __init__(self, shift_size, dim, dim_num, pixel):
        super().__init__()
        self.shift_size = shift_size
        self.dim = dim
        self.dim_num = dim_num
        self.pixel = pixel

    def forward(self, x):
        if self.dim_num == 2:
            B_, C, H, W = x.shape

            pad = self.shift_size // 2
            x = F.pad(x, (pad, pad, pad, pad), "constant", 0)
            xs = torch.chunk(x, self.shift_size, 1)
            x_shift = [torch.roll(x_c, shift, self.dim) for x_c, shift in zip(xs, range(-pad, pad + 1))]
            x_cat = torch.cat(x_shift, 1)
            x_cat = torch.narrow(x_cat, 2, pad, H)
            x_cat = torch.narrow(x_cat, 3, pad, W)

        if self.dim_num == 3:
            B_, C, D, H, W = x.shape
            shape = x.shape
            pad_1 = (self.pixel - shape[self.dim] % self.pixel) % self.pixel
            pad_list_1 = list(0 for i in range((self.dim_num) * 2))
            pad_list_1[(self.dim - 2) * 2] = pad_1
            x = F.pad(x, tuple(pad_list_1)[::-1], mode='circular')
            if self.dim == 2:
                x = x.reshape(B_, C, (D+pad_1)//self.pixel, self.pixel, H, W).permute(0, 1, 3, 2, 4, 5).reshape(B_, C*self.pixel, (D+pad_1)//self.pixel, H, W)
            if self.dim == 3:
                x = x.reshape(B_, C, D, (H+pad_1)//self.pixel, self.pixel, W).permute(0, 1, 4, 2, 3, 5).reshape(B_, C*self.pixel, D, (H+pad_1)//self.pixel, W)
            if self.dim == 4:
                x = x.reshape(B_, C, D, H, (W+pad_1)//self.pixel, self.pixel).permute(0, 1, 5, 2, 3, 4).reshape(B_, C*self.pixel, D, H, (W+pad_1)//self.pixel)


            pad = self.shift_size // 2
            # pad_list = list(0 for i in range((self.dim_num) * 2))
            # pad_list[(self.dim-2) * 2] = pad
            # pad_list[(self.dim-2) * 2 + 1] = pad

            x = F.pad(x, (0,0,0,0,pad,pad), "constant", 0)


            x_shift = []
            xs_1 = torch.chunk(x, self.pixel, 1)
            for inx, xs_i in enumerate(xs_1):
                xs = torch.chunk(xs_i, self.shift_size, 1)
                if inx == 0:
                    x_shift=[torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-pad, pad + 1))]
                else:
                    shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-pad, pad + 1))]
                    x_shift = x_shift + shift

            x_cat = torch.cat(x_shift, 1)
            x_cat = torch.narrow(x_cat, 2, pad, D)

        return x_cat, pad_1

class shift(nn.Module):
    def __init__(self, shift_size, dim, dim_num):
        super().__init__()
        self.shift_size = shift_size
        self.dim = dim
        self.dim_num = dim_num

    def forward(self, x):
        if self.dim_num == 2:
            B_, C, H, W = x.shape
            pad = self.shift_size // 2
            x = F.pad(x, (pad, pad, pad, pad), "constant", 0)
            xs = torch.chunk(x, self.shift_size, 1)
            x_shift = [torch.roll(x_c, shift, self.dim) for x_c, shift in zip(xs, range(-pad, pad + 1))]
            x_cat = torch.cat(x_shift, 1)
            x_cat = torch.narrow(x_cat, 2, pad, H)
            x_cat = torch.narrow(x_cat, 3, pad, W)

        if self.dim_num == 3:
            B_, C, D, H, W = x.shape
            shape = x.shape
            pad = self.shift_size // 2
            x = F.pad(x, (pad, pad, pad, pad, pad, pad), "constant", 0)
            xs = torch.chunk(x, self.shift_size, 1)
            x_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-pad, pad + 1))]
            x_cat = torch.cat(x_shift, 1)


            x_cat = torch.narrow(x_cat, self.dim, pad, shape[self.dim])

        return x_cat


class AxialShift(nn.Module):
    r""" Axial shift

    Args:
        dim (int): Number of input channels.
        shift_size (int): shift size .
        as_bias (bool, optional):  If True, add a learnable bias to as mlp. Default: True
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, shift_size, as_bias=True, proj_drop=0., num_dim=3, pixel=2):
        super().__init__()
        self.dim = dim
        self.shift_size = shift_size
        self.pad = shift_size // 2
        self.num_dim = num_dim
        self.pixel = pixel
        if self.num_dim == 2:
            self.conv1 = nn.Conv2d(dim, dim, 1, 1, 0, groups=1, bias=as_bias)
            self.conv2_1 = nn.Conv2d(dim, dim, 1, 1, 0, groups=1, bias=as_bias)
            self.conv2_2 = nn.Conv2d(dim, dim, 1, 1, 0, groups=1, bias=as_bias)
            self.conv3 = nn.Conv2d(dim, dim, 1, 1, 0, groups=1, bias=as_bias)

            self.actn = nn.GELU()

            self.norm1 = MyNorm(dim)
            self.norm2 = MyNorm(dim)

            self.shift_dim2 = torch_shift(self.shift_size, 2, 2)
            self.shift_dim3 = torch_shift(self.shift_size, 3, 2)
        if self.num_dim == 3:
            self.conv1 = nn.Conv3d(dim, dim, 1, 1, 0, groups=1, bias=as_bias)
            # self.conv2_1 = nn.Conv3d(dim, dim, (3,3,1), 1, (1,1,0), groups=1, bias=as_bias)
            # self.conv2_2 = nn.Conv3d(dim, dim, (3,1,3), 1, (1,0,1), groups=1, bias=as_bias)
            # self.conv2_3 = nn.Conv3d(dim, dim, (1,3,3), 1, (0,1,1), groups=1, bias=as_bias)
            self.conv2_11 = nn.Conv3d(dim*pixel, dim//2, (1, 1, 1), 1, 0, groups=1, bias=as_bias)
            self.conv2_12 = nn.Conv3d(dim//2, dim * pixel, (1, 1, 1), 1, 0, groups=1, bias=as_bias)

            self.conv2_21 = nn.Conv3d(dim*pixel, dim//2, (1, 1, 1), 1, 0, groups=1, bias=as_bias)
            self.conv2_22 = nn.Conv3d(dim//2, dim * pixel, (1, 1, 1), 1, 0, groups=1, bias=as_bias)

            self.conv2_31 = nn.Conv3d(dim*pixel, dim//2, (1, 1, 1), 1, 0, groups=1, bias=as_bias)
            self.conv2_32 = nn.Conv3d(dim//2, dim * pixel, (1, 1, 1), 1, 0, groups=1, bias=as_bias)
            self.conv3 = nn.Conv3d(dim, dim, 1, 1, 0, groups=1, bias=as_bias)

            self.mlp_c = nn.Conv3d(dim, dim, 1, bias=as_bias)

            self.actn = nn.GELU()

            self.norm1 = MyNorm(dim)
            self.norm2_1 = MyNorm(dim//2)
            self.norm2_2 = MyNorm(dim // 2)
            self.norm3 = MyNorm(dim)

            #self.shift_dim2 = torch_shift(self.shift_size, 2, 3, 2)
            self.shift_dim3 = torch_shift(self.shift_size, 3, 3, 2)
            self.shift_dim4 = torch_shift(self.shift_size, 4, 3, 2)

            # self.proj_drop = nn.Dropout(0.0)
            # self.reweight = Mlp(dim, dim // 4, dim * 3)


    def forward(self, x):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        if len(x.shape) == 4:
            B_, C, H, W = x.shape

            x = self.conv1(x)
            x = self.norm1(x)
            x = self.actn(x)


            x_shift_td = self.shift_dim3(x)
            x_shift_lr = self.shift_dim4(x)

            x_lr = self.conv2_1(x_shift_lr)
            x_td = self.conv2_2(x_shift_td)

            x_lr = self.actn(x_lr)
            x_td = self.actn(x_td)

            x = x_lr + x_td
            x = self.norm2(x)

            x = self.conv3(x)
        if len(x.shape) == 5:
            B_, C, D, H, W = x.shape

            # x = self.conv1(x)
            # x = self.norm1(x)
            # x = self.actn(x)

            # x_shift_lr, pad_lr = self.shift_dim4(x)
            # x_shift_td, pad_td  = self.shift_dim3(x)
            x_shift_lr, pad_lr = self.shift_dim4(x)
            x_shift_tb, pad_tb = self.shift_dim3(x)

            # x_lr = self.conv2_1(x_shift_lr)
            # x_td = self.conv2_2(x_shift_td)
            x_lr = self.conv2_11(x_shift_lr)
            x_lr = self.actn(x_lr)
            x_lr = self.norm2_1(x_lr)
            x_lr = self.conv2_12(x_lr)

            x_tb = self.conv2_21(x_shift_tb)
            x_tb = self.actn(x_tb)
            x_tb = self.norm2_1(x_tb)
            x_tb = self.conv2_22(x_tb)

            x_lr = x_lr.reshape(B_, C, self.pixel, D, H, (W + pad_lr) // self.pixel).permute(0, 1, 3, 4, 5, 2).reshape(B_, C, D, H, (W +pad_lr))

            x_tb = x_tb.reshape(B_, C, self.pixel, D, (H + pad_tb) // self.pixel, W ).permute(0, 1, 3, 4, 2, 5).reshape(B_, C, D, (H+ pad_tb), W)
            x_lr = torch.narrow(x_lr, 4, 0, W)
            x_tb = torch.narrow(x_tb, 3, 0, H)

            # c = self.mlp_c(x)

            x = x_lr + x_tb
            x = self.norm3(x)


            # a = (x_lr + x_tb + c).flatten(2).mean(2).unsqueeze(2).unsqueeze(2).unsqueeze(2)
            # a = self.reweight(a).reshape(B_, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(3).unsqueeze(3).unsqueeze(3)
            #
            # out = x_lr * a[0] + x_tb * a[1] + c * a[2]

            out = self.conv3(x)
            # out = self.proj_drop(out)

        return out

    def extra_repr(self) -> str:
        return f'dim={self.dim}, shift_size={self.shift_size}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # conv1
        flops += N * self.dim * self.dim
        # norm 1
        flops += N * self.dim
        # conv2_1 conv2_2
        flops += N * self.dim * self.dim * 2
        # x_lr + x_td
        flops += N * self.dim
        # norm2
        flops += N * self.dim
        # norm3
        flops += N * self.dim * self.dim
        return flops


class AxialShiftedBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        shift_size (int): Shift size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        as_bias (bool, optional): If True, add a learnable bias to Axial Mlp. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, shift_size=7,
                 mlp_ratio=4., as_bias=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.axial_shift = AxialShift(dim, shift_size=shift_size, as_bias=as_bias, proj_drop=drop, num_dim=len(input_resolution))

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, dim_num=len(self.input_resolution))

    def forward(self, x):
        if len(self.input_resolution) == 2:
            B, C, H, W = x.shape

            shortcut = x
            x = self.norm1(x)

            # axial shift block
            x = self.axial_shift(x)  # B, C, H, W

            # FFN
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        if len(self.input_resolution) == 3:
            B, C, D, H, W = x.shape

            shortcut = x
            x = self.norm1(x)

            # axial shift block
            x = self.axial_shift(x)  # B, C, H, W

            # FFN
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, " \
               f"shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # shift mlp
        flops += self.axial_shift.flops(H * W)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


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
        # elif self.input_resolution == 2:
        #     self.reduction = nn.Linear(sum(pool_stride) * dim, 2 * dim, bias=False)
        #     self.norm = norm_layer(sum(pool_stride) * dim)
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
            #x = x.view(B, -1, np.prod(self.pool_strides) * C)  # B H/2*W/2 4*C

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


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm, embed_dim=96):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        if len(self.input_resolution) == 2:
            self.expand = nn.Linear(dim, np.prod(np.array(dim_scale)) * embed_dim, bias=False)
        elif len(self.input_resolution) == 3:
            self.expand = nn.Linear(dim, np.prod(np.array(dim_scale)) * embed_dim, bias=False)

        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        if len(self.input_resolution)==2:
            H, W = self.input_resolution
            x = self.expand(x)
            B, L, C = x.shape
            assert L == H * W, "input feature has wrong size"
            x = x.view(B, H, W, C)
            x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale[0], p2=self.dim_scale[1],
                      c=C // np.prod(np.array(self.dim_scale)))
            x = x.view(B, -1, self.output_dim)

        elif len(self.input_resolution)==3:
            D, H, W = self.input_resolution
            x = self.expand(x)
            B, L, C = x.shape
            assert L == D * H * W, "input feature has wrong size"

            x = x.view(B, D, H, W, C)
            x = rearrange(x, 'b d h w (p1 p2 p3 c)-> b (d p1) (h p2) (w p3) c', p1=self.dim_scale[0], p2=self.dim_scale[1], p3=self.dim_scale[2],
                          c=C // np.prod(np.array(self.dim_scale)))
            x = x.view(B, -1, self.output_dim)
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

    def __init__(self, dim, input_resolution, depth, shift_size=3, as_bias=True,
                 mlp_ratio=4., drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False, pool_strides=None):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            # SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
            #                      num_heads=num_heads, window_size=window_size,
            #                      shift_size=0 if (i % 2 == 0) else window_size // 2,
            #                      mlp_ratio=mlp_ratio,
            #                      qkv_bias=qkv_bias, qk_scale=qk_scale,
            #                      drop=drop, attn_drop=attn_drop,
            #                      drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
            #                      norm_layer=norm_layer)
            AxialShiftedBlock(dim=dim, input_resolution=input_resolution,
                              shift_size=shift_size,
                              mlp_ratio=mlp_ratio,
                              as_bias=as_bias,
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

    def __init__(self, dim, input_resolution, depth, shift_size=3, as_bias=True,
                 mlp_ratio=4., drop=0.,drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False, pool_stride=None, last_layer=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            # SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
            #                      num_heads=num_heads, window_size=window_size,
            #                      shift_size=0 if (i % 2 == 0) else window_size // 2,
            #                      mlp_ratio=mlp_ratio,
            #                      qkv_bias=qkv_bias, qk_scale=qk_scale,
            #                      drop=drop, attn_drop=attn_drop,
            #                      drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
            #                      norm_layer=norm_layer)
            AxialShiftedBlock(dim=dim, input_resolution=input_resolution,
                              shift_size=shift_size,
                              mlp_ratio=mlp_ratio,
                              as_bias=as_bias,
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
        use_conv_embed (bool): Wherther use overlapped convolutional embedding layer. Default: False.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
        use_pre_norm (bool): Whether use pre-normalization before projection. Default: False
        is_stem (bool): Whether current patch embedding is stem. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, spatial_dims=3,
                 is_stem=False):
        super().__init__()
        if is_stem:
            kernel_size = 7
            padding = 2
            stride = 4
        else:
            kernel_size = 3
            padding = 1
            stride = 2
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        size_0 = (img_size[0] - kernel_size + padding + stride) / stride
        size_1 = (img_size[1] - kernel_size + padding + stride) / stride
        size_2 = (img_size[2] - kernel_size + padding + stride) / stride

        patches_resolution = [size_0, size_1, size_2]
        self.img_size = img_size
        self.spatial_dims = spatial_dims
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1] * patches_resolution[2]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.kernel_size = kernel_size
        self.proj = Conv[Conv.CONV, spatial_dims](in_chans, embed_dim, kernel_size=kernel_size, stride=stride,
                                                  padding=padding)

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

        elif len(x_shape) == 5:
            _, _, d, h, w = x_shape
            # FIXME look at relaxing size constraints
            assert d == self.img_size[0] and h == self.img_size[1] and w == self.img_size[2], \
                f"Input image size ({d}*{h}*{w}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}*{self.img_size[2]})."

        x = self.proj(x)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

class PatchEmbed_1(nn.Module):
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

        x = self.proj(x)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
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

    def __init__(self, img_size=224, patch_size=2, in_chans=3, num_classes=1000, shift_size=3, as_bias=True,
                 embed_dim=96, depths=None, depths_decoder=None, mlp_ratio=4., drop_rate=0., drop_path_rate=0.1,
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
                # if len(patches_resolution) == 2:
                #     layer_up = PatchExpand(
                #         # input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                #         #                   patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                #         input_resolution=(patches_resolution[0] // np.prod(np.array(self.pool_strides)[1:,0]),
                #                           patches_resolution[1] // np.prod(np.array(self.pool_strides)[1:,1]),
                #                           ),
                #         dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), dim_scale=2, norm_layer=norm_layer, pool_stride=pool_strides[-1])
                #
                # elif len(patches_resolution) == 3:
                #     layer_up = PatchExpand(input_resolution=(patches_resolution[0] // np.prod(np.array(self.pool_strides)[1:, 0]),
                #                           patches_resolution[1] // np.prod(np.array(self.pool_strides)[1:, 1]),
                #                           patches_resolution[2] // np.prod(np.array(self.pool_strides)[1:, 2]),
                #                           ),
                #         dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), dim_scale=2, norm_layer=norm_layer, pool_stride=pool_strides[-1])
                layer_up = get_conv_layer(len(patches_resolution), int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), int(embed_dim * 2 ** (self.num_layers - 2 - i_layer)), kernel_size=3,
                                          stride=pool_strides[(self.num_layers - 1)], conv_only=True, is_transposed=True,)

            else:

                layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                         input_resolution=list(
                                         patches_resolution[i] // np.prod(np.array(self.pool_strides)[1:-i_layer, i]) for i in range(len(patches_resolution))),
                                         depth=depths[i_layer],
                                         shift_size=shift_size,
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

        self.norm_up = nn.ModuleList()
        #self.norm_up.append(norm_layer(self.embed_dim))
        for inx, i_layer in enumerate(self.layers_up):
            if inx == 0:
                norm_up = norm_layer(self.embed_dim)
            else:
                norm_up = norm_layer(self.embed_dim * 2**(inx-1))
            self.norm_up.append(norm_up)
        #self.norm_up.append(norm_layer(self.embed_dim * 2**(inx+1)))

        self.layers_up_output = nn.ModuleList()
        for i_layer in range(3):
            if i_layer == 0:
                if self.final_upsample == "expand_first":
                    print("---final upsample expand_first---")
                    if len(patches_resolution) == 2:
                        up_layer = FinalPatchExpand_X4(input_resolution=[img_size[0] // patch_size, img_size[1] // patch_size],
                                                      dim_scale=[patch_size, patch_size] , dim=embed_dim, embed_dim=embed_dim )
                        #self.output_layer = nn.Conv2d(in_channels=embed_dim, out_channels=self.num_classes, kernel_size=1, bias=False)
                    if len(patches_resolution) == 3:
                        up_layer = FinalPatchExpand_X4(input_resolution=[img_size[0] // patch_size, img_size[1] // patch_size, img_size[2] // patch_size],
                                                      dim_scale=[patch_size, patch_size, patch_size], dim=embed_dim, embed_dim=embed_dim)
                        #self.output_layer = nn.Conv3d(in_channels=embed_dim, out_channels=self.num_classes, kernel_size=1, bias=False)
            else:
                if self.final_upsample == "expand_first":
                    print("---final upsample expand_first---")
                    if len(patches_resolution) == 2:
                        up_layer = FinalPatchExpand_X4(input_resolution=[img_size[i] // np.prod(np.array(pool_strides)[0:i_layer+1, i]) for i in range(2)], dim=embed_dim*2**(i_layer),
                                                       dim_scale=[np.prod(np.array(pool_strides)[i_layer,i]) for i in range(2)],  embed_dim=embed_dim)

                    if len(patches_resolution) == 3:

                        up_layer = FinalPatchExpand_X4(input_resolution=[img_size[i]//np.prod(np.array(pool_strides)[0:i_layer+1, i]) for i in range(3)], dim=embed_dim*2**(i_layer),
                                                      dim_scale=[np.prod(np.array(pool_strides)[i_layer,i]) for i in range(3)], embed_dim=embed_dim)


            self.layers_up_output.append(up_layer)
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

        #x = self.norm_up[0](x)  # B L C

        return x_upsample

    def forward(self, x):
        x, x_downsample = self.forward_features(x)
        x_upsample = self.forward_up_features(x, x_downsample)
        a = self.output_layer(x_upsample[-1])
        b = self.output_layer_2(x_upsample[-2])
        c = self.output_layer_4(x_upsample[-3])
        # print("a:",a.shape)
        # print("b:", b.shape)
        # print("c:", c.shape)



        return [self.output_layer(x_upsample[-1]), self.output_layer_2(x_upsample[-2]), self.output_layer_4(x_upsample[-3])]
        # return [x]

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (
                    2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops



