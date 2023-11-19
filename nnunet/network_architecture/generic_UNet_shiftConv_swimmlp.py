#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from copy import deepcopy
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
import torch
import numpy as np
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.network_architecture.layers import DropPath, trunc_normal_
import torch.nn.functional
import torch.nn.functional as F
import copy
from nnunet.network_architecture.utils import ensure_tuple_rep, optional_import
rearrange, _ = optional_import("einops", name="rearrange")


class torch_shift(nn.Module):
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
            pad_list = list(0 for i in range((self.dim_num) * 2))
            pad_list[(self.dim-2) * 2] = pad
            pad_list[(self.dim-2) * 2 + 1] = pad

            x = F.pad(x, tuple(pad_list)[::-1], "constant", 0)
            xs = torch.chunk(x, self.shift_size, 1)
            x_shift = [torch.roll(x_c, shift, self.dim) for x_c, shift in zip(xs, range(-pad, pad + 1))]
            x_cat = torch.cat(x_shift, 1)
            x_cat = torch.narrow(x_cat, self.dim, pad, shape[self.dim])

        return x_cat


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
        x = x.view(B, window_size[0], D // window_size[0],  window_size[1], H // window_size[1], window_size[2], W // window_size[2],  C)
        windows = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous().view(-1, window_size[0], window_size[1], window_size[2], C)

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
        B = int(windows.shape[0] / (D * H * W / window_size[0] / window_size[1]/ window_size[2]))
        x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], -1)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous().view(B, D, H, W, -1)

    return x

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
        if self.input_resolution[0] <= window_size[0]:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size[0] = self.input_resolution[0]
        #assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

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
            self.spatial_mlp = nn.Conv1d(self.num_heads * self.window_size[0]*self.window_size[1]*self.window_size[2],
                                         self.num_heads * self.window_size[0]*self.window_size[1]*self.window_size[2],
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


        if len(self.input_resolution) == 3:
            b, d, h, w, c = x.shape
            #window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
            pad_l = pad_t = pad_d0 = 0
            pad_d1 = (self.window_size[0] - d % self.window_size[0]) % self.window_size[0]
            pad_b = (self.window_size[1] - h % self.window_size[1]) % self.window_size[1]
            pad_r = (self.window_size[2] - w % self.window_size[2]) % self.window_size[2]
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
            x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2],
                                           C)  # nW*B, window_size*window_size*window_size, C

            # Window/Shifted-Window Spatial MLP
            x_windows_heads = x_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], self.num_heads,
                                                 C // self.num_heads)
            x_windows_heads = x_windows_heads.transpose(1, 2)  # nW*B, nH, window_size*window_size*window_size, C//nH
            x_windows_heads = x_windows_heads.reshape(-1, self.num_heads * self.window_size[0] * self.window_size[1] * self.window_size[2],
                                                          C // self.num_heads)
            spatial_mlp_windows = self.spatial_mlp(x_windows_heads)  # nW*B, nH*window_size*window_size, C//nH
            spatial_mlp_windows = spatial_mlp_windows.view(-1, self.num_heads, self.window_size[0] * self.window_size[1] * self.window_size[2],
                                                               C // self.num_heads).transpose(1, 2)
            spatial_mlp_windows = spatial_mlp_windows.reshape(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], C)

            # merge windows
            spatial_mlp_windows = spatial_mlp_windows.reshape(-1, self.window_size[0], self.window_size[1], self.window_size[2], C)
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


class ConvDropoutNormNonlin(nn.Module):
    """
    fixes a bug in ConvDropoutNormNonlin where lrelu was used regardless of nonlin. Bad.
    """

    def __init__(self, input_channels, output_channels, input_resolution,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, shift_size=5):
        super(ConvDropoutNormNonlin, self).__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.shift_size = shift_size
        self.C = input_channels
        self.O = output_channels
        self.S = 1
        self.input_resolution = input_resolution

        self.shift_D = torch_shift(self.shift_size, 2, 3)
        self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)
        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs[
            'p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None
        self.instnorm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.lrelu = self.nonlin(**self.nonlin_kwargs)

    def forward(self, x):
        x = self.shift_D(x)
        x = self.conv(x)

        if self.dropout is not None:
            x = self.dropout(x)

        return self.lrelu(self.instnorm(x))


class ConvDropoutNonlinNorm(ConvDropoutNormNonlin):
    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.instnorm(self.lrelu(x))


class StackedConvLayers(nn.Module):
    def __init__(self, input_feature_channels, output_feature_channels, num_convs, input_resolution,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, first_stride=None, basic_block=ConvDropoutNormNonlin):
        '''
        stacks ConvDropoutNormLReLU layers. initial_stride will only be applied to first layer in the stack. The other parameters affect all layers
        :param input_feature_channels:
        :param output_feature_channels:
        :param num_convs:
        :param dilation:
        :param kernel_size:
        :param padding:
        :param dropout:
        :param initial_stride:
        :param conv_op:
        :param norm_op:
        :param dropout_op:
        :param inplace:
        :param neg_slope:
        :param norm_affine:
        :param conv_bias:
        '''
        self.input_channels = input_feature_channels
        self.output_channels = output_feature_channels

        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        if first_stride is not None:
            self.conv_kwargs_first_conv = deepcopy(conv_kwargs)
            self.conv_kwargs_first_conv['stride'] = first_stride
        else:
            self.conv_kwargs_first_conv = conv_kwargs

        super(StackedConvLayers, self).__init__()
        self.blocks = nn.Sequential(
            *([basic_block(input_feature_channels, output_feature_channels, input_resolution, self.conv_op,
                           self.conv_kwargs_first_conv,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs)] +
              [basic_block(output_feature_channels, output_feature_channels, input_resolution, self.conv_op,
                           self.conv_kwargs,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs) for _ in range(num_convs - 1)]))

    def forward(self, x):
        return self.blocks(x)


def print_module_training_status(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d) or isinstance(module, nn.Dropout3d) or \
            isinstance(module, nn.Dropout2d) or isinstance(module, nn.Dropout) or isinstance(module, nn.InstanceNorm3d) \
            or isinstance(module, nn.InstanceNorm2d) or isinstance(module, nn.InstanceNorm1d) \
            or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d) or isinstance(module,
                                                                                                      nn.BatchNorm1d):
        print(str(module), module.training)


class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                                         align_corners=self.align_corners)


class Generic_UNet_shiftConv(SegmentationNetwork):
    DEFAULT_BATCH_SIZE_3D = 2
    DEFAULT_PATCH_SIZE_3D = (64, 192, 160)
    SPACING_FACTOR_BETWEEN_STAGES = 2
    BASE_NUM_FEATURES_3D = 30
    MAX_NUMPOOL_3D = 999
    MAX_NUM_FILTERS_3D = 320

    DEFAULT_PATCH_SIZE_2D = (256, 256)
    BASE_NUM_FEATURES_2D = 30
    DEFAULT_BATCH_SIZE_2D = 50
    MAX_NUMPOOL_2D = 999
    MAX_FILTERS_2D = 480

    use_this_for_batch_size_computation_2D = 19739648
    use_this_for_batch_size_computation_3D = 520000000  # 505789440

    def __init__(self, img_size, input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False,
                 final_nonlin=softmax_helper, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, basic_block=ConvDropoutNormNonlin,
                 seg_output_use_bias=False):
        """
        basically more flexible than v1, architecture is the same

        Does this look complicated? Nah bro. Functionality > usability

        This does everything you need, including world peace.

        Questions? -> f.isensee@dkfz.de
        """
        super(Generic_UNet_shiftConv, self).__init__()
        self.convolutional_upsampling = convolutional_upsampling
        self.convolutional_pooling = convolutional_pooling
        self.upscale_logits = upscale_logits
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}

        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}

        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.weightInitializer = weightInitializer
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.num_classes = num_classes
        self.final_nonlin = final_nonlin
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision

        if conv_op == nn.Conv2d:
            upsample_mode = 'bilinear'
            pool_op = nn.MaxPool2d
            transpconv = nn.ConvTranspose2d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3)] * (num_pool + 1)
            input_resolution = [img_size//(2**d) for d in range(num_pool+ 1)]
        elif conv_op == nn.Conv3d:
            upsample_mode = 'trilinear'
            pool_op = nn.MaxPool3d
            transpconv = nn.ConvTranspose3d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2, 2)] * num_pool
            if conv_kernel_sizes is None or True:
                #conv_kernel_sizes = [(3, 3, 3)] * (num_pool + 1)
                conv_kernel_sizes = [(1, 3, 3)] * (num_pool + 1)
            input_resolution = [(img_size[0], img_size[1], img_size[2])]
            input_resolution += list((img_size[0] // np.prod(np.array(pool_op_kernel_sizes)[:d+1,0]), img_size[1] // np.prod(np.array(pool_op_kernel_sizes)[:d+1,1]), img_size[2] // np.prod(np.array(pool_op_kernel_sizes)[:d+1,2])) for d in range(num_pool))
            print(input_resolution)

        else:
            raise ValueError("unknown convolution dimensionality, conv op: %s" % str(conv_op))

        self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, 0, dtype=np.int64)
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes

        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])

        if max_num_features is None:
            if self.conv_op == nn.Conv3d:
                self.max_num_features = self.MAX_NUM_FILTERS_3D
            else:
                self.max_num_features = self.MAX_FILTERS_2D
        else:
            self.max_num_features = max_num_features

        self.conv_blocks_context = []
        self.conv_blocks_localization = []
        self.td = []
        self.tu = []
        self.seg_outputs = []
        self.conv_concat = []

        output_features = base_num_features
        input_features = input_channels

        self.fc3 = nn.ModuleList()

        num_heads = [1]
        num_heads += [2**(i+1) for i in range(num_pool)]
        dilation_ratio = [4 for i in range(num_pool+1) ]
        dilation_ratio[-1] = 2
        dilation_ratio[-2] = 1

        for d in range(num_pool):
            # determine the first stride
            if d != 0 and self.convolutional_pooling:
                first_stride = pool_op_kernel_sizes[d - 1]
            else:
                first_stride = None

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[d]
            # add convolutions
            self.conv_blocks_context.append(StackedConvLayers(input_features, output_features, num_conv_per_stage,input_resolution[d],
                                                              self.conv_op, self.conv_kwargs, self.norm_op,
                                                              self.norm_op_kwargs, self.dropout_op,
                                                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                              first_stride, basic_block=basic_block))
            if d > 0:
                window_size = [7, input_resolution[d][1]//dilation_ratio[d], input_resolution[d][2]//dilation_ratio[d]]
                self.fc3.append(SwinMLPBlock(input_features, input_resolution[d], num_heads[d], window_size))
                self.conv_concat.append(nn.Conv3d(input_features+output_features, output_features, 3, 1, 1))

            if not self.convolutional_pooling:
                self.td.append(pool_op(pool_op_kernel_sizes[d]))
            input_features = output_features
            output_features = int(np.round(output_features * feat_map_mul_on_downscale))

            output_features = min(output_features, self.max_num_features)

        # now the bottleneck.
        # determine the first stride
        if self.convolutional_pooling:
            first_stride = pool_op_kernel_sizes[-1]
        else:
            first_stride = None

        # the output of the last conv must match the number of features from the skip connection if we are not using
        # convolutional upsampling. If we use convolutional upsampling then the reduction in feature maps will be
        # done by the transposed conv
        if self.convolutional_upsampling:
            final_num_features = output_features
        else:
            final_num_features = self.conv_blocks_context[-1].output_channels

        self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[num_pool]
        self.conv_kwargs['padding'] = self.conv_pad_sizes[num_pool]
        self.conv_blocks_context.append(nn.Sequential(
            StackedConvLayers(input_features, output_features, num_conv_per_stage - 1, input_resolution[num_pool], self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, first_stride, basic_block=basic_block),
            StackedConvLayers(output_features, final_num_features, 1, input_resolution[num_pool], self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, basic_block=basic_block)))

        window_size = [7, input_resolution[num_pool][1] // dilation_ratio[-1], input_resolution[num_pool][2] // dilation_ratio[-1]]

        self.fc3.append(SwinMLPBlock(input_features, input_resolution[num_pool], num_heads[num_pool], window_size))

        self.conv_concat.append(nn.Conv3d(input_features + final_num_features, final_num_features, 3, 1, 1))

        # if we don't want to do dropout in the localization pathway then we set the dropout prob to zero here
        if not dropout_in_localization:
            old_dropout_p = self.dropout_op_kwargs['p']
            self.dropout_op_kwargs['p'] = 0.0

        # now lets build the localization pathway
        for u in range(num_pool):
            nfeatures_from_down = final_num_features
            nfeatures_from_skip = self.conv_blocks_context[
                -(2 + u)].output_channels  # self.conv_blocks_context[-1] is bottleneck, so start with -2
            n_features_after_tu_and_concat = nfeatures_from_skip * 2

            # the first conv reduces the number of features to match those of skip
            # the following convs work on that number of features
            # if not convolutional upsampling then the final conv reduces the num of features again
            if u != num_pool - 1 and not self.convolutional_upsampling:
                final_num_features = self.conv_blocks_context[-(3 + u)].output_channels
            else:
                final_num_features = nfeatures_from_skip

            if not self.convolutional_upsampling:
                self.tu.append(Upsample(scale_factor=pool_op_kernel_sizes[-(u + 1)], mode=upsample_mode))
            else:
                self.tu.append(transpconv(nfeatures_from_down, nfeatures_from_skip, pool_op_kernel_sizes[-(u + 1)],
                                          pool_op_kernel_sizes[-(u + 1)], bias=False))

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[- (u + 1)]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[- (u + 1)]
            self.conv_blocks_localization.append(nn.Sequential(
                StackedConvLayers(n_features_after_tu_and_concat,  nfeatures_from_skip, num_conv_per_stage - 1, input_resolution[- (u + 1)],
                                  self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op,
                                  self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs, basic_block=basic_block),
                StackedConvLayers(nfeatures_from_skip, final_num_features, 1, input_resolution[- (u + 1)], self.conv_op, self.conv_kwargs,
                                  self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                                  self.nonlin, self.nonlin_kwargs, basic_block=basic_block)
            ))

        for ds in range(len(self.conv_blocks_localization)):
            self.seg_outputs.append(conv_op(self.conv_blocks_localization[ds][-1].output_channels, num_classes,
                                            1, 1, 0, 1, 1, seg_output_use_bias))

        self.upscale_logits_ops = []
        cum_upsample = np.cumprod(np.vstack(pool_op_kernel_sizes), axis=0)[::-1]
        for usl in range(num_pool - 1):
            if self.upscale_logits:
                self.upscale_logits_ops.append(Upsample(scale_factor=tuple([int(i) for i in cum_upsample[usl + 1]]),
                                                        mode=upsample_mode))
            else:
                self.upscale_logits_ops.append(lambda x: x)

        if not dropout_in_localization:
            self.dropout_op_kwargs['p'] = old_dropout_p


            # self.fc3_bn.append(self.norm_op(self.S[n], **self.norm_op_kwargs))

        # register all modules properly
        self.conv_concat = nn.ModuleList(self.conv_concat)
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        self.fc3 = nn.ModuleList(self.fc3)
        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        self.td = nn.ModuleList(self.td)
        self.tu = nn.ModuleList(self.tu)
        self.seg_outputs = nn.ModuleList(self.seg_outputs)
        if self.upscale_logits:
            self.upscale_logits_ops = nn.ModuleList(
                self.upscale_logits_ops)  # lambda x:x is not a Module so we need to distinguish here

        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)
            # self.apply(print_module_training_status)

    def forward(self, x):
        skips = []
        seg_outputs = []
        d = []
        h = []
        w = []


        for n in range(len(self.conv_blocks_context) - 1):

            if n > 0:
                out = self.fc3[n-1](x)
                x = self.conv_blocks_context[n](x)
                x = torch.cat([out, x], 1)
                x = self.conv_concat[n-1](x)
            else:
                x = self.conv_blocks_context[n](x)

            skips.append(x)
            if not self.convolutional_pooling:
                x = self.td[n](x)

        out = self.fc3[-1](x)
        x = self.conv_blocks_context[-1](x)
        x = torch.cat([out, x], 1)
        x = self.conv_concat[-1](x)


        for u in range(len(self.tu)):
            x = self.tu[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        if self._deep_supervision and self.do_ds:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            return seg_outputs[-1]
