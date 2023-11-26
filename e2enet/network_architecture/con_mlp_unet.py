import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from copy import deepcopy
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

class ConvDropoutNormNonlin(nn.Module):
    """
    fixes a bug in ConvDropoutNormNonlin where lrelu was used regardless of nonlin. Bad.
    """

    def __init__(self, input_channels, output_channels,
                 conv_op=nn.Conv3d, conv_kwargs=None,
                 norm_op=nn.InstanceNorm3d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout3d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, first_stride=None, shift_size=3):
        super(ConvDropoutNormNonlin, self).__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.0, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        if conv_kwargs is None:
            conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}
            conv_kwargs['kernel_size'] = [1, 3, 3]
            conv_kwargs['padding'] = [0, 1, 1]

        if first_stride is not None:
            conv_kwargs['stride'] = first_stride

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.shift_size = shift_size

        self.shift_D = torch_shift(self.shift_size, 2, 3)
        self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)
        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs['p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None

        self.instnorm = self.norm_op(output_channels, **self.norm_op_kwargs)
        if self.nonlin is not None:
            self.lrelu = self.nonlin(**self.nonlin_kwargs)
        else:
            self.lrelu = None

    def forward(self, x):
        x = self.shift_D(x)
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.instnorm(x)
        if self.lrelu is not None:
            x = self.lrelu(x)
        return x


class Downsample(nn.Module):
    def __init__(self, pool_op_kernel_size):
        super().__init__()
        self.downsample = nn.MaxPool3d(pool_op_kernel_size)

    def forward(self, x):
        return self.downsample(x)

class AxialShiftedBlock(nn.Module):

    def __init__(self, dim, input_resolution, in_chann, out_chann, stride, S_num, parts):
        super().__init__()
        self.dim = dim
        self.input_res = input_resolution
        self.in_chann = in_chann
        self.out_chann = out_chann
        self.stride = stride
        self.parts = parts # d, h, w
        self.do_mlp = False

        self.S_n = S_num
        self.fc3_in = self.input_res[0]//self.parts[0] * self.input_res[1]//self.parts[1] * self.input_res[2]//self.parts[2] * self.S_n
        self.fc3 = nn.Conv3d(self.fc3_in, self.fc3_in, 1, 1, 0, bias=False, groups=self.S_n)
        self.fc3_bn = nn.BatchNorm3d(self.S_n)

        conv_kwargs = {'kernel_size': 1, 'stride': 1, 'padding': 0, 'dilation': 1, 'bias': True}
        self.conv_concat = ConvDropoutNormNonlin(self.in_chann+self.out_chann, self.out_chann, conv_kwargs=conv_kwargs)
        self.conv = ConvDropoutNormNonlin(self.in_chann, self.out_chann, first_stride=self.stride)

    def partition(self, x, c, d_parts, h_parts, w_parts, d, h, w):
        x = x.reshape(-1, c, d, d_parts, h, h_parts, w, w_parts)
        x = x.permute(0, 3, 5, 7, 1, 2, 4, 6)
        return x

    def partition_affine(self, x, S, d_parts, h_parts, w_parts, d, h, w):
        fc_inputs = x.reshape(-1, S * d * h * w, 1, 1, 1)
        out = self.fc3(fc_inputs)
        out = out.reshape(-1, S, d, h, w)
        out = self.fc3_bn(out)
        out = out.reshape(-1, d_parts, h_parts, w_parts, S, d, h, w)
        return out

    def forward(self, x):
        if self.do_mlp:
            B, C, D, H, W = x.shape
            d_parts = self.parts[0]
            h_parts = self.parts[1]
            w_parts = self.parts[2]

            d = D // d_parts  # d=1
            h = H // h_parts
            w = W // w_parts

            partitions = self.partition(x, C, d_parts, h_parts, w_parts, d, h, w)

            #  Channel Perceptron
            fc3_out = self.partition_affine(partitions, self.S_n, d_parts, h_parts, w_parts, d, h, w)
            fc3_out = fc3_out.permute(0, 4, 1, 5, 2, 6, 3, 7)  # N, O, h_parts, out_h, w_parts, out_w
            mlp_out = fc3_out.reshape(B, C, D, H, W)

            con_out = self.conv(x)

            out = torch.cat([mlp_out, con_out], 1)
            out = self.conv_concat(out)
            return out
        else:
            con_out = self.conv(x)
            return con_out

class BasicLayer(nn.Module):

    def __init__(self, dim, input_resolution, depth, input_features, output_features, downsample, pool_sizes, S_num, parts):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.input_features = input_features
        self.output_features = output_features

        # build blocks
        self.blocks = nn.ModuleList([
            AxialShiftedBlock(dim=self.dim, input_resolution=self.input_resolution, in_chann=self.input_features,
                              out_chann=self.output_features, stride=pool_sizes, S_num=S_num, parts=parts)] +
            [AxialShiftedBlock(dim=self.dim, input_resolution=self.input_resolution, in_chann=self.output_features,
                               out_chann=self.output_features, stride=pool_sizes, S_num=S_num, parts=parts) for _ in range(self.depth - 1)]
            )

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

class PatchEmbed(nn.Module):

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, first_stride=None, spatial_dims=3):
        super().__init__()
        kernel_size = 3
        padding = 1

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        if first_stride:
            size_0 = (img_size[0] - kernel_size + padding + first_stride[0]) / first_stride[0]
            size_1 = (img_size[1] - kernel_size + padding + first_stride[1]) / first_stride[1]
            size_2 = (img_size[2] - kernel_size + padding + first_stride[2]) / first_stride[2]
        else:
            size_0 = img_size[0]
            size_1 = img_size[1]
            size_2 = img_size[2]

        patches_resolution = [size_0, size_1, size_2]
        self.img_size = img_size
        self.spatial_dims = spatial_dims
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1] * patches_resolution[2]
        self.stride = first_stride

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = ConvDropoutNormNonlin(self.in_chans, self.embed_dim, dropout_op=None, nonlin=None, first_stride=self.stride)

    def forward(self, x):
        x = self.proj(x)  # B C D H W
        return x

class conv_mlp_unet(SegmentationNetwork):

    def __init__(self, img_size=224, patch_size=2, in_chans=3, num_classes=1000, shift_size=3, as_bias=True,
                 embed_dim=48, depths=2, depths_decoder=None, mlp_ratio=4., drop_rate=0., drop_path_rate=0.1,
                 norm_layer=None, ape=False, patch_norm=True, final_upsample="expand_first", deep_supervision = True,
                 pool_strides=None, **kwargs):
        super().__init__()

        self.conv_op = nn.Conv3d

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
        self.max_num_features = 128
        self.depths = depths

        self.do_ds = deep_supervision
        self.in_chans = in_chans

        # absolute position embedding

        ## input_resolution
        input_resolution = [(img_size[0], img_size[1], img_size[2])]
        input_resolution += list((img_size[0] // np.prod(np.array(self.pool_strides)[:d + 1, 0]),
                                  img_size[1] // np.prod(np.array(self.pool_strides)[:d + 1, 1]),
                                  img_size[2] // np.prod(np.array(self.pool_strides)[:d + 1, 2])) for d in
                                 range(self.num_layers))
        self.S = [2**i for i in range(self.num_layers+1)]
        print(input_resolution)

        # build encoder and bottleneck layers
        output_features = embed_dim
        input_features = in_chans
        self.conv_kernel_sizes = [(3, 3, 3)] * (self.num_layers + 1)
        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])

        self.conv_blocks_localization = nn.ModuleList()
        self.conv_blocks_context = nn.ModuleList()
        self.seg_outputs = nn.ModuleList()
        self.td = nn.ModuleList()
        self.tu = nn.ModuleList()

        for d in range(self.num_layers):
            # determine the first stride
            first_stride = None

            # add convolutions
            self.patches_resolution = input_resolution[d]
            self.S_num = self.S[d]
            self.parts = [self.patches_resolution[0], 2, 2]
            self.conv_blocks_context.append(BasicLayer(dim=self.embed_dim, input_resolution=self.patches_resolution,
                                                       depth=self.depths, input_features=input_features, output_features=output_features,
                                                       downsample=None, pool_sizes=first_stride, S_num=self.S_num, parts=self.parts))
            self.td.append(Downsample(self.pool_strides[d]))
            input_features = output_features
            output_features = int(np.round(output_features * 2))
            output_features = min(output_features, self.max_num_features)

        # now the bottleneck.
        # determine the first stride
        first_stride = None

        # the output of the last conv must match the number of features from the skip connection if we are not using
        # convolutional upsampling. If we use convolutional upsampling then the reduction in feature maps will be
        # done by the transposed conv
        final_num_features = self.conv_blocks_context[-1].output_features

        self.patches_resolution = input_resolution[-1]
        self.parts = [1, 1, 1]
        self.conv_blocks_context.append(BasicLayer(dim=self.embed_dim, input_resolution=self.patches_resolution,
                                                   depth=self.depths, input_features=input_features, output_features=final_num_features,
                                                   downsample=None, pool_sizes=first_stride, S_num=self.S[-1], parts=self.parts))

        # build decoder layers
        # now lets build the localization pathway
        for u in range(self.num_layers):
            nfeatures_from_down = final_num_features
            nfeatures_from_skip = self.conv_blocks_context[-(2 + u)].output_features  # self.conv_blocks_context[-1] is bottleneck, so start with -2

            n_features_after_tu_and_concat = nfeatures_from_skip * 2
            self.patches_resolution = input_resolution[-(u + 2)]
            self.S_num = self.S[-(u + 2)]
            self.parts = [self.patches_resolution[0], 2, 2]

            # the first conv reduces the number of features to match those of skip
            # the following convs work on that number of features
            # if not convolutional upsampling then the final conv reduces the num of features again
            final_num_features = nfeatures_from_skip
            self.tu.append(nn.ConvTranspose3d(nfeatures_from_down, nfeatures_from_skip, self.pool_strides[-(u + 1)],
                           self.pool_strides[-(u + 1)], bias=False))

            self.conv_blocks_localization.append(nn.Sequential(
                BasicLayer(dim=self.embed_dim, input_resolution=self.patches_resolution,
                           depth=self.depths-1, input_features=n_features_after_tu_and_concat, output_features=nfeatures_from_skip,
                           downsample=None, pool_sizes=first_stride, S_num=self.S_num, parts=self.parts),

                BasicLayer(dim=self.embed_dim, input_resolution=self.patches_resolution,
                           depth=1, input_features=nfeatures_from_skip, output_features=final_num_features,
                           downsample=None, pool_sizes=first_stride, S_num=self.S_num, parts=self.parts)
            ))

        for ds in range(len(self.conv_blocks_localization)):
            self.seg_outputs.append(nn.Conv3d(self.conv_blocks_localization[ds][-1].output_features, num_classes,
                                            1, 1, 0, 1, 1, bias=False))

    def forward(self, x):
        skips = []
        seg_outputs = []

        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)
            x = self.td[d](x)

        x = self.conv_blocks_context[-1](x)  ## bottleneck

        for u in range(len(self.tu)):
            x = self.tu[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            seg_outputs.append(self.seg_outputs[u](x))

        if self.do_ds:
            return tuple([seg_outputs[-1]] + [j for j in seg_outputs[:-1][::-1]])
        else:
            return seg_outputs[-1]