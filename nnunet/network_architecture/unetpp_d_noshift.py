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
from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn.functional
from nnunet.network_architecture.neural_network import SegmentationNetwork


def softmax_helper(x):
    return F.softmax(x, 1)

class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)

class torch_shift(nn.Module):
    def __init__(self, shift_size, dim, dim_num):
        super().__init__()
        self.shift_size = shift_size
        self.dim = dim
        self.dim_num = dim_num

    def forward(self, x):
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
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, shift_size=3):
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
        # if self.conv.kernel_size == (1, 3, 3):
        #     x = self.shift_D(x)
            #     print("(kernel {}) -- 3)".format(self.conv.kernel_size))
            # else:
            #     print("(kernel {}) -- 1)".format(self.conv.kernel_size))
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
    def __init__(self, input_feature_channels, output_feature_channels, num_convs,
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
            *([basic_block(input_feature_channels, output_feature_channels, self.conv_op,
                           self.conv_kwargs_first_conv,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs)] +
              [basic_block(output_feature_channels, output_feature_channels, self.conv_op,
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


class Generic_UNetPlusPlus(SegmentationNetwork):
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
    use_this_for_batch_size_computation_3D = 520000000 * 2  # 505789440

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
        super(Generic_UNetPlusPlus, self).__init__()
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
        elif conv_op == nn.Conv3d:
            upsample_mode = 'trilinear'
            pool_op = nn.MaxPool3d
            transpconv = nn.ConvTranspose3d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2, 2)] * num_pool
            if conv_kernel_sizes is None or True:
                conv_kernel_sizes = [(1, 3, 3)] * (num_pool + 1)
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
        # self.conv_blocks_localization = []
        self.loc0 = []
        self.loc1 = []
        self.loc2 = []
        self.loc3 = []
        self.loc4 = []
        self.td = []
        self.up0 = []
        self.up1 = []
        self.up2 = []
        self.up3 = []
        self.up4 = []
        # self.tu = []
        self.seg_outputs = []

        output_features = base_num_features
        input_features = input_channels

        for d in range(num_pool):
            # determine the first stride
            if d != 0 and self.convolutional_pooling:
                first_stride = pool_op_kernel_sizes[d - 1]
            else:
                first_stride = None

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[d]
            # add convolutions
            self.conv_blocks_context.append(StackedConvLayers(input_features, output_features, num_conv_per_stage,
                                                              self.conv_op, self.conv_kwargs, self.norm_op,
                                                              self.norm_op_kwargs, self.dropout_op,
                                                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                              first_stride, basic_block=basic_block))
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
            StackedConvLayers(input_features, output_features, num_conv_per_stage - 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, first_stride, basic_block=basic_block),
            StackedConvLayers(output_features, final_num_features, 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, basic_block=basic_block)))

        # if we don't want to do dropout in the localization pathway then we set the dropout prob to zero here
        if not dropout_in_localization:
            old_dropout_p = self.dropout_op_kwargs['p']
            self.dropout_op_kwargs['p'] = 0.0

        # now lets build the localization pathway
        encoder_features = final_num_features
        self.loc0, self.up0, encoder_features, self.down0 = self.create_nest(0, num_pool, final_num_features, num_conv_per_stage,
                                                                 basic_block, transpconv)
        self.loc1, self.up1, encoder_features1, self.down1 = self.create_nest(1, num_pool, encoder_features, num_conv_per_stage,
                                                                  basic_block, transpconv)
        self.loc2, self.up2, encoder_features2, self.down2 = self.create_nest(2, num_pool, encoder_features1, num_conv_per_stage,
                                                                  basic_block, transpconv)
        self.loc3, self.up3, encoder_features3, self.down3 = self.create_nest(3, num_pool, encoder_features2, num_conv_per_stage,
                                                                  basic_block, transpconv)
        self.loc4, self.up4, encoder_features4, self.down4 = self.create_nest(4, num_pool, encoder_features3, num_conv_per_stage,
                                                                  basic_block, transpconv)

        # for ds in range(len(self.loc0)):
        #     self.seg_outputs.append(conv_op(self.conv_blocks_localization[-(ds+1)][-1].output_channels, num_classes, 1, 1, 0, 1, 1, seg_output_use_bias))

        self.seg_outputs.append(conv_op(self.loc0[-1][-1].output_channels, num_classes,
                                        1, 1, 0, 1, 1, seg_output_use_bias))
        self.seg_outputs.append(conv_op(self.loc0[-2][-1].output_channels, num_classes,
                                        1, 1, 0, 1, 1, seg_output_use_bias))
        self.seg_outputs.append(conv_op(self.loc0[-3][-1].output_channels, num_classes,
                                        1, 1, 0, 1, 1, seg_output_use_bias))
        self.seg_outputs.append(conv_op(self.loc0[-4][-1].output_channels, num_classes,
                                        1, 1, 0, 1, 1, seg_output_use_bias))


        self.upscale_logits_ops = []
        cum_upsample = np.cumprod(np.vstack(pool_op_kernel_sizes), axis=0)[::-1]
        for usl in range(num_pool-1):
            if self.upscale_logits:
                self.upscale_logits_ops.append(Upsample(scale_factor=tuple([int(i) for i in cum_upsample[usl+1]]),
                                                        mode=upsample_mode))
            else:
                self.upscale_logits_ops.append(lambda x: x)

        if not dropout_in_localization:
            self.dropout_op_kwargs['p'] = old_dropout_p

        # register all modules properly
        # self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        self.loc0 = nn.ModuleList(self.loc0)
        self.loc1 = nn.ModuleList(self.loc1)
        self.loc2 = nn.ModuleList(self.loc2)
        self.loc3 = nn.ModuleList(self.loc3)
        self.loc4 = nn.ModuleList(self.loc4)
        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        self.td = nn.ModuleList(self.td)
        self.up0 = nn.ModuleList(self.up0)
        self.up1 = nn.ModuleList(self.up1)
        self.up2 = nn.ModuleList(self.up2)
        self.up3 = nn.ModuleList(self.up3)
        self.up4 = nn.ModuleList(self.up4)

        self.down0 = nn.ModuleList(self.down0)
        self.down1 = nn.ModuleList(self.down1)
        self.down2 = nn.ModuleList(self.down2)
        self.down3 = nn.ModuleList(self.down3)
        self.down4 = nn.ModuleList(self.down4)


        self.seg_outputs = nn.ModuleList(self.seg_outputs)
        if self.upscale_logits:
            self.upscale_logits_ops = nn.ModuleList(
                self.upscale_logits_ops)  # lambda x:x is not a Module so we need to distinguish here

        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)
            # self.apply(print_module_training_status)

    def forward(self, x):
        # print(x.shape)
        seg_outputs = []

        x0_0 = self.conv_blocks_context[0](x)
        x1_0 = self.conv_blocks_context[1](x0_0)
        x0_1 = self.loc4[0](torch.cat([x0_0, self.up4[0](x1_0)], 1))

        x2_0 = self.conv_blocks_context[2](x1_0)
        x1_1 = self.loc3[0](torch.cat([x1_0, self.up3[0](x2_0), self.down3[0](x0_0)], 1))  ## self.loc3[0] dim
        x0_2 = self.loc3[1](torch.cat([x0_1, self.up3[1](x1_1)], 1))


        x3_0 = self.conv_blocks_context[3](x2_0)
        x2_1 = self.loc2[0](torch.cat([x2_0, self.up2[0](x3_0), self.down2[0](x1_0)], 1))
        x1_2 = self.loc2[1](torch.cat([x1_1, self.up2[1](x2_1), self.down2[1](x0_1)], 1))
        x0_3 = self.loc2[2](torch.cat([x0_2, self.up2[2](x1_2)], 1))
        #seg_outputs.append(self.final_nonlin(self.seg_outputs[-1](x0_3)))

        x4_0 = self.conv_blocks_context[4](x3_0)
        x3_1 = self.loc1[0](torch.cat([x3_0, self.up1[0](x4_0), self.down1[0](x2_0)], 1))
        x2_2 = self.loc1[1](torch.cat([x2_1, self.up1[1](x3_1), self.down1[1](x1_1)], 1))
        x1_3 = self.loc1[2](torch.cat([x1_2, self.up1[2](x2_2), self.down1[2](x0_2)], 1))
        x0_4 = self.loc1[3](torch.cat([x0_3, self.up1[3](x1_3)], 1))
        #seg_outputs.append(self.final_nonlin(self.seg_outputs[-2](x0_4)))

        x5_0 = self.conv_blocks_context[5](x4_0)
        x4_1 = self.loc0[0](torch.cat([x4_0, self.up0[0](x5_0), self.down0[0](x3_0)], 1))
        x3_2 = self.loc0[1](torch.cat([x3_1, self.up0[1](x4_1), self.down0[1](x2_1)], 1))
        x2_3 = self.loc0[2](torch.cat([x2_2, self.up0[2](x3_2), self.down0[2](x1_2)], 1))
        x1_4 = self.loc0[3](torch.cat([x1_3, self.up0[3](x2_3), self.down0[3](x0_3)], 1))
        x0_5 = self.loc0[4](torch.cat([x0_4, self.up0[4](x1_4)], 1))

        seg_outputs.append(self.final_nonlin(self.seg_outputs[3](x3_2)))
        seg_outputs.append(self.final_nonlin(self.seg_outputs[2](x2_3)))
        seg_outputs.append(self.final_nonlin(self.seg_outputs[1](x1_4)))
        seg_outputs.append(self.final_nonlin(self.seg_outputs[0](x0_5)))

        if self._deep_supervision and self.do_ds:
            return list([seg_outputs[-1]] + [i(j) for i, j in zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            return seg_outputs[-1]

    # now lets build the localization pathway BACK_UP
    def create_nest(self, z, num_pool, final_num_features, num_conv_per_stage, basic_block, transpconv):
        # print(final_num_features)
        conv_blocks_localization = []
        tu = []
        tdown = []
        i = 0
        for u in range(z, num_pool):
            nfeatures_from_down = final_num_features
            nfeatures_from_skip = self.conv_blocks_context[-(2 + u)].output_channels

            if u != num_pool - 1:
                n_features_after_tu_and_concat = nfeatures_from_skip * 2 + self.conv_blocks_context[-(3 + u)].output_channels
            else:
                n_features_after_tu_and_concat = nfeatures_from_skip * 2


            if i == 0:
                unet_final_features = nfeatures_from_skip
                i += 1
            # the first conv reduces the number of features to match those of skip
            # the following convs work on that number of features
            # if not convolutional upsampling then the final conv reduces the num of features again
            if u != num_pool - 1 and not self.convolutional_upsampling:
                final_num_features = self.conv_blocks_context[-(3 + u)].output_channels
            else:
                final_num_features = nfeatures_from_skip

            if not self.convolutional_upsampling:
                tu.append(Upsample(scale_factor=self.pool_op_kernel_sizes[-(u + 1)], mode=self.upsample_mode))
            else:
                tu.append(transpconv(nfeatures_from_down, nfeatures_from_skip, self.pool_op_kernel_sizes[-(u + 1)],
                          self.pool_op_kernel_sizes[-(u + 1)], bias=False))
                if u + 2 <= len(self.pool_op_kernel_sizes):
                    tdown.append((nn.MaxPool3d(self.pool_op_kernel_sizes[-(u + 2)])))


            if z != 0:
                self.conv_kwargs['kernel_size'] = (1, 3, 3)
                conv_pad_sizes=[i // 2 for i in self.conv_kwargs['kernel_size']]
                self.conv_kwargs['padding'] = conv_pad_sizes

                conv_blocks_localization.append(nn.Sequential(
                    StackedConvLayers(n_features_after_tu_and_concat, final_num_features, num_conv_per_stage - 1,
                                      self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs,
                                      self.dropout_op,
                                      self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs, basic_block=basic_block)))
            else:
                self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[- (u + 1)]
                self.conv_kwargs['padding'] = self.conv_pad_sizes[- (u+1)]
                conv_blocks_localization.append(nn.Sequential(
                    StackedConvLayers(n_features_after_tu_and_concat, nfeatures_from_skip, num_conv_per_stage - 1,
                                      self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs,
                                      self.dropout_op,
                                      self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs, basic_block=basic_block),
                    StackedConvLayers(nfeatures_from_skip, final_num_features, 1, self.conv_op, self.conv_kwargs,
                                      self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                                      self.nonlin, self.nonlin_kwargs, basic_block=basic_block)))
            # print(final_num_features)
        # print('conv_blocks_localization')
        return conv_blocks_localization, tu, unet_final_features, tdown

    @staticmethod
    def compute_approx_vram_consumption(patch_size, num_pool_per_axis, base_num_features, max_num_features,
                                        num_modalities, num_classes, pool_op_kernel_sizes, deep_supervision=False,
                                        conv_per_stage=2):
        """
        This only applies for num_conv_per_stage and convolutional_upsampling=True
        not real vram consumption. just a constant term to which the vram consumption will be approx proportional
        (+ offset for parameter storage)
        :param deep_supervision:
        :param patch_size:
        :param num_pool_per_axis:
        :param base_num_features:
        :param max_num_features:
        :param num_modalities:
        :param num_classes:
        :param pool_op_kernel_sizes:
        :return:
        """
        if not isinstance(num_pool_per_axis, np.ndarray):
            num_pool_per_axis = np.array(num_pool_per_axis)

        npool = len(pool_op_kernel_sizes)

        map_size = np.array(patch_size)
        tmp = np.int64((conv_per_stage * 2 + 1) * np.prod(map_size, dtype=np.int64) * base_num_features +
                       num_modalities * np.prod(map_size, dtype=np.int64) +
                       num_classes * np.prod(map_size, dtype=np.int64))

        num_feat = base_num_features

        for p in range(npool):
            for pi in range(len(num_pool_per_axis)):
                map_size[pi] /= pool_op_kernel_sizes[p][pi]
            num_feat = min(num_feat * 2, max_num_features)
            num_blocks = (conv_per_stage * 2 + 1) if p < (npool - 1) else conv_per_stage  # conv_per_stage + conv_per_stage for the convs of encode/decode and 1 for transposed conv
            tmp += num_blocks * np.prod(map_size, dtype=np.int64) * num_feat
            if deep_supervision and p < (npool - 2):
                tmp += np.prod(map_size, dtype=np.int64) * num_classes
            # print(p, map_size, num_feat, tmp)
        return tmp
