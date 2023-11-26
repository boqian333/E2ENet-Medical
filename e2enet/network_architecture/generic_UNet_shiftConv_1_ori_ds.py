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
from e2enet.utilities.nd_softmax import softmax_helper
from torch import nn
import torch
import numpy as np
from e2enet.network_architecture.initialization import InitWeights_He
from e2enet.network_architecture.neural_network import SegmentationNetwork
import torch.nn.functional
import torch.nn.functional as F

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

    def partition(self, x, d_parts, h_parts, w_parts):
        x = x.reshape(-1, self.C, d_parts, self.d, h_parts, self.h, w_parts, self.w)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)
        return x

    def partition_affine(self, x, d_parts, h_parts, w_parts):
        fc_inputs = x.reshape(-1, self.S * self.d * self.h * self.w, 1, 1, 1)
        out = self.fc3(fc_inputs)
        out = out.reshape(-1, self.S, self.d, self.h, self.w)
        out = self.fc3_bn(out)
        out = out.reshape(-1, d_parts, h_parts, w_parts, self.S, self.d, self.h, self.w)
        return out

    def forward(self, x):
        # origin_shape = x.size()
        # d_parts = origin_shape[2] // self.d
        # h_parts = origin_shape[3] // self.h
        # w_parts = origin_shape[4] // self.w
        # # h_parts = 1
        # # w_parts = 1
        # print(d_parts, h_parts, w_parts)
        #
        # partitions = self.partition(x, d_parts, h_parts, w_parts)
        #
        # #   Channel Perceptron
        # fc3_out = self.partition_affine(partitions, d_parts, h_parts, w_parts)
        #
        # fc3_out = fc3_out.permute(0, 4, 1, 5, 2, 6, 3, 7)  # N, O, h_parts, out_h, w_parts, out_w
        # out = fc3_out.reshape(*origin_shape)
        #
        # print("out:", out.shape)

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

class GlobalPerceptron(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(GlobalPerceptron, self).__init__()
        self.fc1 = nn.Conv3d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.fc2 = nn.Conv3d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x = F.adaptive_avg_pool3d(inputs, output_size=(1, 1, 1))
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        x = F.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1, 1)
        return x

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
        # self.gp = []

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
            self.conv_blocks_context.append(StackedConvLayers(input_features, output_features, num_conv_per_stage,input_resolution[d],
                                                              self.conv_op, self.conv_kwargs, self.norm_op,
                                                              self.norm_op_kwargs, self.dropout_op,
                                                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                              first_stride, basic_block=basic_block))
            self.conv_concat.append(nn.Conv3d(input_features+output_features, output_features, 1, 1, 0))
            # self.gp.append(GlobalPerceptron(input_channels=output_features, internal_neurons=output_features // 4))

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

        self.conv_concat.append(nn.Conv3d(input_features + final_num_features, final_num_features, 1, 1, 0))
        # self.gp.append(GlobalPerceptron(input_channels=final_num_features, internal_neurons=final_num_features // 4))

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
        self.S = [1 for i in range(num_pool+1)]

        self.fc3 = nn.ModuleList()
        for n in range(num_pool):
            first_stride = pool_op_kernel_sizes[n]
            if n == num_pool-1:
                self.fc3.append(nn.Conv3d(input_resolution[n][0]*input_resolution[n][1]*input_resolution[n][2]*self.S[n], input_resolution[n][0]//first_stride[0]*input_resolution[n][1]//first_stride[1]*input_resolution[n][2]//first_stride[2]*self.S[n], 1, 1, 0, bias=False, groups=self.S[n]))
            else:
                self.fc3.append(nn.Conv3d(input_resolution[n][0] // (input_resolution[n][0]//4) * input_resolution[n][1] // 2 * input_resolution[n][2] // 2 * self.S[n], input_resolution[n][0] // (input_resolution[n][0]//4) //first_stride[0]* input_resolution[n][1] // 2 //first_stride[1] * input_resolution[n][2] // 2 //first_stride[2]* self.S[n], 1, 1, 0, bias=False, groups=self.S[n]))

        self.fc3.append(nn.Conv3d(input_resolution[num_pool][0] * input_resolution[num_pool][1] * input_resolution[num_pool][2] * self.S[-1],
                                  input_resolution[num_pool][0] * input_resolution[num_pool][1] * input_resolution[num_pool][2] * self.S[-1], 1,
                                  1, 0, bias=False, groups=self.S[-1]))

        self.fc3_bn = nn.ModuleList()
        for n in range(num_pool+1):
            if n == 0:
                self.fc3_bn.append(nn.BatchNorm3d(self.S[n]))
            else:

                self.fc3_bn.append(nn.BatchNorm3d(self.S[n]))
            # self.fc3_bn.append(self.norm_op(self.S[n], **self.norm_op_kwargs))

        # register all modules properly
        self.conv_concat = nn.ModuleList(self.conv_concat)
        # self.gp = nn.ModuleList(self.gp)
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        self.fc3 = nn.ModuleList(self.fc3)
        self.fc3_bn = nn.ModuleList(self.fc3_bn)
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

    # def partition(self, x, c, d_parts, h_parts, w_parts, d, h, w):
    #     x = x.reshape(-1, c, d_parts, d, h_parts, h, w_parts, w)
    #     x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)
    #     return x

    def partition(self, x, c, d_parts, h_parts, w_parts, d, h, w):
        x = x.reshape(-1, c, d, d_parts, h, h_parts, w, w_parts)
        x = x.permute(0, 3, 5, 7, 1, 2, 4, 6)
        return x

    def partition_affine(self, x, n_layer, S, d_parts, h_parts, w_parts, d, h, w, pool_kernel=[1, 1, 1]):
        fc_inputs = x.reshape(-1, S * d * h * w, 1, 1, 1)
        out = self.fc3[n_layer](fc_inputs)
        out = out.reshape(-1, S, d//pool_kernel[0], h//pool_kernel[1], w//pool_kernel[2])
        out = self.fc3_bn[n_layer](out)
        out = out.reshape(-1, d_parts, h_parts, w_parts, S, d//pool_kernel[0], h//pool_kernel[1], w//pool_kernel[2])
        #print('partition_affine')
        return out


    def forward(self, x):
        skips = []
        seg_outputs = []
        d = []
        h = []
        w = []


        for n in range(len(self.conv_blocks_context) - 1):
            origin_shape = x.size()
            if n == len(self.conv_blocks_context) - 2:
                d_parts = 1
                h_parts = 1
                w_parts = 1
            else:
                d_parts = origin_shape[2]//4
                h_parts = 2
                w_parts = 2

            d = origin_shape[2] // d_parts
            h = origin_shape[3] // h_parts
            w = origin_shape[4] // w_parts

            partitions = self.partition(x, origin_shape[1], d_parts, h_parts, w_parts, d, h, w)

            #   Channel Perceptron
            fc3_out = self.partition_affine(partitions, n, self.S[n], d_parts, h_parts, w_parts, d, h, w, self.pool_op_kernel_sizes[n])

            fc3_out = fc3_out.permute(0, 4, 1, 5, 2, 6, 3, 7)  # N, O, h_parts, out_h, w_parts, out_w


            out = fc3_out.reshape(origin_shape[0], origin_shape[1], origin_shape[2]//2, origin_shape[3]//2, origin_shape[4]//2)


            x = self.conv_blocks_context[n](x)

            skips.append(x)
            if not self.convolutional_pooling:
                x = self.td[n](x)

            x = torch.cat([out, x], 1)
            x = self.conv_concat[n](x)

            #   Global Perceptron
            # global_vec = self.gp[n](x)
            # x = x * global_vec



        origin_shape = x.size()
        partitions = self.partition(x, origin_shape[1], 1, 1, 1, origin_shape[2], origin_shape[3], origin_shape[4])

        #   Channel Perceptron
        fc3_out = self.partition_affine(partitions, len(self.conv_blocks_context) - 1, self.S[-1], 1, 1, 1,
                                        origin_shape[2], origin_shape[3], origin_shape[4])

        fc3_out = fc3_out.permute(0, 4, 1, 5, 2, 6, 3, 7)  # N, O, h_parts, out_h, w_parts, out_w
        out = fc3_out.reshape(*origin_shape)

        x = self.conv_blocks_context[-1](x)

        x = torch.cat([out, x], 1)
        x = self.conv_concat[-1](x)

        # global_vec = self.gp[-1](x)
        # x = x * global_vec

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
