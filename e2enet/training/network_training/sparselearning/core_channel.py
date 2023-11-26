from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .snip import SNIP, GraSP
import copy
import random

import numpy as np
import math

def str2bool(str):
    return True if str.lower() == 'true' else False


def add_sparse_args(parser):
    parser.add_argument('--sparse', type=str2bool, default=True, help='Enable sparse mode. Default: True.')
    parser.add_argument('--adv', type=bool, default=False, help='adv sparse mode. Default: True.')
    parser.add_argument('--init-prune-epoch', type=int, default=0, help='The pruning rate / death rate.')
    parser.add_argument('--final-prune-epoch', type=int, default=1000, help='The density of the overall sparse network.')
    parser.add_argument('--fix', type=bool, default=False, help='Fix sparse connectivity during training. Default: True.')
    parser.add_argument('--sparse_init', type=str, default='uniform', help='sparse initialization: ERK, snip, Grasp')
    parser.add_argument('--growth', type=str, default='random', help='Growth mode. Choose from: momentum, random, random_unfired, and gradient.')
    parser.add_argument('--death', type=str, default='magnitude', help='Death mode / pruning mode. Choose from: magnitude, SET, threshold.')
    parser.add_argument('--redistribution', type=str, default='none', help='Redistribution mode. Choose from: momentum, magnitude, nonzeros, or none.')
    parser.add_argument('--death-rate', type=float, default=0.50, help='The pruning rate / death rate.')
    parser.add_argument('--density', type=float, default=0.3, help='The density of the overall sparse network.')
    parser.add_argument('--final_density', type=float, default=0.05, help='The density of the overall sparse network.')
    parser.add_argument('--update_frequency', type=int, default=5, metavar='N', help='how many iterations to train between parameter exploration')
    parser.add_argument('--decay-schedule', type=str, default='cosine', help='The decay schedule for the pruning rate. Default: cosine. Choose from: cosine, linear.')
class CosineDecay(object):
    def __init__(self, death_rate, T_max, eta_min=0.001, last_epoch=-1):
        self.sgd = optim.SGD(torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))]), lr=death_rate)
        self.cosine_stepper = torch.optim.lr_scheduler.CosineAnnealingLR(self.sgd, T_max, eta_min, last_epoch)

    def step(self):
        self.cosine_stepper.step()

    def get_dr(self):
        return self.sgd.param_groups[0]['lr']

class LinearDecay(object):
    def __init__(self, death_rate, factor=0.99, frequency=600):
        self.factor = factor
        self.steps = 0
        self.frequency = frequency

    def step(self):
        self.steps += 1

    def get_dr(self, death_rate):
        if self.steps > 0 and self.steps % self.frequency == 0:
            return death_rate*self.factor
        else:
            return death_rate


class Masking(object):
    def __init__(self, optimizer, death_rate=0.3, growth_death_ratio=1.0, death_rate_decay=None, death_mode='magnitude', growth_mode='momentum', redistribution_mode='momentum', threshold=0.001, train_loader=None, T_max=0., args=None):
        growth_modes = ['random', 'momentum', 'momentum_neuron', 'gradient']
        if growth_mode not in growth_modes:
            print('Growth mode: {0} not supported!'.format(growth_mode))
            print('Supported modes are:', str(growth_modes))

        self.args = args
        self.device = torch.device("cuda")
        self.growth_mode = growth_mode
        self.death_mode = death_mode
        self.growth_death_ratio = growth_death_ratio
        self.redistribution_mode = redistribution_mode
        self.death_rate_decay = death_rate_decay

        self.masks = {}
        self.modules = []
        self.names = []
        self.optimizer = optimizer

        # stats
        self.name2zeros = {}
        self.num_remove = {}
        self.num_death = {}
        self.name2nonzeros = {}
        self.death_rate = death_rate
        self.baseline_nonzero = None
        self.steps = 0
        self.explore_step = 0

        self.pruned_masks = {}
        self.regrowed_masks = {}
        self.pre_masks = None
        self.decay_flag = True

        self.total_nozeros = 0
        self.total_weights = 0
        self.loader = train_loader
        self.regrow_ratio = 1.01
        self.adv = self.args.adv
        self.curr_density = 0.0
        self.regrow_ones = 0
        self.T_max = T_max

        # if fix, then we do not explore the sparse connectivity
        if self.args.fix:
            self.prune_every_k_steps = None
        else:
            self.prune_every_k_steps = self.args.update_frequency

    def init(self, mode='ERK', density=0.05, erk_power_scale=1.0):
        self.density = density
        if mode == 'GMP':
            self.baseline_nonzero = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.masks[name] = torch.ones_like(weight, dtype=torch.float32, requires_grad=False).cuda()
                    self.baseline_nonzero += (self.masks[name] != 0).sum().int().item()

        elif mode == 'lottery_ticket':
            print('initialize by lottery ticket')
            self.baseline_nonzero = 0
            weight_abs = []
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    weight_abs.append(torch.abs(weight))

            # Gather all scores in a single vector and normalise
            all_scores = torch.cat([torch.flatten(x) for x in weight_abs])
            num_params_to_keep = int(len(all_scores) * self.density)

            threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
            acceptable_score = threshold[-1]

            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.masks[name] = ((torch.abs(weight)) >= acceptable_score).float()
                    self.baseline_nonzero += (self.masks[name]!=0).sum().int().item()

        elif mode == 'uniform':
            self.baseline_nonzero = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    # if np.prod(weight.shape[-3:]) == 1: continue
                    if weight.shape[0] == 48:
                        #if density === 0.3
                        density_n = 0.2
                    else:
                        density_n = density
                    k_size = np.prod(weight.shape[-3:])
                    nonzeros = weight.numel()*density_n

                    slice = torch.zeros((weight.shape[0], weight.shape[1]))
                    idx_tensor = torch.nonzero(slice < 1)

                    kernel_num = round(nonzeros/k_size)
                    idx_rand = random.sample(list(range(0, idx_tensor.shape[0])), kernel_num)

                    kk = idx_tensor[idx_rand]
                    idx_x = kk[:, 0]
                    idx_y = kk[:, 1]
                    self.masks[name][idx_x, idx_y] = 1.0

                    # aa = self.masks[name].cpu().numpy()
                    self.baseline_nonzero += (self.masks[name] != 0).sum().item()
                    density_new = (self.masks[name] != 0).sum().item()/self.masks[name].numel()
                    print(f"layer: {name}, shape: {self.masks[name].shape}, density: {density_new}")

        elif mode == 'uniform_ori':
            self.baseline_nonzero = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.masks[name][:] = (torch.rand(weight.shape) < density).float().data.cuda()  # lsw
                    # self.masks[name][:] = (torch.rand(weight.shape) < density).float().data #lsw
                    self.baseline_nonzero += weight.numel() * density
                    print(f"layer: {name}, shape: {self.masks[name].shape}, density: {density}")

        elif mode == 'snip':
            print('initialize by snip')
            layer_wise_sparsities, keep_masks = SNIP(self.module, self.density, self.loader, self.device)
            # re-sample mask positions
            for sparsity_, name in zip(layer_wise_sparsities, self.masks):
                self.masks[name][:] = (torch.rand(self.masks[name].shape) < (1-sparsity_)).float().data.cuda()

            # self.baseline_nonzero = 0
            # for spe_initial, mask in zip(keep_masks, self.masks):
            #     assert (spe_initial.shape == self.masks[mask].shape)
            #     self.masks[mask] = spe_initial
            #     self.baseline_nonzero += self.masks[mask].numel() * density

        elif mode == 'GraSP':
            print('initialize by GraSP')
            layer_wise_sparsities = GraSP(self.module, self.density, self.loader, self.device)
            # re-sample mask positions
            for sparsity_, name in zip(layer_wise_sparsities, self.masks):
                self.masks[name][:] = (torch.rand(self.masks[name].shape) < (1-sparsity_)).float().data.cuda()


        elif mode == 'ERK':
            print('initialize by ERK')
            total_params = 0
            for name, weight in self.masks.items():
                total_params += weight.numel()
            is_epsilon_valid = False
            # # The following loop will terminate worst case when all masks are in the
            # custom_sparsity_map. This should probably never happen though, since once
            # we have a single variable or more with the same constant, we have a valid
            # epsilon. Note that for each iteration we add at least one variable to the
            # custom_sparsity_map and therefore this while loop should terminate.
            dense_layers = set()
            while not is_epsilon_valid:
                # We will start with all layers and try to find right epsilon. However if
                # any probablity exceeds 1, we will make that layer dense and repeat the
                # process (finding epsilon) with the non-dense layers.
                # We want the total number of connections to be the same. Let say we have
                # for layers with N_1, ..., N_4 parameters each. Let say after some
                # iterations probability of some dense layers (3, 4) exceeded 1 and
                # therefore we added them to the dense_layers set. Those layers will not
                # scale with erdos_renyi, however we need to count them so that target
                # paratemeter count is achieved. See below.
                # eps * (p_1 * N_1 + p_2 * N_2) + (N_3 + N_4) =
                #    (1 - default_sparsity) * (N_1 + N_2 + N_3 + N_4)
                # eps * (p_1 * N_1 + p_2 * N_2) =
                #    (1 - default_sparsity) * (N_1 + N_2) - default_sparsity * (N_3 + N_4)
                # eps = rhs / (\sum_i p_i * N_i) = rhs / divisor.

                divisor = 0
                rhs = 0
                raw_probabilities = {}
                for name, mask in self.masks.items():
                    n_param = np.prod(mask.shape)
                    n_zeros = n_param * (1 - self.density)
                    n_ones = n_param * self.density

                    if name in dense_layers:
                        rhs -= n_zeros
                    else:
                        rhs += n_ones
                        raw_probabilities[name] = (np.sum(mask.shape) / np.prod(mask.shape)) ** erk_power_scale
                        divisor += raw_probabilities[name] * n_param
                epsilon = rhs / divisor
                max_prob = np.max(list(raw_probabilities.values()))
                max_prob_one = max_prob * epsilon
                if max_prob_one > 1:
                    is_epsilon_valid = False
                    for mask_name, mask_raw_prob in raw_probabilities.items():
                        if mask_raw_prob == max_prob:
                            print(f"Sparsity of var:{mask_name} had to be set to 0.")
                            dense_layers.add(mask_name)
                else:
                    is_epsilon_valid = True

            density_dict = {}
            total_nonzero = 0.0
            # With the valid epsilon, we can set sparsities of the remaning layers.
            for name, mask in self.masks.items():
                n_param = np.prod(mask.shape)
                if name in dense_layers:
                    density_dict[name] = 1.0
                else:
                    probability_one = epsilon * raw_probabilities[name]
                    density_dict[name] = probability_one
                print(
                    f"layer: {name}, shape: {mask.shape}, density: {density_dict[name]}"
                )
                self.masks[name][:] = (torch.rand(mask.shape) < density_dict[name]).float().data.cuda()

                total_nonzero += density_dict[name] * mask.numel()
            print(f"Overall sparsity {total_nonzero / total_params}")

        self.apply_mask()
        self.fired_masks = copy.deepcopy(self.masks) # used for ITOP
        # self.print_nonzero_counts()

        total_size = 0
        for name, weight in self.masks.items():
            total_size  += weight.numel()
        print('Total Model parameters:', total_size)

        sparse_size = 0
        for name, weight in self.masks.items():
            sparse_size += (weight != 0).sum().int().item()

        print('Total parameters under sparsity level of {0}: {1}'.format(self.density, sparse_size / total_size))


    def step(self):
        #self.optimizer.step()
        self.apply_mask()

        if self.decay_flag:
            self.death_rate_decay.step()
            self.death_rate = self.death_rate_decay.get_dr()
        else:
            self.death_rate = 0.001
            self.adv = False

        self.steps += 1

        if self.prune_every_k_steps is not None:

            if self.steps % self.prune_every_k_steps == 0:
                ## low to high regrow
                self.explore_step += 1
                self.truncate_weights()

                self.cal_nonzero_counts()
                self.curr_density = self.total_nozeros / self.total_weights
                print('curr_density: {0:.4f}, final_density:{1:.4f}'.format(self.curr_density, self.args.final_density))

                _, _ = self.fired_masks_update()
                if self.explore_step > 1:
                    self.print_nonzero_counts()
                self.pre_masks = copy.deepcopy(self.pruned_masks)


    def add_module(self, module, density, sparse_init='ER'):
        self.modules.append(module)
        self.module = module
        for name, tensor in module.named_parameters():
            if ('loc' in name and 'context' not in name) or 'up' in name:
                self.names.append(name)
                self.masks[name] = torch.zeros_like(tensor, dtype=torch.float32, requires_grad=False).cuda()

        print('Removing biases...')
        self.remove_weight_partial_name('bias')
        print('Removing biases...')
        self.remove_weight_partial_name('instnorm')
        print('Removing 2D batch norms...')
        self.remove_type(nn.BatchNorm2d)
        print('Removing 1D batch norms...')
        self.remove_type(nn.BatchNorm1d)
        self.init(mode=sparse_init, density=density)

    def cal_nonzero_counts(self):
        self.total_nozeros = 0
        self.total_weights = 0

        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name]
                self.total_nozeros += (mask != 0).sum().item()
                self.total_weights += mask.numel()


    def cal_grow_schedule(self):
        self.total_nozeros = 0
        self.total_weights = 0

        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name]
                self.total_nozeros += (mask != 0).sum().item()
                self.total_weights += mask.numel()

        curr_prune_iter = int(self.steps / self.prune_every_k_steps)
        final_iter = int((self.args.final_prune_epoch * len(self.loader)) / self.prune_every_k_steps)
        ini_iter = int((self.args.init_prune_epoch * len(self.loader)) / self.prune_every_k_steps)
        total_prune_iter = final_iter - ini_iter

        process_flag = (self.regrow_ratio > 1.0) or (self.curr_density < (self.args.final_density-0.0003))
        if curr_prune_iter >= ini_iter and curr_prune_iter <= final_iter:
            prune_decay = (1 - ((curr_prune_iter - ini_iter) / total_prune_iter)) ** 3
            curr_sparse_level = self.args.density + (self.args.final_density - self.args.density) * (1 - prune_decay)

            curr_ones = self.total_weights * curr_sparse_level
            self.regrow_ones = int(curr_ones - self.total_nozeros * (1-self.death_rate))

            if process_flag:
                self.regrow_ratio = self.regrow_ones / (self.total_nozeros * self.death_rate)

                print('******************************************************')
                print('Pruning Progress is {0}/{1}--curr_sparse_level:{2:.4f} regrow_ratio:{3:.4f}'.format(curr_prune_iter, total_prune_iter,
                                                                                      curr_sparse_level, self.regrow_ratio))
                print('******************************************************')

            else:
                self.regrow_ratio = 1.0
        else:
            self.regrow_ratio = 1.0



    def remove_weight(self, name):
        if name in self.masks:
            print('Removing {0} of size {1} = {2} parameters.'.format(name, self.masks[name].shape,
                                                                      self.masks[name].numel()))
            self.masks.pop(name)
        elif name + '.weight' in self.masks:
            print('Removing {0} of size {1} = {2} parameters.'.format(name, self.masks[name + '.weight'].shape,
                                                                      self.masks[name + '.weight'].numel()))
            self.masks.pop(name + '.weight')
        else:
            print('ERROR', name)

    def remove_weight_partial_name(self, partial_name):
        removed = set()
        for name in list(self.masks.keys()):
            if partial_name in name:

                print('Removing {0} of size {1} with {2} parameters...'.format(name, self.masks[name].shape,
                                                                                   np.prod(self.masks[name].shape)))
                removed.add(name)
                self.masks.pop(name)

        print('Removed {0} layers.'.format(len(removed)))

        i = 0
        while i < len(self.names):
            name = self.names[i]
            if name in removed:
                self.names.pop(i)
            else:
                i += 1

    def remove_type(self, nn_type):
        for module in self.modules:
            for name, module in module.named_modules():
                if isinstance(module, nn_type):
                    self.remove_weight(name)

    def apply_mask(self):
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name in self.masks:
                    tensor.data = tensor.data*self.masks[name] ## 只是将data置0；gradient不管
                    # reset momentum
                    if 'momentum_buffer' in self.optimizer.state[tensor]:
                        self.optimizer.state[tensor]['momentum_buffer'] = self.optimizer.state[tensor]['momentum_buffer']*self.masks[name]

    def truncate_weights_GMP(self, epoch):
        '''
        Implementation  of GMP To prune, or not to prune: exploring the efficacy of pruning for model compression https://arxiv.org/abs/1710.01878
        :param epoch: current training epoch
        :return:
        '''
        prune_rate = 1 - self.density
        curr_prune_epoch = epoch
        total_prune_epochs = self.args.multiplier * self.args.final_prune_epoch - self.args.multiplier * self.args.init_prune_epoch + 1
        if epoch >= self.args.multiplier * self.args.init_prune_epoch and epoch <= self.args.multiplier * self.args.final_prune_epoch:
            prune_decay = (1 - ((curr_prune_epoch - self.args.multiplier * self.args.init_prune_epoch) / total_prune_epochs)) ** 3
            curr_prune_rate = prune_rate - (prune_rate * prune_decay)

            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue

                    x, idx = torch.sort(torch.abs(weight.data.view(-1)))
                    p = int(curr_prune_rate * weight.numel())
                    self.masks[name].data.view(-1)[idx[:p]] = 0.0
            self.apply_mask()
        total_size = 0
        for name, weight in self.masks.items():
            total_size += weight.numel()
        print('Total Model parameters:', total_size)

        sparse_size = 0
        for name, weight in self.masks.items():
            sparse_size += (weight != 0).sum().int().item()

        print('Total parameters under sparsity level of {0}: {1} after epoch of {2}'.format(self.density, sparse_size / total_size, epoch))


    def truncate_weights_global(self):

        print('Pruning and growing globally')
        self.baseline_nonzero = 0
        total_num_nonzoros = 0
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name]
                self.name2nonzeros[name] = mask.sum().item()
                total_num_nonzoros += self.name2nonzeros[name]

        weight_abs = []
        masks_vector = []
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                weight_abs.append(torch.abs(weight))
                masks_vector.append(self.masks[name])

        # Gather all scores in a single vector and normalise
        all_scores = torch.cat([torch.flatten(x) for x in weight_abs])
        num_params_to_keep = int(total_num_nonzoros * (1 - self.death_rate))

        # Masks to keep
        masks_ori = copy.deepcopy(self.masks)
        all_marks = torch.cat([torch.flatten(x) for x in masks_vector]).float()
        total_regrow = self.regrow_ratio * (total_num_nonzoros * self.death_rate)

        n = (all_marks == 0).sum().item()
        expeced_growth_probability = (total_regrow / n)
        new_weights = torch.rand(all_marks.shape).cuda() < expeced_growth_probability # lsw
        new_all_marks = all_marks.byte() | new_weights.byte()

        # new_weights = torch.rand(all_marks.shape).cuda() + all_marks
        # y, idx = torch.sort(new_weights, descending=True)
        # all_marks.data.view(-1)[idx[-total_regrow:]] = 1.0

        threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
        acceptable_score = threshold[-1]

        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                # ## fc; no prune
                # if 'classifier' in name:
                #     self.pruned_masks[name] = self.masks[name]
                #     continue

                self.masks[name] = ((torch.abs(weight)) >= acceptable_score).float()
                self.pruned_masks[name] = self.masks[name]

                # set the pruned weights to zero
                weight.data = weight.data * self.masks[name]
                if 'momentum_buffer' in self.optimizer.state[weight]:
                    self.optimizer.state[weight]['momentum_buffer'] = self.optimizer.state[weight]['momentum_buffer'] * \
                                                                      self.masks[name]

        ### grow random/gradient
        curr_index = 0
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                mark = self.masks[name]
                mark_ori = masks_ori[name]
                num_ele = mark.numel()
                regrow_num = (new_all_marks[curr_index:(curr_index+num_ele)]==1).sum().item() - (mark_ori==1).sum().item()

                # n = (mark == 0).sum().item()
                # expeced_growth_probability = (regrow_num / n)
                # new_weights = torch.rand(mark.shape).cuda() < expeced_growth_probability  # lsw
                # new_mask_ = mark.byte() | new_weights.byte()

                grad = self.get_gradient_for_weights(weight)
                grad = grad * (mark_ori == 0).float()
                y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
                mark.data.view(-1)[idx[:regrow_num]] = 1.0
                new_mask_ = mark

                # exchanging masks
                self.masks.pop(name)
                self.masks[name] = new_mask_.float()
                curr_index = curr_index + num_ele

        self.apply_mask()


    def truncate_weights(self):

        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name]
                self.name2nonzeros[name] = mask.sum().item()
                self.name2zeros[name] = mask.numel() - self.name2nonzeros[name]

                # death
                if self.death_mode == 'magnitude':
                    new_mask, num_death = self.kernel_death(mask, weight, name) ##
                    #new_mask = self.magnitude_death(mask, weight, name)
                elif self.death_mode == 'SET':
                    new_mask = self.magnitude_and_negativity_death(mask, weight, name)
                elif self.death_mode == 'Taylor_FO':
                    new_mask = self.taylor_FO(mask, weight, name)
                elif self.death_mode == 'threshold':
                    new_mask = self.threshold_death(mask, weight, name)

                self.num_remove[name] = int(self.name2nonzeros[name] - new_mask.sum().item())
                self.num_death[name] = num_death
                self.masks[name][:] = new_mask

                ## pick up the remain weights(before regrow)
                self.pruned_masks[name] = new_mask

        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                new_mask = self.masks[name].data.byte()

                # growth
                if self.growth_mode == 'random':
                    new_mask = self.kernel_growth(name, new_mask, weight)  ##
                    #new_mask = self.random_growth(name, new_mask, weight)

                if self.growth_mode == 'random_unfired':
                    new_mask = self.random_unfired_growth(name, new_mask, weight)

                elif self.growth_mode == 'momentum':
                    new_mask = self.momentum_growth(name, new_mask, weight)

                elif self.growth_mode == 'gradient':
                    new_mask = self.kernel_grad_growth(name, new_mask, weight)

                new_nonzero = new_mask.sum().item()

                # exchanging masks
                self.masks.pop(name)
                self.masks[name] = new_mask.float()

                ## pick up the remain weights(after regrow)
                self.regrowed_masks[name] = new_mask.float()

        self.apply_mask()


    '''
                    DEATH
    '''

    def threshold_death(self, mask, weight, name):
        return (torch.abs(weight.data) > self.threshold)

    def taylor_FO(self, mask, weight, name):

        num_remove = math.ceil(self.death_rate * self.name2nonzeros[name])
        num_zeros = self.name2zeros[name]
        k = math.ceil(num_zeros + num_remove)

        x, idx = torch.sort((weight.data * weight.grad).pow(2).flatten())
        mask.data.view(-1)[idx[:k]] = 0.0

        return mask

    def magnitude_death(self, mask, weight, name):

        num_remove = math.ceil(self.death_rate*self.name2nonzeros[name])
        if num_remove == 0.0: return weight.data != 0.0
        num_zeros = self.name2zeros[name]

        x, idx = torch.sort(torch.abs(weight.data.view(-1)))
        n = idx.shape[0]

        k = math.ceil(num_zeros + num_remove)
        threshold = x[k-1].item()

        return (torch.abs(weight.data) > threshold)


    def kernel_death(self, mask, weight, name):

        # print(f"{name} before death: {mask.sum().item()}")

        k_size = np.prod(weight.shape[-3:])
        data_tensor = copy.deepcopy(weight.data)
        data_sum1 = torch.sum(torch.abs(data_tensor), dim=-1)
        data_sum2 = torch.sum(data_sum1, dim=-1)
        data_sum = torch.sum(data_sum2, dim=-1)

        prune_num = math.ceil(self.death_rate*self.name2nonzeros[name]/k_size)
        value, idx = torch.sort(data_sum.data.view(-1))  ## s-b

        num_zeros = math.ceil(self.name2zeros[name]/k_size)
        idx2death = torch.nonzero(data_sum.data<=value[num_zeros+prune_num-1].item())

        mask[idx2death[:, 0], idx2death[:, 1]] = 0.0

        # print(f"{name} after death: {mask.sum().item()}")
        return mask, prune_num

    def magnitude_and_negativity_death(self, mask, weight, name):
        num_remove = math.ceil(self.death_rate*self.name2nonzeros[name])
        num_zeros = self.name2zeros[name]

        # find magnitude threshold
        # remove all weights which absolute value is smaller than threshold
        x, idx = torch.sort(weight[weight > 0.0].data.view(-1))
        k = math.ceil(num_remove/2.0)
        if k >= x.shape[0]:
            k = x.shape[0]

        threshold_magnitude = x[k-1].item()

        # find negativity threshold
        # remove all weights which are smaller than threshold
        x, idx = torch.sort(weight[weight < 0.0].view(-1))
        k = math.ceil(num_remove/2.0)
        if k >= x.shape[0]:
            k = x.shape[0]
        threshold_negativity = x[k-1].item()


        pos_mask = (weight.data > threshold_magnitude) & (weight.data > 0.0)
        neg_mask = (weight.data < threshold_negativity) & (weight.data < 0.0)


        new_mask = pos_mask | neg_mask
        return new_mask

    '''
                    GROWTH
    '''

    def random_unfired_growth(self, name, new_mask, weight):
        total_regrowth = self.num_remove[name]
        n = (new_mask == 0).sum().item()
        if n == 0: return new_mask
        num_nonfired_weights = (self.fired_masks[name]==0).sum().item()

        if total_regrowth <= num_nonfired_weights:
            idx = (self.fired_masks[name].flatten() == 0).nonzero()
            indices = torch.randperm(len(idx))[:total_regrowth]

            # idx = torch.nonzero(self.fired_masks[name].flatten())
            new_mask.data.view(-1)[idx[indices]] = 1.0
        else:
            new_mask[self.fired_masks[name]==0] = 1.0
            n = (new_mask == 0).sum().item()
            expeced_growth_probability = ((total_regrowth-num_nonfired_weights) / n)
            new_weights = torch.rand(new_mask.shape).cuda() < expeced_growth_probability
            new_mask = new_mask.byte() | new_weights
        return new_mask

    def kernel_growth(self, name, new_mask, weight):
        # print(f"{name} before grow: {new_mask.sum().item()} growth num: {self.num_death[name]}")
        # total_regrowth = self.num_remove[name]
        num_growth = self.num_death[name]

        data_tensor = copy.deepcopy(new_mask)
        data_sum1 = torch.sum(torch.abs(data_tensor), dim=-1)
        data_sum2 = torch.sum(data_sum1, dim=-1)
        data_sum = torch.sum(data_sum2, dim=-1)
        # aa = data_sum.cpu().numpy()

        idx_tensor = torch.nonzero(data_sum.data < 1)
        idx_rand = random.sample(list(range(0, idx_tensor.shape[0])), num_growth)

        idx2grow = idx_tensor[idx_rand]

        data_tensor[idx2grow[:, 0], idx2grow[:, 1]] = 1.0
        # print(f"{name} after grow: {data_tensor.sum().item()}")
        return data_tensor

    def random_growth(self, name, new_mask, weight):
        total_regrowth = self.num_remove[name]
        n = (new_mask==0).sum().item()
        if n == 0: return new_mask
        expeced_growth_probability = (total_regrowth/n)
        new_weights = torch.rand(new_mask.shape).cuda() < expeced_growth_probability
        new_mask_ = new_mask.byte() | new_weights.byte()
        if (new_mask_!=0).sum().item() == 0:
            new_mask_ = new_mask
        return new_mask_

    def momentum_growth(self, name, new_mask, weight):
        total_regrowth = self.num_remove[name]
        grad = self.get_momentum_for_weight(weight)
        grad = grad*(new_mask==0).float()
        y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
        new_mask.data.view(-1)[idx[:total_regrowth]] = 1.0

        return new_mask

    def gradient_growth(self, name, new_mask, weight):
        total_regrowth = self.num_remove[name]
        grad = self.get_gradient_for_weights(weight)
        grad = grad*(new_mask==0).float()

        y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
        new_mask.data.view(-1)[idx[:total_regrowth]] = 1.0

        return new_mask

    def kernel_grad_growth(self, name, new_mask, weight):
        num_growth = self.num_death[name]
        if num_growth == 0: return new_mask

        mask_tensor = copy.deepcopy(new_mask)
        mask_sum1 = torch.sum(torch.abs(mask_tensor), dim=-1)
        mask_sum = torch.squeeze(torch.sum(mask_sum1, dim=-1))

        data_tensor = self.get_gradient_for_weights(weight)
        data_sum1 = torch.sum(torch.abs(data_tensor), dim=-1)
        data_sum = torch.squeeze(torch.sum(data_sum1, dim=-1))

        grad = data_sum * (mask_sum < 1).float()

        value, idx = torch.sort(grad.data.view(-1), descending=True)  ## b-s
        idx2death = torch.nonzero(grad.data > value[num_growth].item())
        new_mask[idx2death[:, 0], idx2death[:, 1]] = 1.0

        # print(f"{name} after death: {mask.sum().item()}")
        return new_mask


    def momentum_neuron_growth(self, name, new_mask, weight):
        total_regrowth = self.num_remove[name]
        grad = self.get_momentum_for_weight(weight)

        M = torch.abs(grad)
        if len(M.shape) == 2: sum_dim = [1]
        elif len(M.shape) == 4: sum_dim = [1, 2, 3]

        v = M.mean(sum_dim).data
        v /= v.sum()

        slots_per_neuron = (new_mask==0).sum(sum_dim)

        M = M*(new_mask==0).float()
        for i, fraction  in enumerate(v):
            neuron_regrowth = math.floor(fraction.item()*total_regrowth)
            available = slots_per_neuron[i].item()

            y, idx = torch.sort(M[i].flatten())
            if neuron_regrowth > available:
                neuron_regrowth = available
            threshold = y[-(neuron_regrowth)].item()
            if threshold == 0.0: continue
            if neuron_regrowth < 10: continue
            new_mask[i] = new_mask[i] | (M[i] > threshold)

        return new_mask

    '''
                UTILITY
    '''
    def get_momentum_for_weight(self, weight):
        if 'exp_avg' in self.optimizer.state[weight]:
            adam_m1 = self.optimizer.state[weight]['exp_avg']
            adam_m2 = self.optimizer.state[weight]['exp_avg_sq']
            grad = adam_m1/(torch.sqrt(adam_m2) + 1e-08)
        elif 'momentum_buffer' in self.optimizer.state[weight]:
            grad = self.optimizer.state[weight]['momentum_buffer']
        return grad

    def get_gradient_for_weights(self, weight):
        grad = weight.grad.clone()
        return grad

    def print_nonzero_counts(self):
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name]
                num_nonzeros = (mask != 0).sum().item()

                ## compare pruned mask
                pre_masks_neg = self.pre_masks[name].data < 1.0
                pruned_masks_neg = self.pruned_masks[name].data < 1.0
                comp_1 = self.pre_masks[name].data.byte() & self.pruned_masks[name].data.byte()
                comp_2 = pre_masks_neg.byte() & pruned_masks_neg.byte()
                diff = self.pre_masks[name].numel()-(comp_1.sum().item() + comp_2.sum().item())

                val = '{0}: {1}->{2}, density: {3:.3f}, diff: {4}'.format(name, self.name2nonzeros[name], num_nonzeros, num_nonzeros/float(mask.numel()), diff)
                print(val)


        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue
                print('Death rate: {0}\n'.format(self.death_rate))
                break

    def fired_masks_update(self):
        ntotal_fired_weights = 0.0
        ntotal_weights = 0.0
        layer_fired_weights = {}
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                ## 曾经激活过的那些weights
                self.fired_masks[name] = self.masks[name].data.byte() | self.fired_masks[name].data.byte()
                ntotal_fired_weights += float(self.fired_masks[name].sum().item())
                ntotal_weights += float(self.fired_masks[name].numel())
                layer_fired_weights[name] = float(self.fired_masks[name].sum().item())/float(self.fired_masks[name].numel())
                print('Layerwise percentage of the fired weights of', name, 'is:', layer_fired_weights[name])
        total_fired_weights = ntotal_fired_weights/ntotal_weights
        print('The percentage of the total fired weights is:', total_fired_weights)
        return layer_fired_weights, total_fired_weights

    ### add function
    def death_decay_update(self, death_rate_decay=None, decay_flag=True):
        self.death_rate_decay = death_rate_decay
        self.decay_flag = decay_flag
