import itertools
from torch.optim.optimizer import required
from torch.optim import SGD
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class Shortcut(nn.Module):
    def __init__(self, stride, in_planes, expansion, planes, total_task):
        super(Shortcut, self).__init__()
        self.identity = True
        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != expansion*planes:
            self.identity = False
            self.conv1 = nn.Conv2d(in_planes, expansion*planes, kernel_size=1, stride=stride, bias=False)
            self.bn1 = nn.ModuleList()
            for _ in range(total_task):
                self.bn1.append(nn.BatchNorm2d(expansion*planes))

    def forward(self, t, x):
        if self.identity:
            out = self.shortcut(x)
        else:
            out = self.conv1(x)
            out = self.bn1[t](out)
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, pooling=False, total_task=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.bn1 = nn.ModuleList()
        for _ in range(total_task):
            self.bn1.append(nn.BatchNorm2d(planes))
        # self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.ModuleList()
        for _ in range(total_task):
            self.bn2.append(nn.BatchNorm2d(planes))
        # self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = Shortcut(stride, in_planes, self.expansion, planes, total_task)

        self.gate = torch.sigmoid
        self.ec1 = nn.ParameterList()
        self.ec2 = nn.ParameterList()
        for _ in range(total_task):
            self.ec1.append(nn.Parameter(torch.randn(1, planes)))
            self.ec2.append(nn.Parameter(torch.randn(1, planes)))
        # self.ec1 = nn.Embedding(5, planes)
        # self.ec2 = nn.Embedding(5, planes)

        self.pooling = pooling


    def mask(self, t, s=1):
        gc1 = self.gate(s * self.ec1[t])
        gc2 = self.gate(s * self.ec2[t])
        return [gc1, gc2]

    def mask_out(self, out, mask):
        mask = mask.view(1, -1, 1, 1)
        out = out * mask.expand_as(out)
        return out

    def forward(self, t, x, msk, s):
        masks = self.mask(t, s=s)
        gc1, gc2 = masks

        msk.append(masks)

        out = F.relu(self.bn1[t](self.conv1(x)))
        out = self.mask_out(out, gc1)

        out = self.bn2[t](self.conv2(out))
        out += self.shortcut(t, x)
        out = F.relu(out)

        if self.pooling:
            out = F.avg_pool2d(out, 4)

        out = self.mask_out(out, gc2)
        return t, out, msk, s


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2, total_classes=100):
        last_dim = 1024 * block.expansion
        super(ResNet, self).__init__()

        self.total_task = total_classes // num_classes

        self.in_planes = 128
        self.last_dim = last_dim

        # self.normalize = NormalizeLayer()

        self.conv1 = conv3x3(3, 128)
        self.bn1 = nn.ModuleList()
        for _ in range(self.total_task):
            self.bn1.append(nn.BatchNorm2d(128))

        self.ec0 = nn.ParameterList()
        for _ in range(self.total_task):
            self.ec0.append(nn.Parameter(torch.randn(1, 128)))
        # self.ec0 = nn.Embedding(5, 64)

        self.gate = torch.sigmoid

        self.layer1 = self._make_layer(block, 128, num_blocks[0], stride=1, pooling=False, total_task=self.total_task)
        self.layer2 = self._make_layer(block, 256, num_blocks[1], stride=2, pooling=False, total_task=self.total_task) # LAYER2.0.SHORTCUT
        self.layer3 = self._make_layer(block, 512, num_blocks[2], stride=2, pooling=False, total_task=self.total_task) # LAYER3.0.SHORTCUT
        self.layer4 = self._make_layer(block, 1024, num_blocks[3], stride=2, pooling=True, total_task=self.total_task) # LAYER4.0.SHORTCUT

        self.last = nn.ModuleList()
        for _ in range(self.total_task):
            self.last.append(nn.Linear(1024, num_classes))

    def _make_layer(self, block, planes, num_blocks, stride, pooling=False, total_task=None):
        strides = [stride] + [1]*(num_blocks-1)
        pooling_ = False
        layers = nn.ModuleList()
        for i, stride in enumerate(strides):
            if i == len(strides) - 1:
                pooling_ = pooling
            layers.append(block(self.in_planes, planes, stride, pooling_, total_task))
            self.in_planes = planes * block.expansion
        return layers
        # layers = []
        # for stride in strides:
        #     layers.append(block(self.in_planes, planes, stride))
        #     self.in_planes = planes * block.expansion
        # return nn.Sequential(*layers)

    def mask(self, t, s=1):
        gc1 = self.gate(s * self.ec0[t])
        return gc1

    def mask_out(self, out, mask):
        mask = mask.view(1, -1, 1, 1)
        out = out * mask.expand_as(out)
        return out

    def forward(self, t, x, s=1):
        out_list = []
        msk = []

        gc0 = self.mask(t, s=s)
        msk.append(gc0)

        # x = self.normalize(x)
        x = self.conv1(x)
        x = self.bn1[t](x)
        x = F.relu(x)
        x = self.mask_out(x, gc0)

        for op in self.layer1:
            t, x, msk, s = op(t, x, msk, s)
        # out = self.layer1(out)

        for op in self.layer2:
            t, x, msk, s = op(t, x, msk, s)
        # out = self.layer2(out)

        for op in self.layer3:
            t, x, msk, s = op(t, x, msk, s)
        # out = self.layer3(out)

        for op in self.layer4:
            t, x, msk, s = op(t, x, msk, s) # the output x is (100, 512, 1, 1)
        # out = self.layer4(out)

        # out = F.avg_pool2d(x, 4)
        out = x.view(x.size(0), -1)

        out = self.last[t](out)
        # y = []
        # for c in range(10):
        #     y.append(self.last[c](out))

        return out, list(itertools.chain(*msk))
        # return y, list(itertools.chain(*msk))


def ResNet18(num_classes, total_classes):
    return ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes, total_classes=total_classes)

def HAT_reg(args, masks):
    """ masks and self.mask_pre must have values in the same order """
    reg, count = 0., 0.
    if args.mask_pre is not None:
        for m, mp in zip(masks, args.mask_pre.values()):
            aux = 1. - mp
            reg += (m * aux).sum()
            count += aux.sum()
    else:
        for m in masks:
            reg += m.sum()
            count += np.prod(m.size()).item()
    reg /= count
    return args.reg_lambda * reg

def cum_mask(smax, t, model, mask_pre, accelerator):
    """ 
        Keep track of mask values. 
        This will be used later as a regularizer in the optimization
    """
    try:
        model = model.module
    except AttributeError:
        model = model

    task_id = torch.tensor([t]).to(accelerator.device)
    mask = {}
    for n, _ in model.named_parameters():
        names = n.split('.')
        checker = [i for i in ['ec0', 'ec1', 'ec2'] if i in names]
        if names[0] == 'module':
            names = names[1:]
        if checker:
            if 'layer' in n:
                gc1, gc2 = model.__getattr__(names[0])[int(names[1])].mask(task_id, s=smax)
                if checker[0] == 'ec1':
                    n = '.'.join(n.split('.')[:-1]) # since n is like layer2.0.ec1.8, where the last number 8 indicates task id
                    mask[n] = gc1.detach()
                    mask[n].requires_grad = False
                elif checker[0] == 'ec2':
                    n = '.'.join(n.split('.')[:-1])
                    mask[n] = gc2.detach()
                    mask[n].requires_grad = False
                # elif 'down_sample' in n:
                #     mask[n] = self.model.__getattr__(names[0]).down_sample.mask(t, s=self.smax).detach()
                #     mask[n].requires_grad = False

            elif checker[0] == 'ec0':
                n = '.'.join(n.split('.')[:-1])
                mask[n] = model.mask(task_id, smax).detach()
                mask[n].requires_grad = False

    if mask_pre is None:
        mask_pre = {}
        for n in mask.keys():
            mask_pre[n] = mask[n]
    else:
        for n in mask.keys():
            mask_pre[n] = torch.max(mask_pre[n], mask[n])
    return mask_pre

def freeze_mask(P, t, model, mask_pre):
    """
        Eq (2) in the paper. self.mask_back is a dictionary whose keys are
        the convolutions' parameter names. Each value of a key is a matrix, whose elements are
        approximately binary.
    """
    try:
        model = model.module
    except AttributeError:
        model = model

    mask_back = {}
    for n, p in model.named_parameters():
        names = n.split('.')
        if 'layer' not in names[0]:
            if n == 'conv1.weight':
                mask_back[n] = 1 - mask_pre['ec0'].data.view(-1, 1, 1, 1).expand_as(p)
        # elif 'layer' in names[0]:
        elif 'layer1' in n:
            if n == 'layer1.0.conv1.weight':
                post = mask_pre['layer1.0.ec1'].data.view(-1, 1, 1, 1).expand_as(p)
                pre  = mask_pre['ec0'].data.view(1, -1, 1, 1).expand_as(p)
                mask_back[n] = 1 - torch.min(post, pre)
            elif n == 'layer1.0.conv2.weight':
                post = mask_pre['layer1.0.ec2'].data.view(-1, 1, 1, 1).expand_as(p)
                pre  = mask_pre['layer1.0.ec1'].data.view(1, -1, 1, 1).expand_as(p)
                mask_back[n] = 1 - torch.min(post, pre)

            elif n == 'layer1.1.conv1.weight':
                post = mask_pre['layer1.1.ec1'].data.view(-1, 1, 1, 1).expand_as(p)
                pre  = mask_pre['layer1.0.ec2'].data.view(1, -1, 1, 1).expand_as(p)
                mask_back[n] = 1 - torch.min(post, pre)
            elif n == 'layer1.1.conv2.weight':
                post = mask_pre['layer1.1.ec2'].data.view(-1, 1, 1, 1).expand_as(p)
                pre  = mask_pre['layer1.1.ec1'].data.view(1, -1, 1, 1).expand_as(p)
                mask_back[n] = 1 - torch.min(post, pre)

        elif 'layer2' in n:
            if n == 'layer2.0.conv1.weight':
                post = mask_pre['layer2.0.ec1'].data.view(-1, 1, 1, 1).expand_as(p)
                pre  = mask_pre['layer1.1.ec2'].data.view(1, -1, 1, 1).expand_as(p)
                mask_back[n] = 1 - torch.min(post, pre)
            elif n == 'layer2.0.conv2.weight':
                post = mask_pre['layer2.0.ec2'].data.view(-1, 1, 1, 1).expand_as(p)
                pre  = mask_pre['layer2.0.ec1'].data.view(1, -1, 1, 1).expand_as(p)
                mask_back[n] = 1 - torch.min(post, pre)

            elif n == 'layer2.0.shortcut.conv1.weight':
                post = mask_pre['layer2.0.ec2'].data.view(-1, 1, 1, 1).expand_as(p)
                pre  = mask_pre['layer1.1.ec2'].data.view(1, -1, 1, 1).expand_as(p)
                mask_back[n] = 1 - torch.min(post, pre)

            elif n == 'layer2.1.conv1.weight':
                post = mask_pre['layer2.1.ec1'].data.view(-1, 1, 1, 1).expand_as(p)
                pre  = mask_pre['layer2.0.ec2'].data.view(1, -1, 1, 1).expand_as(p)
                mask_back[n] = 1 - torch.min(post, pre)
            elif n == 'layer2.1.conv2.weight':
                post = mask_pre['layer2.1.ec2'].data.view(-1, 1, 1, 1).expand_as(p)
                pre  = mask_pre['layer2.1.ec1'].data.view(1, -1, 1, 1).expand_as(p)
                mask_back[n] = 1 - torch.min(post, pre)

        elif 'layer3' in n:
            if n == 'layer3.0.conv1.weight':
                post = mask_pre['layer3.0.ec1'].data.view(-1, 1, 1, 1).expand_as(p)
                pre  = mask_pre['layer2.1.ec2'].data.view(1, -1, 1, 1).expand_as(p)
                mask_back[n] = 1 - torch.min(post, pre)
            elif n == 'layer3.0.conv2.weight':
                post = mask_pre['layer3.0.ec2'].data.view(-1, 1, 1, 1).expand_as(p)
                pre  = mask_pre['layer3.0.ec1'].data.view(1, -1, 1, 1).expand_as(p)
                mask_back[n] = 1 - torch.min(post, pre)

            elif n == 'layer3.0.shortcut.conv1.weight':
                post = mask_pre['layer3.0.ec2'].data.view(-1, 1, 1, 1).expand_as(p)
                pre  = mask_pre['layer2.1.ec2'].data.view(1, -1, 1, 1).expand_as(p)
                mask_back[n] = 1 - torch.min(post, pre)

            elif n == 'layer3.1.conv1.weight':
                post = mask_pre['layer3.1.ec1'].data.view(-1, 1, 1, 1).expand_as(p)
                pre  = mask_pre['layer3.0.ec2'].data.view(1, -1, 1, 1).expand_as(p)
                mask_back[n] = 1 - torch.min(post, pre)
            elif n == 'layer3.1.conv2.weight':
                post = mask_pre['layer3.1.ec2'].data.view(-1, 1, 1, 1).expand_as(p)
                pre  = mask_pre['layer3.1.ec1'].data.view(1, -1, 1, 1).expand_as(p)
                mask_back[n] = 1 - torch.min(post, pre)

        elif 'layer4' in n:
            if n == 'layer4.0.conv1.weight':
                post = mask_pre['layer4.0.ec1'].data.view(-1, 1, 1, 1).expand_as(p)
                pre  = mask_pre['layer3.1.ec2'].data.view(1, -1, 1, 1).expand_as(p)
                mask_back[n] = 1 - torch.min(post, pre)
            elif n == 'layer4.0.conv2.weight':
                post = mask_pre['layer4.0.ec2'].data.view(-1, 1, 1, 1).expand_as(p)
                pre  = mask_pre['layer4.0.ec1'].data.view(1, -1, 1, 1).expand_as(p)
                mask_back[n] = 1 - torch.min(post, pre)

            elif n == 'layer4.0.shortcut.conv1.weight':
                post = mask_pre['layer4.0.ec2'].data.view(-1, 1, 1, 1).expand_as(p)
                pre  = mask_pre['layer3.1.ec2'].data.view(1, -1, 1, 1).expand_as(p)
                mask_back[n] = 1 - torch.min(post, pre)

            elif n == 'layer4.1.conv1.weight':
                post = mask_pre['layer4.1.ec1'].data.view(-1, 1, 1, 1).expand_as(p)
                pre  = mask_pre['layer4.0.ec2'].data.view(1, -1, 1, 1).expand_as(p)
                mask_back[n] = 1 - torch.min(post, pre)
            elif n == 'layer4.1.conv2.weight':
                post = mask_pre['layer4.1.ec2'].data.view(-1, 1, 1, 1).expand_as(p)
                pre  = mask_pre['layer4.1.ec1'].data.view(1, -1, 1, 1).expand_as(p)
                mask_back[n] = 1 - torch.min(post, pre)
    return mask_back

def compensation(model, args, thres_cosh=50, s=1):
    """ Equation before Eq. (4) in the paper """
    for n, p in model.named_parameters():
        if 'ec' in n:
            if p.grad is not None:
                num = torch.cosh(torch.clamp(s * p.data, -thres_cosh, thres_cosh)) + 1
                den = torch.cosh(p.data) + 1
                p.grad *= args.smax / s * num / den

class SGD_hat(SGD):
    def __init__(self, params, lr=required, momentum=0, dampening=0, 
                 weight_decay=0, nesterov=False):
        super(SGD_hat, self).__init__(params, lr, momentum, dampening,
                                      weight_decay, nesterov)

    @torch.no_grad()
    def step(self, closure=None, hat=False):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                # temp = torch.ones(d_p.size()).to(d_p.device)
                # if len(d_p.size()) > 1:
                #     # temp = torch.sum(d_p, dim=1).detach()
                #     temp = d_p.detach()
                #     temp = torch.tensor(temp > 1e-30, dtype=torch.int) + torch.tensor(temp < -1e-30, dtype=torch.int)
                #     temp = temp.to(d_p.device)
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                if hat:
                    if p.hat is not None:
                        d_p = d_p * p.hat
                # print(d_p.size(), temp.size())
                # print(temp.expand_as(d_p), temp.expand_as(d_p).size())
                # d_p = d_p * temp#.expand_as(d_p)

                p.add_(d_p, alpha=-group['lr'])

        return loss

def compensation_clamp(model, thres_emb=6):
    # Constrain embeddings
    for n, p in model.named_parameters():
        if 'ec' in n:
            if p.grad is not None:
                p.data.copy_(torch.clamp(p.data, -thres_emb, thres_emb))


import math

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.optimizer import required


def warmup_cosine(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 0.5 * (1.0 + torch.cos(math.pi * x))


def warmup_constant(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return max((x - 1.) / (warmup - 1.), 0)


SCHEDULES = {
    'warmup_cosine': warmup_cosine,
    'warmup_constant': warmup_constant,
    'warmup_linear': warmup_linear,
}


class Adam(Optimizer):

    def __init__(self, params, lr=required, warmup=-1, t_total=-1, schedule='warmup_linear', b1=0.9, b2=0.999, e=1e-6,
                 weight_decay=0.01, max_grad_norm=1.0):
        if lr is not required and lr < 0.0:
            raise ValueError(
                "Invalid learning rate: {} - should be >= 0.0".format(lr))
        if schedule not in SCHEDULES:
            raise ValueError("Invalid schedule parameter: {}".format(schedule))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError(
                "Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))
        if not 0.0 <= b1 < 1.0:
            raise ValueError(
                "Invalid b1 parameter: {} - should be in [0.0, 1.0[".format(b1))
        if not 0.0 <= b2 < 1.0:
            raise ValueError(
                "Invalid b2 parameter: {} - should be in [0.0, 1.0[".format(b2))
        if not e >= 0.0:
            raise ValueError(
                "Invalid epsilon value: {} - should be >= 0.0".format(e))
        defaults = dict(lr=lr, schedule=schedule, warmup=warmup, t_total=t_total,
                        b1=b1, b2=b2, e=e, weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm)
        super(Adam, self).__init__(params, defaults)

    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(
                        state['step'] / group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']
                lr.append(lr_scheduled)
        return lr

    def step(self, closure=None, hat=False, **kwargs):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        # other argument --------------
        if 'type' in kwargs:
            type = kwargs['type']
        if 't' in kwargs:
            t = kwargs['t']
        if 'mask_back' in kwargs:
            mask_back = kwargs['mask_back']
        # other argument --------------
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p_id, p in enumerate(group['params']):

                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['next_m'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['next_v'] = torch.zeros_like(p.data)

                next_m, next_v = state['next_m'], state['next_v']
                beta1, beta2 = group['b1'], group['b2']

                # Add grad clipping
                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(p, group['max_grad_norm'])

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                next_m.mul_(beta1).add_(grad, alpha=1-beta1)
                next_v.mul_(beta2).addcmul_(grad, grad.conj(), value=1-beta2)
                update = next_m / (next_v.sqrt() + group['e'])

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.

                if group['weight_decay'] > 0.0:
                    update += group['weight_decay'] * p.data

                # -----------------------------------block the grad even for adam
                if hat:
                    if p.hat is not None:
                        update = update * p.hat
                # -----------------------------------block the grad even for adam

                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(
                        state['step'] / group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']

                update_with_lr = lr_scheduled * update

                p.data.add_(-update_with_lr)

                state['step'] += 1

        return 