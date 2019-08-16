# coding=utf-8
import math
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn import Module, BatchNorm1d

import collections
from itertools import repeat

import numpy

normal_sampling = torch.tensor([-1.64761461, -1.45512392, -1.33302146, -1.24118489, -1.16649474,
    -1.10293101, -1.0472058 , -0.99731721, -0.95195146, -0.91019756, -0.87139658, -0.83505566,
    -0.80079598, -0.76831968, -0.73738809, -0.70780676, -0.67941502, -0.65207844, -0.62568332,
    -0.6001325 , -0.57534225, -0.55123978, -0.52776141, -0.50485098, -0.4824587 , -0.46054014,
    -0.43905545, -0.41796868, -0.39724726, -0.37686153, -0.35678436, -0.33699083, -0.31745792,
    -0.29816434, -0.27909025, -0.26021714, -0.24152764, -0.22300538, -0.20463492, -0.18640156,
    -0.16829133, -0.15029084, -0.13238723, -0.11456811, -0.09682146, -0.0791356 , -0.06149911,
    -0.04390081, -0.02632966, -0.00877475, 0.00877475,  0.02632966,  0.04390081,  0.06149911,
    0.0791356 , 0.09682146,  0.11456811,  0.13238723,  0.15029084,  0.16829133, 0.18640156,
    0.20463492,  0.22300538,  0.24152764,  0.26021714, 0.27909025,  0.29816434,  0.31745792,
    0.33699083,  0.35678436, 0.37686153,  0.39724726,  0.41796868,  0.43905545,  0.46054014,
    0.4824587 ,  0.50485098,  0.52776141,  0.55123978,  0.57534225, 0.6001325 ,  0.62568332,
    0.65207844,  0.67941502,  0.70780676, 0.73738809,  0.76831968,  0.80079598,  0.83505566,
    0.87139658, 0.91019756,  0.95195146,  0.99731721,  1.0472058 ,  1.10293101, 1.16649474,
    1.24118489,  1.33302146,  1.45512392,  1.64761461]).cuda() * numpy.sqrt(2.0) / 0.96

def center_function(f, mult=1.0):
    y = f(normal_sampling)
    mean = torch.mean(y).item()
    stddev = torch.sqrt(torch.mean((y - mean)**2)).item()
    print(mean, stddev)
    mult /= stddev
    def ff(x):
        return (f(x) - mean) *mult
    return ff

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)

def double_factorial(n):
     if n <= 1:
         return 1
     else:
         return n * double_factorial(n-2)

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, beta=0.1):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features).cuda())

        self.stdv = numpy.sqrt(1-beta**2) / math.sqrt(in_features*1.0)
        self.beta = beta

        if bias:
            self.bias = Parameter(torch.Tensor(out_features).cuda())
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.normal_(0.0, 1.0)#.uniform_(-alpha, alpha)
        if self.bias is not None:
            self.bias.data.normal_(0.0, 1.0)#.uniform_(-alpha, alpha)

    def forward(self, input):
        if self.bias is not None:
            return F.linear(input, self.weight) * self.stdv + self.bias * self.beta
        else:
            return F.linear(input, self.weight) * self.stdv

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class LinearNet(Module):
    def __init__(self, widths, non_lin=torch.relu, bias=True, beta=0.1, batch_norm=False):
        super(LinearNet, self).__init__()
        self.widths = widths
        self.depth = len(self.widths)-1
        self.non_lin = non_lin
        self.beta = beta
        self.batch_norm = batch_norm

        self.pre_alpha = [None for i in range(self.depth)]
        self.alpha = [None for i in range(self.depth)]

        self.linears = []
        for i in range(self.depth):
            lin = Linear(widths[i], widths[i+1], bias, beta)
            self.add_module('lin'+str(i).zfill(2), lin)
            self.linears += [lin]

        if self.batch_norm:
            self.bns = []
            for i in range(1,self.depth):
                bn = BatchNorm1d(widths[i], affine=False, eps=0.1)#, track_running_stats=False)
                self.add_module('bn'+str(i).zfill(2), bn)
                self.bns += [bn] 


    def reset_parameters(self):
        for l in self.linears:
            l.reset_parameters()

    def  forward(self, x):
        self.alpha[0] = x
        for i in range(self.depth-1):
            self.pre_alpha[i+1] = self.linears[i](self.alpha[i])
            #if self.batch_norm:
            #    self.pre_alpha[i+1] = self.bns[i](self.pre_alpha[i+1])
            self.alpha[i+1] = self.non_lin(self.pre_alpha[i+1])
            if self.batch_norm:
                self.alpha[i+1] = self.bns[i](self.alpha[i+1])

        return self.linears[self.depth-1](self.alpha[self.depth-1])

    def Sigma(self, i):
        return torch.matmul(self.alpha[i-1], torch.t(self.alpha[i-1])) / self.widths[i-1] + self.beta**2

    def Sigma_dot(self, i, retain_graph=False):
        alpha_dot = torch.autograd.grad(self.alpha[i-1].sum(), self.pre_alpha[i-1], retain_graph=retain_graph)[0]
        return torch.matmul(alpha_dot, torch.t(alpha_dot)) / self.widths[i-1]

    def Sigma_ddot(self, i, retain_graph=False):
        alpha_dot = torch.autograd.grad(self.alpha[i-1].sum(), self.pre_alpha[i-1], create_graph=True)[0]
        alpha_ddot = torch.autograd.grad(alpha_dot.sum(), self.pre_alpha[i-1], retain_graph=retain_graph)[0]
        return torch.matmul(alpha_ddot, torch.t(alpha_ddot)) / self.widths[i-1]

    def NTK(self, retain_graph=False):
        K = self.Sigma(1)
        for i in range(1, self.depth):
            K = K * self.Sigma_dot(i+1, retain_graph) + self.Sigma(i+1)
        return K

    def moments_S(self):
        NTK = self.Sigma(1)
        m2 = 0
        mom1 = self.alpha[1].clone().zero_()
        covar_m1 = torch.zeros([1,1]).cuda()
        move_m1 = torch.zeros([1,1]).cuda()

        for j in range(1, self.depth):
            alpha_dot = torch.autograd.grad(self.alpha[j].sum(), self.pre_alpha[j], create_graph=True)[0]
            alpha_ddot = torch.autograd.grad(alpha_dot.sum(), self.pre_alpha[j], create_graph=True)[0]
            alpha_dddot = torch.autograd.grad(alpha_ddot.sum(), self.pre_alpha[j], retain_graph=True)[0]
            
            Sigma = self.Sigma(j+1)
            Sigma_dot = torch.matmul(alpha_dot, torch.t(alpha_dot)) / self.widths[j]
            Sigma_ddot = torch.matmul(alpha_ddot, torch.t(alpha_ddot)) / self.widths[j]
            Mixed = torch.matmul(alpha_ddot, torch.t(self.alpha[j])) / self.widths[j]
            Mixed_dot = torch.matmul(alpha_dddot, torch.t(alpha_dot)) / self.widths[j]


            m2 = m2 * Sigma_dot + NTK*NTK*Sigma_ddot + 2 * NTK * Sigma_dot
            
            #var_m1 = var_m1*Sigma_dot + torch.diag(NTK)*torch.diag(NTK).unsqueeze(-1)*Sigma_ddot
            #mom1 = mom1 * alpha_dot + alpha_ddot * NTK.diag().view([-1, 1])

            move_m1 = move_m1 * Sigma_dot 
            move_m1 += NTK * (covar_m1.diag().view([-1, 1]) * Mixed_dot + covar_m1 * Sigma_ddot)
            move_m1 += covar_m1.diag().view([-1, 1]) * Mixed + covar_m1 * Sigma_dot
            move_m1 += NTK.diag().view([-1, 1]) * (NTK * Mixed_dot + Mixed)

            covar_m1 = covar_m1 * Sigma_dot + (covar_m1.diag() + NTK.diag()).view([-1, 1]) * Mixed
            NTK = NTK * Sigma_dot + Sigma
            '''
            if j < self.depth-1:
                mom1 = torch.matmul(mom1, self.linears[j].weight) / numpy.sqrt(self.widths[j])
            else:
                var_m1 = torch.matmul(mom1, torch.t(mom1)) / self.widths[j]
                covar_mm1 = torch.matmul(self.alpha[j], torch.t(mom1)) / self.widths[j]
            '''
        return (covar_m1, move_m1), m2



'''
class Bilinear(Module):
    def __init__(self, in1_features, in2_features, out_features, bias=True):
        super(Bilinear, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in1_features, in2_features))

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input1, input2):
        return F.bilinear(input1, input2, self.weight, self.bias)

    def extra_repr(self):
        return 'in1_features={}, in2_features={}, out_features={}, bias={}'.format(
            self.in1_features, self.in2_features, self.out_features, self.bias is not None
        )
'''

def make_mask(w,s):
    #return 1 / numpy.sqrt(w)
    mask = torch.Tensor(w).cuda().zero_() + 1
    for i in range(1, w // s + 1):
        mask[s*i:w] += 1
        mask[0:w-s*i] += 1
    mask = 1 / mask
    mask = mask / mask.sum()
    return mask.sqrt()

def make_mask_transpose(w,s):
    #return numpy.sqrt(s / w)
    mask = torch.Tensor(w).cuda().zero_() + 1
    for i in range(1, w // s + 1):
        mask[s*i:w] += 1
        mask[0:w-s*i] += 1
    return 1 / mask.sqrt()
    
class _ConvNd(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias, beta):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.beta = beta
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size).cuda())
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size).cuda())
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels).cuda())
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.normal_(0.0, 1.0) # .uniform_(-alpha, alpha)
        if self.bias is not None:
            self.bias.data.normal_(0.0, 1.0)#.uniform_(-alpha, alpha)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


class Conv1d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, beta=0.1):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)

        self.stdv = 1. / math.sqrt(in_channels*1.0)
        self.mask = make_mask(kernel_size, stride)

        super(Conv1d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _single(0), groups, bias, beta)

    def forward(self, input):
        mask = self.mask.cuda(input.device)
        return F.conv1d(input*self.stdv, mask*self.weight, self.bias*self.beta, self.stride,
                        self.padding, self.dilation, self.groups)



class Conv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, beta=0.1):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        self.stdv = 1. / math.sqrt(in_channels*1.0)
        self.mask = make_mask(kernel_size[0], stride[0]).unsqueeze(-1)*make_mask(kernel_size[1], stride[1])
        #self.mask = make_mask(kernel_size[0], stride[0])*make_mask(kernel_size[1], stride[1])

        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, beta)

    def forward(self, input):
        mask = self.mask.cuda(input.device)
        return F.conv2d(input*self.stdv, mask*self.weight, self.bias*self.beta, self.stride,
                        self.padding, self.dilation, self.groups)



class Conv3d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, beta=0.1):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)

        self.stdv = 1. / math.sqrt(in_channels*1.0)
        self.mask = make_mask(kernel_size[0], stride[0]).unsqueeze(-1)*make_mask(kernel_size[1], stride[1])
        self.mask = self.mask.unsqueeze(-1)*make_mask(kernel_size[2], stride[2])

        super(Conv3d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _triple(0), groups, bias, beta)

    def forward(self, input):
        return F.conv3d(input*self.stdv, self.mask*self.weight, self.bias*self.beta, self.stride,
                        self.padding, self.dilation, self.groups)


class _ConvTransposeNd(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias, beta):
        super(_ConvTransposeNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.beta = beta
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size).cuda())
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size).cuda())
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels).cuda())
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.normal_(0.0, 1.0)
        if self.bias is not None:
            self.bias.data.normal_(0.0, 1.0) # .zero_()

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

def group_coord_pool2d(x, kernel_size, stride=None, group_size=4):
    sh = x.shape
    x = x.view([sh[0], -1, group_size, sh[2], sh[3]])
    logits = torch.exp(x[:, :, 0:1, :, :])
    x = x * logits
    partition = F.avg_pool2d(logits.squeeze(2), kernel_size, stride)
    y = F.avg_pool2d(x.view(sh), kernel_size, stride)
    sh = y.shape
    y = y.view([sh[0], -1, group_size, sh[2], sh[3]])
    return (y / partition.unsqueeze(2)).view([sh[0], -1, sh[2], sh[3]])



class ConvTranspose1d(_ConvTransposeNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, output_padding=0, groups=1, bias=True, beta=0.1):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        output_padding = _single(output_padding)

        self.stdv = 1. / math.sqrt(in_channels)
        self.mask = make_mask_transpose(kernel_size[0], stride[0])

        super(ConvTranspose1d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, output_padding, groups, bias, beta)

    def forward(self, input, output_size=None):
        #output_padding = self._output_padding(input, output_size)
        return F.conv_transpose1d(
            input*self.stdv, self.weight*self.mask, self.bias*self.beta, self.stride, self.padding,
            self.output_padding, self.groups, self.dilation)



class ConvTranspose2d(_ConvTransposeNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, output_padding=0, groups=1, bias=True, beta=0.1):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        output_padding = _pair(output_padding)

        self.stdv = 1. / math.sqrt(in_channels)
        self.mask = make_mask_transpose(kernel_size[0], stride[0]).unsqueeze(-1)*make_mask_transpose(kernel_size[1], stride[1])
        #self.mask = make_mask_transpose(kernel_size[0], stride[0])*make_mask_transpose(kernel_size[1], stride[1])

        super(ConvTranspose2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, output_padding, groups, bias, beta)

    def forward(self, input, output_size=None):
        #output_padding = self._output_padding(input, output_size)
        mask = self.mask.cuda(input.device)
        return F.conv_transpose2d(
            input*self.stdv, self.weight*mask, self.bias*self.beta, self.stride, self.padding,
            self.output_padding, self.groups, self.dilation)


class ConvTranspose3d(_ConvTransposeNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, output_padding=0, groups=1, bias=True, beta=0.1):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        output_padding = _triple(output_padding)


        self.stdv = 1. / math.sqrt(in_channels)
        self.mask = make_mask_transpose(kernel_size[0], stride[0]).unsqueeze(-1)*make_mask_transpose(kernel_size[1], stride[1])
        self.mask = self.mask.unsqueeze(-1)*make_mask(kernel_size[2], stride[2])

        super(ConvTranspose3d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, output_padding, groups, bias, beta)

    def forward(self, input, output_size=None):
        #output_padding = self._output_padding(input, output_size)
        return F.conv_transpose3d(
            input*self.stdv, self.weight*self.mask, self.bias*self.beta, self.stride, self.padding,
            self.output_padding, self.groups, self.dilation)
