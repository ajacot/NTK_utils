# coding=utf-8
import math
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn import Module

import collections
from itertools import repeat

def center_function(f, mult=1.0):
    x = torch.normal(torch.zeros([200])).cuda()
    y = f(x)
    mean = torch.mean(y)
    stddev = torch.sqrt(torch.mean((y - mean)**2))
    print(mean, stddev)
    mult /= stddev
    def ff(x):
        return (f(x) - mean) *mult
    return ff

alpha = 1.5**(1.0 / 3.0)

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

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, beta=0.1):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))

        self.stdv = 1. / math.sqrt(in_features*1.0)
        self.beta = beta
        
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
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
        #return F.linear(input, self.weight * self.stdv, self.bias * self.beta)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class LinearNet(Module):
    def __init__(self, widths, non_lin=torch.relu, bias=True, beta=0.1):
        super(LinearNet, self).__init__()
        self.widths = widths
        self.depth = len(self.widths)-1
        self.non_lin = non_lin
        self.beta = beta

        self.pre_alpha = [None for i in range(self.depth)]
        self.alpha = [None for i in range(self.depth)]

        self.linears = []
        for i in range(self.depth):
            lin = Linear(widths[i], widths[i+1], bias, beta)
            self.add_module('lin'+str(i).zfill(2), lin)
            self.linears += [lin]

    def reset_parameters(self):
        for l in self.linears:
            l.reset_parameters()

    def  forward(self, x):
        self.alpha[0] = x
        for i in range(self.depth-1):
            self.pre_alpha[i+1] = self.linears[i](self.alpha[i])
            self.alpha[i+1] = self.non_lin(self.pre_alpha[i+1])

        return self.linears[self.depth-1](self.alpha[self.depth-1])

    def Sigma(self, i):
        return torch.matmul(self.alpha[i-1], torch.t(self.alpha[i-1])) / self.widths[i-1] + self.beta**2

    def Sigma_dot(self, i):
        alpha_dot = torch.autograd.grad(self.alpha[i-1].sum(), self.pre_alpha[i-1])[0]
        return torch.matmul(alpha_dot, torch.t(alpha_dot)) / self.widths[i-1]

    def NTK(self):
        K = self.Sigma(1)
        for i in range(1, self.depth):
            K = K * self.Sigma_dot(i+1) + self.Sigma(i+1)
        return K

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
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.uniform_(-alpha, alpha)
        if self.bias is not None:
            self.bias.data.zero_()#.uniform_(-alpha, alpha)

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
        
        n = in_channels
        for k in kernel_size:
            n *= k

        self.stdv = 1. / math.sqrt(n*1.0)
        
        super(Conv1d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _single(0), groups, bias, beta)

    def forward(self, input):
        return F.conv1d(input*self.stdv, self.weight, self.bias*self.beta, self.stride,
                        self.padding, self.dilation, self.groups)



class Conv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, beta=0.1):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        
        n = in_channels
        for k in kernel_size:
            n *= k

        self.stdv = 1. / math.sqrt(n*1.0)
        
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, beta)

    def forward(self, input):
        return F.conv2d(input*self.stdv, self.weight, self.bias*self.beta, self.stride,
                        self.padding, self.dilation, self.groups)



class Conv3d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, beta=0.1):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)

        n = in_channels
        for k in kernel_size:
            n *= k

        self.stdv = 1. / math.sqrt(n*1.0)
        
        super(Conv3d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _triple(0), groups, bias, beta)

    def forward(self, input):
        return F.conv3d(input*self.stdv, self.weight, self.bias*self.beta, self.stride,
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
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.uniform_(-alpha, alpha)
        if self.bias is not None:
            self.bias.data.uniform_(-alpha, alpha) # .zero_()

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


'''
class _ConvTransposeMixin(object):
    def forward(self, input, output_size=None):
        output_padding = self._output_padding(input, output_size)
        func = self._backend.ConvNd(
            self.stride, self.padding, self.dilation, self.transposed,
            output_padding, self.groups)
        if self.bias is None:
            return func(input, self.weight)
        else:
            return func(input, self.weight, self.bias)

    def _output_padding(self, input, output_size):
        if output_size is None:
            return self.output_padding

        output_size = list(output_size)
        k = input.dim() - 2
        if len(output_size) == k + 2:
            output_size = output_size[-2:]
        if len(output_size) != k:
            raise ValueError(
                "output_size must have {} or {} elements (got {})"
                .format(k, k + 2, len(output_size)))

        def dim_size(d):
            return ((input.size(d + 2) - 1) * self.stride[d] -
                    2 * self.padding[d] + self.kernel_size[d])

        min_sizes = [dim_size(d) for d in range(k)]
        max_sizes = [min_sizes[d] + self.stride[d] - 1 for d in range(k)]
        for size, min_size, max_size in zip(output_size, min_sizes, max_sizes):
            if size < min_size or size > max_size:
                raise ValueError((
                    "requested an output size of {}, but valid sizes range "
                    "from {} to {} (for an input of {})").format(
                        output_size, min_sizes, max_sizes, input.size()[2:]))

        return tuple([output_size[d] - min_sizes[d] for d in range(k)])
'''

class ConvTranspose1d(_ConvTransposeNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, output_padding=0, groups=1, bias=True, beta=0.1):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        output_padding = _single(output_padding)

        n = in_channels
        for k in kernel_size:
            n *= k
        for s in stride:
            n /= s*1.0

        self.stdv = 1. / math.sqrt(n)

        super(ConvTranspose1d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, output_padding, groups, bias, beta)

    def forward(self, input, output_size=None):
        #output_padding = self._output_padding(input, output_size)
        return F.conv_transpose1d(
            input*self.stdv, self.weight, self.bias*self.beta, self.stride, self.padding,
            self.output_padding, self.groups, self.dilation)



class ConvTranspose2d(_ConvTransposeNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, output_padding=0, groups=1, bias=True, beta=0.1):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        output_padding = _pair(output_padding)

        n = in_channels
        for k in kernel_size:
            n *= k
        for s in stride:
            n /= s*1.0

        self.stdv = 1. / math.sqrt(n)

        super(ConvTranspose2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, output_padding, groups, bias, beta)

    def forward(self, input, output_size=None):
        #output_padding = self._output_padding(input, output_size)
        return F.conv_transpose2d(
            input*self.stdv, self.weight, self.bias*self.beta, self.stride, self.padding,
            self.output_padding, self.groups, self.dilation)


class ConvTranspose3d(_ConvTransposeNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, output_padding=0, groups=1, bias=True, beta=0.1):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        output_padding = _triple(output_padding)

        n = in_channels
        for k in kernel_size:
            n *= k
        for s in stride:
            n /= s*1.0

        self.stdv = 1. / math.sqrt(n)

        super(ConvTranspose3d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, output_padding, groups, bias, beta)

    def forward(self, input, output_size=None):
        #output_padding = self._output_padding(input, output_size)
        return F.conv_transpose3d(
            input*self.stdv, self.weight, self.bias*self.beta, self.stride, self.padding,
            self.output_padding, self.groups, self.dilation)


