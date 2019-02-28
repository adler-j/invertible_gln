import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.io


def parameters(conv):
    n = 'data/theta_taylor' #, 'theta_taylor_single', 'theta_taylor'
    mat = scipy.io.loadmat(n + '.mat')
    theta = mat['theta'].squeeze()
    W = conv.weight.detach().numpy()

    nrm = np.sum(np.abs(W))
    if nrm > 15:
        print('Overflow likely')

    m = np.arange(1, len(theta) + 1)
    vals = m * np.ceil(nrm / theta)
    mstar = min(1 + np.argmin(vals), 56)
    s = int(np.ceil(nrm / theta[mstar - 1]))

    return mstar, s


def maxnorm(x):
    return torch.max(torch.abs(x))

def exponential(conv, vec, t=1, tol=1e-16):
    """Simple implementation of e^net * vec."""
    # Naming according to paper
    A = conv
    B = vec

    mstar, s = parameters(conv)

    F = B
    for i in range(1, s + 1):
        c1 = maxnorm(B)
        for j in range(1, mstar + 1):
            B = t * A(B) / (s * j)
            c2 = maxnorm(B)
            F = F + B
            if c1 + c2 <= tol * maxnorm(F):
                break
                # return F
            c1 = c2
        B = F
    return F


class Conv2d(nn.Module):
    def __init__(self, channels, kernel_size):
        super(PixelShuffle, self).__init__()

        if kernel_size % 2 == 0:
            raise Exception('kernel must be odd')
        else:
            padding = kernel_size // 2

        self.conv = nn.Conv2d(channels, channels, kernel_size, stride=1,
                              padding=padding, bias=False)
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        return exponential(self.conv, x) + self.bias

    @property
    def inverse(self):
        forward = self

        class Conv2dInverse(nn.Module):
            def forward(self, x):
                return exponential(forward.conv, x - forward.bias, t=-1)



class LeakyRelu(nn.LeakyReLU):
    @property
    def inverse(self):
        return LeakyRelu(1 / self.negative_slop, self.inplace)


class PixelShuffle(nn.Module):
    def __init__(self, block_size):
        super(PixelShuffle, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def forward(self, x):
        output = x.permute(0, 2, 3, 1)
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 2)
        stack = [t_t.contiguous().view(batch_size, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 1, 3)
        output = output.permute(0, 3, 1, 2)
        return output.contiguous()

    @property
    def inverse(self):
        forward = self

        class InvertibleConv2dInverse(nn.Module):
            def forward(_, x):
                output = x.permute(0, 2, 3, 1)
                (batch_size, d_height, d_width, d_depth) = output.size()
                s_depth = int(d_depth / forward.block_size_sq)
                s_width = int(d_width * forward.block_size)
                s_height = int(d_height * forward.block_size)
                t_1 = output.contiguous().view(batch_size, d_height, d_width, forward.block_size_sq, s_depth)
                spl = t_1.split(forward.block_size, 3)
                stack = [t_t.contiguous().view(batch_size, d_height, s_width, s_depth) for t_t in spl]
                output = torch.stack(stack, 0).transpose(0, 1).permute(0, 2, 1, 3, 4).contiguous().view(batch_size, s_height, s_width, s_depth)
                output = output.permute(0, 3, 1, 2)
                return output.contiguous()

