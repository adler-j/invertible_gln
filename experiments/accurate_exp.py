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

# Create phantom
image = np.zeros([50, 50])
image[25, 25] = 1

# Apply e^A to x, then compute inverse (e^-A)
inp = torch.from_numpy(image[None, None].astype('float32'))

conv = nn.Conv2d(1, 1,
                 kernel_size=(3, 3),
                 stride=1,
                 padding=1,
                 bias=False)
torch.nn.init.xavier_uniform_(conv.weight)
conv.weight.data *= 3

result = exponential(conv, inp)
inverse = exponential(conv, result, t=-1)

# Reconver results and display.
out = result.detach().numpy()[0, 0]
out_inv = inverse.detach().numpy()[0, 0]

plt.figure('image')
plt.imshow(image)
plt.colorbar()

plt.figure('out')
plt.imshow(out)
plt.colorbar()

plt.figure('out_inv')
plt.imshow(out_inv)
plt.colorbar()

plt.figure('image - out_inv')
plt.imshow(image - out_inv)
plt.colorbar()

plt.figure('out - image')
plt.imshow(out - image)
plt.colorbar()