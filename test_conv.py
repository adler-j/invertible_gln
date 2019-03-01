import invertible_layers as il
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

# Create phantom
image = np.zeros([50, 50])
image[25, 25] = 1

# Apply e^A to x, then compute inverse (e^-A)
inp = torch.from_numpy(image[None, None].astype('float32'))

conv = il.Conv2d(1, 5, orth=True)

out_pt = conv(inp)
out_inv_pt = conv.inverse(out_pt)

out = out_pt.detach().cpu().numpy()[0, 0]
out_inv = out_inv_pt.detach().cpu().numpy()[0, 0]

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