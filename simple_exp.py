import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class Net(nn.Module):
    """Applies convolution with a 5x5 kernel."""
    def __init__(self):
        super(Net, self).__init__()

        self.conv = nn.Conv2d(1, 1,
                              kernel_size=(5, 5),
                              stride=1,
                              padding=2,
                              bias=False)
        self._initialize_weights()

    def forward(self, x):
        return self.conv(x)

    def _initialize_weights(self):
        self.conv.weight[:] = 0.5


def exponential(net, vec, t=1, n=100):
    """Simple implementation of e^net * vec."""
    result = vec
    for i in range(n):
        result = result + (t / n) * net(result)
    return result


# Create phantom
image = np.zeros([50, 50])
image[25, 25] = 1

# Apply e^A to x, then compute inverse (e^-A)
inp = torch.from_numpy(image[None, None].astype('float32'))

net = Net()
result = exponential(net, inp)
inverse = exponential(net, result, t=-1)

# Reconver results and display.
out = result.detach().numpy()[0, 0]
out_inv = inverse.detach().numpy()[0, 0]

plt.figure()
plt.imshow(image)

plt.figure()
plt.imshow(out)

plt.figure()
plt.imshow(out_inv)