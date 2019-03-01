"""Denoise image using reversible network."""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import invertible_layers as il
import matplotlib.pyplot as plt


def train(model, device, train_loader, optimizer, epoch):
    model.train()

    inverse = getattr(model, 'inverse', None)

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)

        target = data
        noise = 0.5 * torch.randn(data.shape).to(device)
        data = data + noise

        optimizer.zero_grad()
        output = model(data)
        if inverse is not None:
            inverted = inverse(output)
            reco_loss = F.mse_loss(inverted, data)

        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            plt.figure('input')
            plt.imshow(data.detach().cpu().numpy()[0, 0])
            plt.pause(0.001)

            plt.figure('results')
            plt.imshow(output.detach().cpu().numpy()[0, 0])
            plt.pause(0.001)

            plt.figure('target')
            plt.imshow(target.detach().cpu().numpy()[0, 0])
            plt.pause(0.001)

            if inverse is not None:
                plt.figure('inverse')
                plt.imshow(inverted.detach().cpu().numpy()[0, 0])
                plt.pause(0.001)

                plt.figure('diff')
                plt.imshow((inverted - data).detach().cpu().numpy()[0, 0],
                           clim=[-1, 1], cmap='coolwarm')
                plt.pause(0.001)
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, reco_loss: {}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item(),
                    reco_loss))
            else:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item()))

use_cuda = torch.cuda.is_available()
epochs = 100
batch_size = 64

device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
                   batch_size=batch_size, shuffle=True, **kwargs)


if 1:
    # Invertible
    model = il.Sequential(il.PixelShuffle(4),
                          il.Conv2d(16, 3, orth=True),
                          il.LeakyReLU(0.2),
                          il.Conv2d(16, 3, orth=True),
                          il.LeakyReLU(0.2),
                          il.Conv2d(16, 3, orth=True),
                          il.PixelUnShuffle(4))
else:
    # Not invertible
    model = il.Sequential(il.PixelShuffle(4),
                          nn.Conv2d(16, 16, 3, 1, 1),
                          il.LeakyReLU(0.5),
                          nn.Conv2d(16, 16, 3, 1, 1),
                          il.LeakyReLU(0.5),
                          nn.Conv2d(16, 16, 3, 1, 1),
                          il.PixelUnShuffle(4))

model = model.to(device)
optimizer = optim.Adam(model.parameters())

for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer, epoch)
