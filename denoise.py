import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import invertible_layers as il

import matplotlib.pyplot as plt

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()

    inverse = getattr(model, 'inverse', None)

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        target = data
        data = data + 0.1 * torch.randn(data.shape).to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

            plt.figure('input')
            plt.imshow(data.detach().cpu().numpy()[0, 0])
            plt.pause(0.001)

            plt.figure('results')
            plt.imshow(output.detach().cpu().numpy()[0, 0])
            plt.pause(0.001)

            if inverse is not None:
                inverted = inverse(output)
                plt.figure('inverse')
                plt.imshow(inverted.detach().cpu().numpy()[0, 0])
                plt.pause(0.001)

                plt.figure('diff')
                plt.imshow((inverted - data).detach().cpu().numpy()[0, 0],
                           clim=[-1, 1], cmap='coolwarm')
                plt.pause(0.001)

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')
args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
                   batch_size=args.batch_size, shuffle=True, **kwargs)


if 1:
    # Invertible
    model = il.Sequential(il.PixelShuffle(4),
                          il.Conv2d(16, 3),
                          il.LeakyReLU(0.5),
                          il.Conv2d(16, 3),
                          il.LeakyReLU(0.5),
                          il.Conv2d(16, 3),
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

for epoch in range(1, args.epochs + 1):
    train(args, model, device, train_loader, optimizer, epoch)

if (args.save_model):
    torch.save(model.state_dict(),"mnist_cnn.pt")