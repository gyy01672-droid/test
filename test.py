import torchvision
from sympy.printing.pytorch import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

dataset=torchvision.datasets.CIFAR10(root='./data',transform=transforms.ToTensor(), train=False, download=True)
dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)
class test (nn.Module):
    def __init__(self):
        super(test, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)
    def forward(self, input):
        input = self.conv1(input)
        return input
test1 = test()
for data in dataloader:
    imgs,tragets=data
    output=test1(imgs)
    print(output.shape)
