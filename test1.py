import torch
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader

dataset=torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
class test1(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1=(19688,10)
    def forward(self, x):
        output=self.linear(x)
        return output
Test=test1()
for data in dataloader:
    imgs,targets=data
    print(imgs.shape)
    output=torch.flatten(imgs)
    print(output.shape)
    output=Test(output)
    print(output.shape)
