from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

train_data = CIFAR10(root=".", download=True, train=True, transform=ToTensor())
valid_data = CIFAR10(root=".", download=True, train=False, transform=ToTensor())

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=128, shuffle=False)

def make_loaders(batch_size=128):
    """data loaders for extra experiments"""
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader                          