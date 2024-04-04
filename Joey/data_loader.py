import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

CTImageDataset = ImageFolder("dataset", ToTensor())

train_set, val_set, test_set = torch.utils.data.random_split(CTImageDataset, [40000, 10000, 8954])
train_loader = DataLoader(dataset=train_set, shuffle=False, batch_size=1500)
val_loader = DataLoader(dataset=val_set, shuffle=False, batch_size=1500)
test_loader = DataLoader(dataset=test_set, batch_size=1500)