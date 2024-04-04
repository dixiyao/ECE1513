import torch
import matplotlib.pyplot as plt
from model import ResNet18_Model
from train import train
from test import test
from data_loader import train_loader, val_loader, test_loader

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ResNet18_Model(3).to(device)
train_loss, train_acc, val_loss, val_acc = train(model, train_loader, val_loader, batch_size=1500, lr=1e-4)

plt.plot(train_loss, label="training loss")
plt.plot(val_loss, label="validation loss")
plt.title("Training & Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(loc="upper right", prop={'size':6})
plt.show()

plt.plot(train_acc, label="training accuracy")
plt.plot(val_acc, label="validation accuracy")
plt.title("Training & Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(loc="lower right", prop={'size':6})
plt.show()

test_loss, test_acc = test(model, test_loader, 1500)
