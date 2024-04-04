import os
import torch
from torchvision import datasets, transforms
from linear_models import FCN
import numpy as np


data_dir = './dataset'
TRAIN = 'train'
VAL = 'valid'
TEST = 'test'

data_transforms = transforms.Compose([             
        transforms.Resize((128,128)), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], # Gaussian Noise
                             [0.229, 0.224, 0.225])])
image_datasets = datasets.ImageFolder(
        data_dir ,
        transform=data_transforms)
train_dataset, val_dataset,test_dataset=torch.utils.data.random_split(image_datasets,[40000,10000,8954])
train_dataloader=torch.utils.data.DataLoader(train_dataset,batch_size=4,shuffle=True)
val_dataloader=torch.utils.data.DataLoader(val_dataset,batch_size=4,shuffle=False)
test_dataloader=torch.utils.data.DataLoader(test_dataset,batch_size=4,shuffle=False)

model=FCN()

device=torch.device("cuda")
model=model.to(torch.device(device))
    
loss_fn=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)

epochs=[]
training_losses=[]
validation_losses=[]
training_accuracy=[]
validation_accuracy=[]
test_losses=[]
test_accuracy=[]
for epoch in range(0,10):
    epochs.append(epoch)
    epoch+=1
    model=model.train()
    total=0
    losses=0
    correct=0   
    for i,data in enumerate(train_dataloader,0):
        inputs,labels=data
        inputs=inputs.to(torch.device(device))
        labels=labels.to(torch.device(device))
        optimizer.zero_grad()
        outputs=model(inputs)
        loss=loss_fn(outputs,labels)
        loss.backward()
        optimizer.step()
        total+=labels.size(0)
        losses+=loss.item()*inputs.size(0)
        correct+=torch.sum(torch.argmax(outputs,dim=1)==labels).item()
    training_loss=losses/total
    training_losses.append(training_loss)
    training_accuracy.append(correct/total)

    model=model.eval()
    total=0
    losses=0
    correct=0
    for data in val_dataloader:
        inputs,labels=data
        inputs=inputs.to(torch.device(device))
        labels=labels.to(torch.device(device))
        outputs=model(inputs)
        val_loss=loss_fn(outputs,labels)
        total+=labels.size(0)
        losses+=val_loss.item()*inputs.size(0)
        correct+=torch.sum(torch.argmax(outputs,dim=1)==labels).item()
    validation_loss=losses/total
    validation_losses.append(validation_loss)
    validation_accuracy.append(correct/total)

    total=0
    losses=0
    correct=0
    for data in test_dataloader:
        inputs,labels=data
        inputs=inputs.to(torch.device(device))
        labels=labels.to(torch.device(device))
        outputs=model(inputs)
        test_loss=loss_fn(outputs,labels)
        total+=labels.size(0)
        losses+=test_loss.item()*inputs.size(0)
        correct+=torch.sum(torch.argmax(outputs,dim=1)==labels).item()
    test_loss=losses/total
    test_losses.append(test_loss)
    test_accuracy.append(correct/total)

    print("Epoch: ",epoch,"Training Loss: ",training_loss,"Validation Loss: ",validation_loss,"Test Loss: ",test_loss)
    print("Training Accuracy: ",training_accuracy[-1],"Validation Accuracy: ",validation_accuracy[-1],"Test Accuracy: ",correct/total)
np.save('training_loss.npy',np.array(training_losses))
np.save('validation_loss.npy',np.array(validation_losses))
np.save('training_accuracy.npy',np.array(training_accuracy))
np.save('validation_accuracy.npy',np.array(validation_accuracy))
np.save('test_loss.npy',np.array(test_losses))
np.save('test_accuracy.npy',np.array(test_accuracy))

    