import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, train_data_loader, val_data_loader, batch_size=16, epochs=15, lr=1e-3):
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0.0

        model.train()
        for i, (X, y) in enumerate(train_data_loader):
            optimizer.zero_grad()

            X, y = X.to(device), y.to(device)
            y_pred = model(X)

            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()

            pred = torch.argmax(y_pred, 1)
            running_loss += loss.detach().item()
            correct += (sum(pred==y)).item()
        
        running_vloss = 0.0
        vCorrect = 0.0

        model.eval()
        with torch.no_grad():
            for i, (vX, vy) in enumerate(val_data_loader):
                vX, vy = vX.to(device), vy.to(device)
                vy_pred = model(vX)

                vloss = loss_fn(vy_pred, vy)

                vpred = torch.argmax(vy_pred, 1)
                running_vloss += vloss.detach().item()
                vCorrect += (sum(vpred==vy)).item()
        
        train_loss.append(running_loss / (len(train_data_loader)*batch_size))
        train_acc.append(correct / (len(train_data_loader)*batch_size))
        val_loss.append(running_vloss / (len(val_data_loader)*batch_size))
        val_acc.append(vCorrect / (len(val_data_loader)*batch_size))
        print("Epoch {:d}, Training Loss: {:.6f}, Accuracy: {:.3f}".format(epoch+1, running_loss/(len(train_data_loader)*batch_size), correct/(len(train_data_loader)*batch_size)))
        print("Epoch {:d}, Validation Loss: {:.6f}, Accuracy: {:.3f}".format(epoch+1, running_vloss/(len(val_data_loader)*batch_size), vCorrect/(len(val_data_loader)*batch_size)))

    return train_loss, train_acc, val_loss, val_acc
