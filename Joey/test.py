import torch
import torch.nn as nn

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test(model, data_loader, batch_size=16):
    correct = 0.0
    running_loss = 0.0
    loss_fn = nn.CrossEntropyLoss()

    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)

            pred = torch.argmax(y_pred, 1)
            running_loss += loss.detach().item()
            correct += (sum(pred==y)).item()

        print("Test Loss: {:.6f}, Accuracy: {:.3f}".format(running_loss/(len(data_loader)*batch_size), correct/(len(data_loader)*batch_size)))
    
    return running_loss/(len(data_loader)*batch_size), correct/(len(data_loader)*batch_size)
        