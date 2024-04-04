import torch
import torch.nn as nn

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)
    
class FCN(nn.Module):
    def __init__(self):
        super(FCN,self).__init__()
        self.layer1=torch.nn.Linear(128*128*3,96)
        self.layer2=torch.nn.Linear(96,46)
        self.layer3=torch.nn.Linear(46,6)

    # step 4
    def forward(self,x):
        x=x.view(-1,1*128*128*3)
        x=torch.nn.functional.relu(self.layer1(x))
        x=torch.nn.functional.relu(self.layer2(x))
        x=self.layer3(x)
        return x