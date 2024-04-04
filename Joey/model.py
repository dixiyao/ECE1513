import torch
import torch.nn as nn

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ResidualBuildingBlock(nn.Module):
    def __init__(self, in_channels, out_channels_1, out_channels_2, stride_1=1, stride_2=1):
        """
        Input:
            in_channels: input channel to conv1
            out_channels_1: output channel to conv1
            out_channels_2: output channel to conv2
            stride_1: stride of conv1
            stride_2: stride of conv2 (will be 2 if requires downsampling)
        """
        super(ResidualBuildingBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels_1 = out_channels_1
        self.out_channels_2 = out_channels_2
        self.stride_1 = stride_1
        self.stride_2 = stride_2

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels_1,
            kernel_size=3,
            stride=stride_1,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels_1)
        self.conv2 = nn.Conv2d(
            out_channels_1,
            out_channels_2,
            kernel_size=3,
            stride=stride_2,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels_2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.stride_1 != self.stride_2:
            conv2d = nn.Conv2d(
                self.in_channels,
                self.out_channels_2,
                kernel_size=3,
                stride=self.stride_1,
                padding=1,
                bias=False,
                device=device
            )
            bn = nn.BatchNorm2d(self.out_channels_2, device=device)
            identity = bn(conv2d(identity))

        x += identity
        out = self.relu(x)

        return out
    
class ResNet18_Model(nn.Module):
    def __init__(self, img_channels):
        super(ResNet18_Model, self).__init__()

        self.conv1 = nn.Conv2d(img_channels, 8, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.ResnetBlock1 = ResidualBuildingBlock(8, 8, 8, 1, 1)
        self.ResnetBlock2 = ResidualBuildingBlock(8, 8, 8, 1, 1)
        self.ResnetBlock3 = ResidualBuildingBlock(8, 8, 8, 1, 1)

        self.ResnetBlock4 = ResidualBuildingBlock(8, 16, 16, 2, 1)
        self.ResnetBlock5 = ResidualBuildingBlock(16, 16, 16, 1, 1)
        self.ResnetBlock6 = ResidualBuildingBlock(16, 16, 16, 1, 1)

        self.ResnetBlock7 = ResidualBuildingBlock(16, 32, 32, 2, 1)
        self.ResnetBlock8 = ResidualBuildingBlock(32, 32, 32, 1, 1)
        self.ResnetBlock9 = ResidualBuildingBlock(32, 32, 32, 1, 1)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=32, out_features=6, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)
        
        x = self.ResnetBlock1(x)
        x = self.ResnetBlock2(x)
        x = self.ResnetBlock3(x)

        x = self.ResnetBlock4(x)
        x = self.ResnetBlock5(x)
        x = self.ResnetBlock6(x)
        
        x = self.ResnetBlock7(x)
        x = self.ResnetBlock8(x)
        x = self.ResnetBlock9(x)

        x = self.pool(x)
        x = x.view(x.shape[0], -1)

        out = self.fc(x)

        return out
