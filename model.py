import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.preplayer = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False), nn.MaxPool2d(2, 2), nn.BatchNorm2d(128), nn.ReLU())
        self.res1 = ResBlock(in_channels=128)
        self.layer2 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False), nn.MaxPool2d(2, 2), nn.BatchNorm2d(256), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False), nn.MaxPool2d(2, 2), nn.BatchNorm2d(512), nn.ReLU())
        self.res2 = ResBlock(in_channels=512)
        self.pool = nn.MaxPool2d(4)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.preplayer(x)
        x = self.layer1(x)
        x = self.res1(x) + x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.res2(x) + x
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=-1)
        return x

