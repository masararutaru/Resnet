import torch.nn as nn
import torch.nn.functional as F


class MLP_MNIST(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class SimpleCNN_CIFAR10(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)  # 32→16→8 と2回使う
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.fc = nn.Linear(128 * 8 * 8, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # (B,32,32,32)
        x = self.pool(x)            # (B,32,16,16)
        x = F.relu(self.conv2(x))   # (B,64,16,16)
        x = self.pool(x)            # (B,64,8,8)
        x = F.relu(self.conv3(x))   # (B,128,8,8)
        x = x.flatten(1)            # (B, 128*8*8)
        x = self.fc(x)
        return x


class DeepCNN_CIFAR10(nn.Module):
    """ResNetなしの深いCNN - 勾配消失問題を再現するため（ResNet論文当時の技術レベル）"""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # 入力層
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        
        # 深い層（ResNetなし、BatchNormなし）
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv4 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv5 = nn.Conv2d(64, 64, 3, 1, 1)
        
        # ダウンスampling
        self.pool1 = nn.MaxPool2d(2, 2)  # 32→16
        
        self.conv6 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv7 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv8 = nn.Conv2d(128, 128, 3, 1, 1)
        
        self.pool2 = nn.MaxPool2d(2, 2)  # 16→8
        
        self.conv9 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv10 = nn.Conv2d(256, 256, 3, 1, 1)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # (B,64,32,32)
        x = F.relu(self.conv2(x))   # (B,64,32,32)
        x = F.relu(self.conv3(x))   # (B,64,32,32)
        x = F.relu(self.conv4(x))   # (B,64,32,32)
        x = F.relu(self.conv5(x))   # (B,64,32,32)
        x = self.pool1(x)           # (B,64,16,16)
        x = F.relu(self.conv6(x))   # (B,128,16,16)
        x = F.relu(self.conv7(x))   # (B,128,16,16)
        x = F.relu(self.conv8(x))   # (B,128,16,16)
        x = self.pool2(x)           # (B,128,8,8)
        x = F.relu(self.conv9(x))   # (B,256,8,8)
        x = F.relu(self.conv10(x))  # (B,256,8,8)
        x = self.avgpool(x)         # (B,256,1,1)
        x = x.flatten(1)            # (B,256)
        x = self.fc(x)
        return x


class BasicBlock(nn.Module):
    """ResNetの基本ブロック"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # スキップ接続（入力と出力のチャンネル数が異なる場合）
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += self.shortcut(residual)  # スキップ接続
        out = F.relu(out)
        
        return out


class ResNet_CIFAR10(nn.Module):
    """ResNet - スキップ接続で勾配消失問題を解決"""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # 入力層
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # ResNetブロック
        self.layer1 = self._make_layer(64, 64, 2, stride=1)   # 2ブロック
        self.layer2 = self._make_layer(64, 128, 2, stride=2)  # 2ブロック + ダウンスampling
        self.layer3 = self._make_layer(128, 256, 2, stride=2) # 2ブロック + ダウンスampling
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)      # (B,64,32,32)
        x = self.layer1(x)     # (B,64,32,32)
        x = self.layer2(x)     # (B,128,16,16)
        x = self.layer3(x)     # (B,256,8,8)
        x = self.avgpool(x)    # (B,256,1,1)
        x = x.flatten(1)       # (B,256)
        x = self.fc(x)
        return x 