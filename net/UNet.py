import torch
from torch import nn
import torch.nn.functional as F

class UNet_Encoder(nn.Module):
    def __init__(self, Cin, Cout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(Cin, Cout, 3, 1, 1),
            nn.BatchNorm2d(Cout),
            nn.ReLU(),
            nn.Conv2d(Cout, Cout, 3, 1, 1),
            nn.BatchNorm2d(Cout),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class UNet_Decoder(nn.Module):
    def __init__(self, Cin):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(Cin, Cin//2, 3, 1, 1),
            nn.BatchNorm2d(Cin//2),
            nn.ReLU(),
            nn.Conv2d(Cin//2, Cin//2, 3, 1, 1),
            nn.BatchNorm2d(Cin//2),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, Cin, Cout):
        super().__init__()
        self.encoder1 = UNet_Encoder(Cin, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.encoder2 = UNet_Encoder(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.encoder3 = UNet_Encoder(128, 256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.encoder4 = UNet_Encoder(256, 512)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.mid = nn.Sequential(
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )
        self.upconv1 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.decoder1 = UNet_Decoder(1024)
        self.upconv2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.decoder2 = UNet_Decoder(512)
        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.decoder3 = UNet_Decoder(256)
        self.upconv4 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.decoder4 = UNet_Decoder(128)
        self.readout = nn.Conv2d(64, Cout, 1, 1, 0)




    def forward(self, x):
        x1 = self.encoder1(x)
        x = self.pool1(x1)
        x2 = self.encoder2(x)
        x = self.pool2(x2)
        x3 = self.encoder3(x)
        x = self.pool3(x3)
        x4 = self.encoder4(x)
        x = self.pool4(x4)
        x = self.mid(x)
        x = self.upconv1(x)
        x = torch.cat((x4, x), dim=1)
        x = self.decoder1(x)
        x = self.upconv2(x)
        x = torch.cat((x3, x), dim=1)
        x = self.decoder2(x)
        x = self.upconv3(x)
        x = torch.cat((x2, x), dim=1)
        x = self.decoder3(x)
        x = self.upconv4(x)
        x = torch.cat((x1, x), dim=1)
        x = self.decoder4(x)
        x = F.sigmoid(self.readout(x))
        return x
