import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary

def conv2d(chIn, chOut, kernel_size, stride, padding, bias=True, norm=True, relu=False): 
    layers = []
    layers.append(nn.Conv2d(chIn, chOut, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
    if norm: 
        layers.append(nn.BatchNorm2d(chOut, affine=bias))
    if relu: 
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def deconv2d(chIn, chOut, kernel_size, stride, padding, output_padding, bias=True, norm=True, relu=False): 
    layers = []
    layers.append(nn.ConvTranspose2d(chIn, chOut, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias))
    if norm: 
        layers.append(nn.BatchNorm2d(chOut, affine=bias))
    if relu: 
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)

# Reproduce the network architecture from "RouteNet: Routability Prediction for Mixed-Size Designs Using Convolutional Neural Network"
class RouteNet(nn.Module): 
    def __init__(self, Cin, Cout): 
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upscale = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.sigmoid = nn.Sigmoid()
        self.conv1 = conv2d(Cin, 32, kernel_size=9, stride=1, padding='same', norm=True, relu=True)
        self.conv2 = conv2d(32,  64, kernel_size=7, stride=1, padding='same', norm=True, relu=True)
        self.conv3 = conv2d(64,  32, kernel_size=9, stride=1, padding='same', norm=True, relu=True)
        self.conv4 = conv2d(32,  32, kernel_size=7, stride=1, padding='same', norm=True, relu=True)
        self.deconv4 = deconv2d(32,  16,   kernel_size=9, stride=2, padding=4, output_padding=1, norm=True, relu=True)
        self.deconv3 = conv2d(16+32, 16,   kernel_size=5, stride=1, padding='same', norm=True, relu=True)
        self.deconv2 = deconv2d(16,  4,    kernel_size=5, stride=2, padding=2, output_padding=1, norm=True, relu=True)
        self.deconv1 = conv2d(4,     Cout, kernel_size=3, stride=1, padding='same', norm=False, relu=False)

    def forward(self, x): 
        x = self.conv1(x)
        x = self.pool(x)
        skip = x
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = self.deconv4(x)
        x = self.deconv3(torch.cat([x, skip], dim=1))
        x = self.deconv2(x)
        x = self.deconv1(x)
        x = self.sigmoid(x)

        return x


if __name__ == "__main__": 
    routenet = RouteNet(Cin=3, Cout=2)
    if torch.cuda.is_available():
        routenet = routenet.cuda()
        summary(routenet, input_size=(3, 256, 256), device="cuda")
    
    
    