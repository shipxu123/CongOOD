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


def linear(chIn, chOut, bias=True, norm=True, relu=False): 
    layers = []
    layers.append(nn.Linear(chIn, chOut, bias=bias))
    if norm: 
        layers.append(nn.BatchNorm1d(chOut, affine=bias))
    if relu: 
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)



# Reproduce the network architecture from "Painting on Placement: Forecasting Routing Congestion using Conditional Generative Adversarial Nets"
class Generator(nn.Module): 
    def __init__(self, Cin, Cout): 
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upscale = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.sigmoid = nn.Sigmoid()
        self.conv1 = conv2d(Cin, 64,  kernel_size=3, stride=1, padding='same', norm=True, relu=True)
        self.conv2 = conv2d(64,  128, kernel_size=3, stride=1, padding='same', norm=True, relu=True)
        self.conv3 = conv2d(128, 256, kernel_size=3, stride=1, padding='same', norm=True, relu=True)
        self.conv4 = conv2d(256, 512, kernel_size=3, stride=1, padding='same', norm=True, relu=True)
        self.conv5 = conv2d(512, 512, kernel_size=3, stride=1, padding='same', norm=True, relu=True)
        self.conv6 = conv2d(512, 512, kernel_size=3, stride=1, padding='same', norm=True, relu=True)
        self.conv7 = conv2d(512, 512, kernel_size=3, stride=1, padding='same', norm=True, relu=True)
        self.conv8 = conv2d(512, 512, kernel_size=3, stride=1, padding='same', norm=True, relu=True)
        self.deconv8 = conv2d(512, 512,  kernel_size=3, stride=1, padding='same', norm=True,  relu=True)
        self.deconv7 = conv2d(512, 512,  kernel_size=3, stride=1, padding='same', norm=True,  relu=True)
        self.deconv6 = conv2d(512, 512,  kernel_size=3, stride=1, padding='same', norm=True,  relu=True)
        self.deconv5 = conv2d(512, 512,  kernel_size=3, stride=1, padding='same', norm=True,  relu=True)
        self.deconv4 = conv2d(512, 256,  kernel_size=3, stride=1, padding='same', norm=True,  relu=True)
        self.deconv3 = conv2d(256, 128,  kernel_size=3, stride=1, padding='same', norm=True,  relu=True)
        self.deconv2 = conv2d(128, 64,   kernel_size=3, stride=1, padding='same', norm=True,  relu=True)
        self.deconv1 = conv2d(64,  Cout, kernel_size=3, stride=1, padding='same', norm=False, relu=False)
        

    def forward(self, x): 
        x = self.conv1(x)
        conv1 = self.pool(x)
        x = self.conv2(conv1)
        conv2= self.pool(x)
        x = self.conv3(conv2)
        conv3 = self.pool(x)
        x = self.conv4(conv3)
        conv4 = self.pool(x)
        x = self.conv5(conv4)
        conv5 = self.pool(x)
        x = self.conv6(conv5)
        x = self.pool(x)
        x = self.conv7(x)
        x = self.pool(x)
        x = self.conv8(x)
        x = self.pool(x)
        x = self.upscale(x)
        x = self.deconv8(x)
        x = self.upscale(x)
        x = self.deconv7(x)
        x = self.upscale(x)
        x = self.deconv6(x) + conv5
        x = self.upscale(x)
        x = self.deconv5(x) + conv4
        x = self.upscale(x)
        x = self.deconv4(x) + conv3
        x = self.upscale(x)
        x = self.deconv3(x) + conv2
        x = self.upscale(x)
        x = self.deconv2(x) + conv1
        x = self.upscale(x)
        x = self.deconv1(x)
        x = self.sigmoid(x)

        return x

class Discriminator(nn.Module): 
    def __init__(self, Cin): 
        super().__init__()
        pool = nn.MaxPool2d(kernel_size=2, stride=2)
        sigmoid = nn.Sigmoid()
        conv1 = conv2d(Cin, 64,  kernel_size=3, stride=1, padding=1, norm=True, relu=True)  
        conv2 = conv2d(64,  128, kernel_size=3, stride=1, padding=1, norm=True, relu=True) 
        conv3 = conv2d(128, 256, kernel_size=3, stride=1, padding=1, norm=True, relu=True) 
        conv4 = conv2d(256, 512, kernel_size=3, stride=1, padding=1, norm=True, relu=True) 
        conv5 = conv2d(512, 1,   kernel_size=3, stride=1, padding=1, norm=True, relu=True) 
        flatten = nn.Flatten()
        fc1 = linear(32*32*1, 1, norm=False, relu=False)
        self._seq = nn.Sequential(conv1, pool, conv2, pool, conv3, pool, 
                                  conv4, conv5, flatten, fc1, sigmoid)

    def forward(self, x): 
        return self._seq(x) 



if __name__ == "__main__": 
    netG = Generator(Cin=3, Cout=2)
    netD = Discriminator(Cin=2)
    if torch.cuda.is_available():
        netG = netG.cuda()
        netD = netD.cuda()
        summary(netG, input_size=(3, 256, 256), device="cuda")
        summary(netD, input_size=(2, 256, 256), device="cuda")
