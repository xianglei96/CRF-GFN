import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
      super(UpsampleConvLayer, self).__init__()
      reflection_padding = kernel_size // 2
      self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
      self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.relu = nn.PReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = torch.add(out, residual)
        return out
        
class CMResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(CMResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.conv3 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.conv4 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.bottleneck = ConvLayer(3*channels, channels, kernel_size=1, stride=1)
        self.relu = nn.PReLU()

    def forward(self, x):
        residual = x
        out1 = self.relu(self.conv1(x))
        out1 = self.conv2(out1)
        out1 = torch.add(out1, residual)

        residual = out1
        out2 = self.relu(self.conv3(out1))
        out2 = self.conv4(out2)
        out2 = torch.add(out2, residual)

        out = torch.cat((x, out1, out2), 1)
        out = self.bottleneck(out) + x

        return out

class Net(nn.Module):
    def __init__(self, res_blocks=12):
        super(Net, self).__init__()

        self.conv_input = ConvLayer(3, 16, kernel_size=11, stride=1)
        self.dense0 = CMResidualBlock(16)
        self.conv2x = ConvLayer(16, 32, kernel_size=3, stride=2)
        self.dense1 = CMResidualBlock(32)
        self.conv4x = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.dense2 = CMResidualBlock(64)
        self.conv8x = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.dense3 = CMResidualBlock(128)
        self.conv16x = ConvLayer(128, 256, kernel_size=3, stride=2)
        self.dense4 = CMResidualBlock(256)


        self.dehaze1 = nn.Sequential()
        for i in range(0, 4):
            self.dehaze1.add_module('res%d' % i,CMResidualBlock(256))
            
        self.dehaze2 = nn.Sequential()
        for i in range(4, 8):
            self.dehaze2.add_module('res%d' % i, CMResidualBlock(256))
            
        self.dehaze3 = nn.Sequential()
        for i in range(8, 12):
            self.dehaze3.add_module('res%d' % i, CMResidualBlock(256))
            
        self.gate = nn.Conv2d(256 * 3, 3, 3, 1, 1, bias=True)
            

        self.convd16x = UpsampleConvLayer(256, 128, kernel_size=3, stride=2)
        self.dense_4 = CMResidualBlock(128)
        self.convd8x = UpsampleConvLayer(128, 64, kernel_size=3, stride=2)
        self.dense_3 = CMResidualBlock(64)
        self.convd4x = UpsampleConvLayer(64, 32, kernel_size=3, stride=2)
        self.dense_2 = CMResidualBlock(32)
        self.convd2x = UpsampleConvLayer(32, 16, kernel_size=3, stride=2)
        self.dense_1 = CMResidualBlock(16)

        self.conv_output = ConvLayer(16, 3, kernel_size=3, stride=1)


    def forward(self, x):
        x = self.dense0(self.conv_input(x))
        res2x =self.dense1(self.conv2x(x))
        res4x = self.dense2(self.conv4x(res2x))

        res8x = self.dense3(self.conv8x(res4x))
        res16x = self.dense4(self.conv16x(res8x))

        res_dehaze = res16x
        #res16x = self.dehaze(res16x)
        y1 = self.dehaze1(res16x)
        y2 = self.dehaze2(y1)
        y3 = self.dehaze3(y2)                
        gates = self.gate(torch.cat((y1, y2, y3), dim=1))
        gated_y = y1 * gates[:, [0], :, :] + y2 * gates[:, [1], :, :] + y3 * gates[:, [2], :, :]
        res16x = gated_y
        res16x = torch.add(res_dehaze, res16x)

        res16x = self.dense_4(self.convd16x(res16x))
        res16x = F.upsample(res16x, res8x.size()[2:], mode='bilinear')
        res8x = torch.add(res16x, res8x)

        res8x = self.dense_3(self.convd8x(res8x))
        res8x = F.upsample(res8x, res4x.size()[2:], mode='bilinear')
        res4x = torch.add(res8x, res4x)

        res4x = self.dense_2(self.convd4x(res4x))
        res4x = F.upsample(res4x, res2x.size()[2:], mode='bilinear')
        res2x = torch.add(res4x, res2x)

        res2x = self.dense_1(self.convd2x(res2x))
        res2x = F.upsample(res2x, x.size()[2:], mode='bilinear')
        x = torch.add(res2x, x)

        x = self.conv_output(x)

        return x