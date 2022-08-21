import torch
import torch.nn as nn
import torch.nn.functional as F

class ReluBNConv(nn.Module):
    '''
        Convlution with Dilated Conv
        Relu + BN + Conv
    '''
    def __init__(self, in_channels = 3, out_channels = 3, dirate = 1):
        super(ReluBNConv,self).__init__()
        self.conv_s1 = nn.Conv2d(in_channels, out_channels, 3, \
            padding = 1 * dirate, dilation = 1 * dirate)
        self.bn_s1 = nn.BatchNorm2d(out_channels)
        self.relu_s1 = nn.ReLU(inplace = True)
    def forward(self,x):
        x = self.conv_s1(x)
        x = self.bn_s1(x)
        x = self.relu_s1(x)
        return x

    
def UpDownSampling(src,tar):
    '''
        Upsampling and Downsampling. 
            -- decided by 'tar'
    '''
    src = F.upsample(src, size = tar.shape[2:], mode = 'bilinear')
    return src


class RSU7(nn.Module): # RSU-7
    def __init__(self, in_channels = 3, mid_channels = 12, out_channels = 3):
        super(RSU7, self).__init__()
        self.ReluBNConvin = ReluBNConv(in_channels, out_channels, dirate = 1)
        self.ReluBNConv1 = ReluBNConv(out_channels, mid_channels, dirate = 1)
        self.pool1 = nn.MaxPool2d(2, stride = 2, ceil_mode = True)
        self.ReluBNConv2 = ReluBNConv(mid_channels, mid_channels, dirate = 1)
        self.pool2 = nn.MaxPool2d(2, stride = 2, ceil_mode = True)
        self.ReluBNConv3 = ReluBNConv(mid_channels, mid_channels, dirate = 1)
        self.pool3 = nn.MaxPool2d(2, stride = 2, ceil_mode = True)
        self.ReluBNConv4 = ReluBNConv(mid_channels, mid_channels, dirate = 1)
        self.pool4 = nn.MaxPool2d(2, stride = 2, ceil_mode = True)
        self.ReluBNConv5 = ReluBNConv(mid_channels, mid_channels,dirate = 1)
        self.pool5 = nn.MaxPool2d(2, stride = 2, ceil_mode = True)
        
        self.ReluBNConv6 = ReluBNConv(mid_channels, mid_channels, dirate = 1)
        self.ReluBNConv7 = ReluBNConv(mid_channels, mid_channels, dirate = 2)
        self.ReluBNConv6d = ReluBNConv(mid_channels * 2, mid_channels, dirate = 1)
        self.ReluBNConv5d = ReluBNConv(mid_channels * 2, mid_channels, dirate = 1)
        self.ReluBNConv4d = ReluBNConv(mid_channels * 2, mid_channels, dirate = 1)
        self.ReluBNConv3d = ReluBNConv(mid_channels * 2, mid_channels, dirate = 1)
        self.ReluBNConv2d = ReluBNConv(mid_channels * 2, mid_channels, dirate = 1)
        self.ReluBNConv1d = ReluBNConv(mid_channels * 2, out_channels, dirate = 1)

    def forward(self,x):
        xin = self.ReluBNConvin(x)
        x1 = self.ReluBNConv1(xin)
        x = self.pool1(x1)
        x2 = self.ReluBNConv2(x)
        x = self.pool2(x2)
        x3 = self.ReluBNConv3(x)
        x = self.pool3(x3)
        x4 = self.ReluBNConv4(x)
        x = self.pool4(x4)
        x5 = self.ReluBNConv5(x)
        x = self.pool5(x5)
        x6 = self.ReluBNConv6(x)
        x7 = self.ReluBNConv7(x6)
        
        x6d =  self.ReluBNConv6d(torch.cat((x7, x6), 1))
        x6dup = UpDownSampling(x6d, x5)
        x5d =  self.ReluBNConv5d(torch.cat((x6dup, x5), 1))
        x5dup = UpDownSampling(x5d, x4)
        x4d = self.ReluBNConv4d(torch.cat((x5dup, x4), 1))
        x4dup = UpDownSampling(x4d, x3)
        x3d = self.ReluBNConv3d(torch.cat((x4dup, x3), 1))
        x3dup = UpDownSampling(x3d, x2)
        x2d = self.ReluBNConv2d(torch.cat((x3dup, x2), 1))
        x2dup = UpDownSampling(x2d, x1)
        x1d = self.ReluBNConv1d(torch.cat((x2dup, x1), 1))
        return x1d + xin


class RSU6(nn.Module): # RSU-6
    def __init__(self, in_channels = 3, mid_channels = 12, out_channels = 3):
        super(RSU6,self).__init__()
        self.ReluBNConvin = ReluBNConv(in_channels, out_channels, dirate = 1)
        self.ReluBNConv1 = ReluBNConv(out_channels, mid_channels, dirate = 1)
        self.pool1 = nn.MaxPool2d(2, stride = 2, ceil_mode = True)
        self.ReluBNConv2 = ReluBNConv(mid_channels, mid_channels, dirate = 1)
        self.pool2 = nn.MaxPool2d(2, stride = 2, ceil_mode = True)
        self.ReluBNConv3 = ReluBNConv(mid_channels, mid_channels, dirate = 1)
        self.pool3 = nn.MaxPool2d(2, stride = 2, ceil_mode = True)
        self.ReluBNConv4 = ReluBNConv(mid_channels, mid_channels, dirate = 1)
        self.pool4 = nn.MaxPool2d(2, stride = 2, ceil_mode = True)
        
        self.ReluBNConv5 = ReluBNConv(mid_channels, mid_channels, dirate = 1)
        self.ReluBNConv6 = ReluBNConv(mid_channels, mid_channels, dirate = 2)
        self.ReluBNConv5d = ReluBNConv(mid_channels * 2,mid_channels, dirate = 1)
        self.ReluBNConv4d = ReluBNConv(mid_channels * 2,mid_channels, dirate = 1)
        self.ReluBNConv3d = ReluBNConv(mid_channels * 2,mid_channels, dirate = 1)
        self.ReluBNConv2d = ReluBNConv(mid_channels * 2,mid_channels, dirate = 1)
        self.ReluBNConv1d = ReluBNConv(mid_channels * 2,out_channels, dirate = 1)

    def forward(self,x):
        xin = self.ReluBNConvin(x)
        x1 = self.ReluBNConv1(xin)
        x = self.pool1(x1)
        x2 = self.ReluBNConv2(x)
        x = self.pool2(x2)
        x3 = self.ReluBNConv3(x)
        x = self.pool3(x3)
        x4 = self.ReluBNConv4(x)
        x = self.pool4(x4)
        x5 = self.ReluBNConv5(x)
        x6 = self.ReluBNConv6(x5)

        x5d =  self.ReluBNConv5d(torch.cat((x6, x5),1))
        x5dup = UpDownSampling(x5d, x4)
        x4d = self.ReluBNConv4d(torch.cat((x5dup, x4),1))
        x4dup = UpDownSampling(x4d, x3)
        x3d = self.ReluBNConv3d(torch.cat((x4dup, x3),1))
        x3dup = UpDownSampling(x3d, x2)
        x2d = self.ReluBNConv2d(torch.cat((x3dup, x2),1))
        x2dup = UpDownSampling(x2d, x1)
        x1d = self.ReluBNConv1d(torch.cat((x2dup, x1),1))
        return x1d + xin


class RSU5(nn.Module): # RSU-5
    def __init__(self, in_channels = 3, mid_channels = 12, out_channels = 3):
        super(RSU5,self).__init__()
        self.ReluBNConvin = ReluBNConv(in_channels, out_channels, dirate = 1)
        self.ReluBNConv1 = ReluBNConv(out_channels,mid_channels, dirate = 1)
        self.pool1 = nn.MaxPool2d(2,stride=2, ceil_mode = True)
        self.ReluBNConv2 = ReluBNConv(mid_channels, mid_channels, dirate = 1)
        self.pool2 = nn.MaxPool2d(2,stride=2, ceil_mode = True)
        self.ReluBNConv3 = ReluBNConv(mid_channels, mid_channels, dirate = 1)
        self.pool3 = nn.MaxPool2d(2,stride=2, ceil_mode = True)
        self.ReluBNConv4 = ReluBNConv(mid_channels, mid_channels, dirate = 1)
        self.ReluBNConv5 = ReluBNConv(mid_channels, mid_channels, dirate = 2)
        
        self.ReluBNConv4d = ReluBNConv(mid_channels * 2, mid_channels, dirate = 1)
        self.ReluBNConv3d = ReluBNConv(mid_channels * 2, mid_channels, dirate = 1)
        self.ReluBNConv2d = ReluBNConv(mid_channels * 2, mid_channels, dirate = 1)
        self.ReluBNConv1d = ReluBNConv(mid_channels * 2, out_channels, dirate = 1)

    def forward(self,x):
        xin = self.ReluBNConvin(x)
        x1 = self.ReluBNConv1(xin)
        x = self.pool1(x1)
        x2 = self.ReluBNConv2(x)
        x = self.pool2(x2)
        x3 = self.ReluBNConv3(x)
        x = self.pool3(x3)
        x4 = self.ReluBNConv4(x)
        x5 = self.ReluBNConv5(x4)

        x4d = self.ReluBNConv4d(torch.cat((x5, x4),1))
        x4dup = UpDownSampling(x4d, x3)
        x3d = self.ReluBNConv3d(torch.cat((x4dup, x3),1))
        x3dup = UpDownSampling(x3d, x2)
        x2d = self.ReluBNConv2d(torch.cat((x3dup, x2),1))
        x2dup = UpDownSampling(x2d, x1)
        x1d = self.ReluBNConv1d(torch.cat((x2dup, x1),1))
        return x1d + xin


class RSU4(nn.Module): # RSU-4
    def __init__(self, in_channels=3, mid_channels=12, out_channels=3):
        super(RSU4,self).__init__()
        self.ReluBNConvin = ReluBNConv(in_channels, out_channels, dirate = 1)
        self.ReluBNConv1 = ReluBNConv(out_channels, mid_channels, dirate = 1)
        self.pool1 = nn.MaxPool2d(2,stride=2, ceil_mode = True)
        self.ReluBNConv2 = ReluBNConv(mid_channels, mid_channels, dirate = 1)
        self.pool2 = nn.MaxPool2d(2,stride=2, ceil_mode = True)
        self.ReluBNConv3 = ReluBNConv(mid_channels, mid_channels, dirate = 1)
        self.ReluBNConv4 = ReluBNConv(mid_channels, mid_channels, dirate = 2)

        self.ReluBNConv3d = ReluBNConv(mid_channels * 2, mid_channels, dirate = 1)
        self.ReluBNConv2d = ReluBNConv(mid_channels * 2, mid_channels, dirate = 1)
        self.ReluBNConv1d = ReluBNConv(mid_channels * 2, out_channels, dirate = 1)

    def forward(self,x):
        xin = self.ReluBNConvin(x)
        x1 = self.ReluBNConv1(xin)
        x = self.pool1(x1)
        x2 = self.ReluBNConv2(x)
        x = self.pool2(x2)
        x3 = self.ReluBNConv3(x)
        x4 = self.ReluBNConv4(x3)
        
        x3d = self.ReluBNConv3d(torch.cat((x4, x3), 1))
        x3dup = UpDownSampling(x3d,x2)
        x2d = self.ReluBNConv2d(torch.cat((x3dup, x2), 1))
        x2dup = UpDownSampling(x2d,x1)
        x1d = self.ReluBNConv1d(torch.cat((x2dup, x1), 1))
        return x1d + xin


class RSU4F(nn.Module): # RSU-4F
    def __init__(self, in_channels=3, mid_channels=12, out_channels=3):
        super(RSU4F,self).__init__()
        self.ReluBNConvin = ReluBNConv(in_channels, out_channels, dirate = 1)
        self.ReluBNConv1 = ReluBNConv(out_channels, mid_channels, dirate = 1)
        self.ReluBNConv2 = ReluBNConv(mid_channels, mid_channels, dirate = 2)
        self.ReluBNConv3 = ReluBNConv(mid_channels, mid_channels, dirate = 4)
        self.ReluBNConv4 = ReluBNConv(mid_channels, mid_channels, dirate = 8)
        
        self.ReluBNConv3d = ReluBNConv(mid_channels * 2, mid_channels,dirate = 4)
        self.ReluBNConv2d = ReluBNConv(mid_channels * 2, mid_channels, dirate = 2)
        self.ReluBNConv1d = ReluBNConv(mid_channels * 2, out_channels, dirate = 1)

    def forward(self,x):
        xin = self.ReluBNConvin(x)
        x1 = self.ReluBNConv1(xin)
        x2 = self.ReluBNConv2(x1)
        x3 = self.ReluBNConv3(x2)
        x4 = self.ReluBNConv4(x3)
        
        x3d = self.ReluBNConv3d(torch.cat((x4, x3), 1))
        x2d = self.ReluBNConv2d(torch.cat((x3d, x2), 1))
        x1d = self.ReluBNConv1d(torch.cat((x2d, x1), 1))
        return x1d + xin


class UNETPP(nn.Module):
    '''
        UNet++ Framework
    '''
    def __init__(self, in_channels = 3, out_channels = 1):
        super(UNETPP,self).__init__()
        #Encoder
        self.stage1 = RSU7(in_channels, 16, 64)
        self.pool12 = nn.MaxPool2d(2, stride = 2, ceil_mode = True)
        self.stage2 = RSU6(64, 16, 64)
        self.pool23 = nn.MaxPool2d(2, stride = 2, ceil_mode = True)
        self.stage3 = RSU5(64, 16, 64)
        self.pool34 = nn.MaxPool2d(2, stride = 2, ceil_mode = True)
        self.stage4 = RSU4(64, 16, 64)
        self.pool45 = nn.MaxPool2d(2, stride = 2, ceil_mode = True)
        self.stage5 = RSU4F(64, 16, 64)
        self.pool56 = nn.MaxPool2d(2, stride = 2, ceil_mode = True)
        self.stage6 = RSU4F(64, 16, 64)
        
        # Decoder
        self.stage5d = RSU4F(128, 16, 64)
        self.stage4d = RSU4(128, 16, 64)
        self.stage3d = RSU5(128, 16, 64)
        self.stage2d = RSU6(128, 16, 64)
        self.stage1d = RSU7(128, 16, 64)
        self.side1 = nn.Conv2d(64, out_channels, 3, padding = 1)
        self.side2 = nn.Conv2d(64, out_channels, 3, padding = 1)
        self.side3 = nn.Conv2d(64, out_channels, 3, padding = 1)
        self.side4 = nn.Conv2d(64, out_channels, 3, padding = 1)
        self.side5 = nn.Conv2d(64, out_channels, 3, padding = 1)
        self.side6 = nn.Conv2d(64, out_channels, 3, padding = 1)
        self.outconv = nn.Conv2d(out_channels * 6, out_channels, 1)

    def forward(self,x):
        # Encoder
        x1 = self.stage1(x)
        x = self.pool12(x1)
        x2 = self.stage2(x)
        x = self.pool23(x2)
        x3 = self.stage3(x)
        x = self.pool34(x3)
        x4 = self.stage4(x)
        x = self.pool45(x4)
        x5 = self.stage5(x)
        x = self.pool56(x5)
        x6 = self.stage6(x)
        x6up = UpDownSampling(x6, x5)

        # Decoder
        x5d = self.stage5d(torch.cat((x6up, x5), 1))
        x5dup = UpDownSampling(x5d, x4)
        x4d = self.stage4d(torch.cat((x5dup, x4), 1))
        x4dup = UpDownSampling(x4d, x3)
        x3d = self.stage3d(torch.cat((x4dup, x3), 1))
        x3dup = UpDownSampling(x3d, x2)
        x2d = self.stage2d(torch.cat((x3dup, x2), 1))
        x2dup = UpDownSampling(x2d, x1)
        x1d = self.stage1d(torch.cat((x2dup, x1), 1))

        # Side Output
        d1 = self.side1(x1d)
        d2 = self.side2(x2d)
        d2 = UpDownSampling(d2, d1)
        d3 = self.side3(x3d)
        d3 = UpDownSampling(d3, d1)
        d4 = self.side4(x4d)
        d4 = UpDownSampling(d4, d1)
        d5 = self.side5(x5d)
        d5 = UpDownSampling(d5, d1)
        d6 = self.side6(x6)
        d6 = UpDownSampling(d6, d1)
        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))
        return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), \
            F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)

bce_loss = nn.BCELoss(size_average=True)

def multi_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    loss0 = bce_loss(d0.reshape(labels_v.shape), labels_v)
    loss1 = bce_loss(d1.reshape(labels_v.shape), labels_v)
    loss2 = bce_loss(d2.reshape(labels_v.shape), labels_v)
    loss3 = bce_loss(d3.reshape(labels_v.shape), labels_v)
    loss4 = bce_loss(d4.reshape(labels_v.shape), labels_v)
    loss5 = bce_loss(d5.reshape(labels_v.shape), labels_v)
    loss6 = bce_loss(d6.reshape(labels_v.shape), labels_v)
    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    return loss0, loss