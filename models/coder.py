import torch
import math
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils import data
import torch.utils.model_zoo as model_zoo
from torchvision import models

affine_par = True

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1,  dilation_ = 1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
        self.bn1 = nn.BatchNorm2d(planes,affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
            padding = 1
            if dilation_ == 2:
                padding = 2
            elif dilation_ == 4:
                padding = 4
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                               padding=padding, bias=False, dilation = dilation_)
        self.bn2 = nn.BatchNorm2d(planes,affine = affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine = affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
class Encoder(nn.Module):
    def __init__(self):
        self.inplanes = 64
        super(Encoder, self).__init__()
        # self.conv1_p = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=True)
        self.conv1 = self.conv1 = nn.Conv2d(3, 64,
                               kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True) # 1/2, 64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.layer1 = self._make_layer(Bottleneck, 64, 3) # 1/4, 256
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2) # 1/8, 512
        self.res4 = self._make_layer(Bottleneck, 256, 6, stride=1, dilation__ = 2) # 1/16, 1024
        self.res5 = self._make_layer(Bottleneck, 512, 3, stride=1, dilation__ = 4) # 1/32, 2048

        # initial weight
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # for m in self.modules():
        #         #     if isinstance(m, nn.BatchNorm2d):
        #         #         for p in m.parameters():
        #         #             p.requires_grad = False

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

# in_p
    def _make_layer(self, block, planes, blocks, stride=1,dilation__ = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation__ == 2 or dilation__ == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par),
            )
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation_=dilation__, downsample = downsample ))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,dilation_=dilation__))

        return nn.Sequential(*layers)

    def forward(self, in_f):
        extract_feature = []    #
        f = (in_f - Variable(self.mean)) / Variable(self.std)
        x = self.conv1(f) # + self.conv1_n(n)
        x = self.bn1(x)
        c1 = self.relu(x)   # 1/2, 64
        extract_feature.append(c1)
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.layer1(x)   # 1/4, 256
        extract_feature.append(r2)  #
        r3 = self.layer2(r2) # 1/8, 512
        extract_feature.append(r3)  # 
        r4 = self.res4(r3) # 1/16, 1024
        extract_feature.append(r4)  # 
        r5 = self.res5(r4) # 1/32, 2048
        extract_feature.append(r5)  # 
        return extract_feature


class Refine(nn.Module):
    def __init__(self, inplanes, planes, scale_factor=2):
        super(Refine, self).__init__()
        self.convFS1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1)
        self.convFS2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.convFS3 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.convMM1 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.convMM2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.scale_factor = scale_factor

    def forward(self, f, pm):
        s = self.convFS1(f)
        sr = self.convFS2(F.relu(s))
        sr = self.convFS3(F.relu(sr))
        s = s + sr
        # print(s.shape)
        # m = s + F.upsample(pm, size=[s.shape[2],s.shape[3]], mode='bilinear')
        m = s + F.interpolate(pm,size=[s.shape[2],s.shape[3]],mode='bilinear',align_corners=False)

        mr = self.convMM1(F.relu(m))
        mr = self.convMM2(F.relu(mr))
        m = m + mr
        return m


class Decoder(nn.Module):
    def __init__(self, in_channel = 1):
        super(Decoder, self).__init__()
        mdim = 128 # 
        #self.GC = GC(4096, mdim)  # 1/32 -> 1/32
        self.GC = nn.Sequential(               # 
            nn.Conv2d(2048,mdim*2,kernel_size=3,padding=1),
            nn.Conv2d(mdim*2,mdim*2,kernel_size=3,padding=1),
            nn.Conv2d(mdim*2,mdim,kernel_size=3,padding=1)
        )
        self.convG =  nn.Conv2d(in_channel*16+mdim + 1, mdim, kernel_size=3, padding=1)
        self.convG1 = nn.Conv2d(in_channel*16+mdim + 1, mdim, kernel_size=3, padding=1)
        self.convG2 = nn.Conv2d(mdim, mdim, kernel_size=3, padding=1)
        self.RF4 = Refine(1024, mdim)  # 1/16 -> 1/8
        self.RF3 = Refine(512, mdim)  # 1/8 -> 1/4
        self.RF2 = Refine(256, mdim)  # 1/4 -> 1
        
        self.pred5 = nn.Conv2d(mdim, in_channel, kernel_size=(3, 3), padding=(1, 1), stride=1) 
        self.pred4 = nn.Conv2d(mdim, in_channel, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.pred3 = nn.Conv2d(mdim, in_channel, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.pred2 = nn.Conv2d(mdim, in_channel, kernel_size=(3, 3), padding=(1, 1), stride=1)

    def forward(self, r5, x5, r4, r3, r2, pre_mask):
        # x = torch.cat((r5, x5), dim=1)
        x = self.GC(r5)
        pre_mask =  F.adaptive_avg_pool2d(pre_mask, x.size()[-2::])
        x = torch.cat((x, x5, pre_mask),dim=1)
        # x = self.GC(x)           # 
        r = self.convG1(F.relu(x))
        x = self.convG(F.relu(x))
        r = self.convG2(F.relu(r))
        # print(x.shape)
        # print(r.shape)
        m5 = x + r  # out: 1/32, 64
        # print(m5.shape)
        m4 = self.RF4(r4, m5)  # out: 1/16, 64
        m3 = self.RF3(r3, m4)  # out: 1/8, 64
        m2 = self.RF2(r2, m3)  # out: 1/4, 64

        p2 = self.pred2(F.relu(m2))

        # p3 = self.pred3(F.relu(m3))
        # p4 = self.pred4(F.relu(m4))
        # p5 = self.pred5(F.relu(m5))

        return p2

