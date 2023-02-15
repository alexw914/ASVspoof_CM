#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch, os
import torch.nn as nn
from pytorch_model_summary import summary
import math

class AttentiveStatsPool(torch.nn.Module):
    def __init__(self, in_dim, bottleneck_dim=128, context=True, stddev=True):
        super().__init__()
        # Use Conv1d with stride == 1 rather than Linear, then we don't need to transpose inputs.
        self.context = context
        self.stddev = stddev
        if self.context == True:
            self.linear1 = torch.nn.Conv1d(in_dim*3, bottleneck_dim, kernel_size=1) # equals W and b in the paper
        else:
            self.linear1 = torch.nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1) # equals W and b in the paper
        self.bn1     = torch.nn.BatchNorm1d(bottleneck_dim)
        self.linear2 = torch.nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1) # equals V and k in the paper
    def forward(self, x):
        # DON'T use ReLU here! In experiments, I find ReLU hard to converge.
        if self.context:
            t = x.size()[-1] 
            global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
        else:
            global_x = x
        alpha = torch.relu(self.linear1(global_x))
        alpha = torch.tanh(self.bn1(alpha))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        if self.stddev == False:
            return mean
        residuals = torch.sum(alpha * x ** 2, dim=2) - mean ** 2
        std = torch.sqrt(residuals.clamp(min=1e-9))

        return torch.cat([mean, std], dim=1)

class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=8):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
#         self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
#         out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=8):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
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
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ResNetSE(nn.Module):
    def __init__(self, layers, num_filters, lin_neurons=256, input_size=40, **kwargs):
        super(ResNetSE, self).__init__()

        # print('Embedding size is %d, encoder %s.'%(lin_neurons, "ASP"))        
        self.inplanes   = num_filters[0]
        self.n_mels     = input_size
        # self.ins_norm = nn.BatchNorm1d(self.n_mels)

        self.conv1 = nn.Conv2d(1, num_filters[0] , kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.block = SEBasicBlock
        
        self.layer1 = self._make_layer(self.block, num_filters[0], layers[0])
        self.layer2 = self._make_layer(self.block, num_filters[1], layers[1], stride=(2, 2))
        self.layer3 = self._make_layer(self.block, num_filters[2], layers[2], stride=(2, 2))
        self.layer4 = self._make_layer(self.block, num_filters[3], layers[3], stride=(2, 2))

        outmap_size = math.ceil(self.n_mels/8)
        cat_channel = outmap_size * num_filters[3]

        self.pooling = AttentiveStatsPool(cat_channel, 128, context=False, stddev=False)
        self.pooling_bn = nn.BatchNorm1d(cat_channel)
        self.fc = nn.Linear(cat_channel, lin_neurons)
        self.fc_out = nn.Linear(lin_neurons,2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

    def forward(self, x):

        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.reshape(x.size()[0],-1,x.size()[-1])
        x = self.pooling_bn(self.pooling(x))

        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        x = self.fc_out(x)
        return x


def MainModel(lin_neurons=256, **kwargs):
    # Number of filters
    num_filters = [32, 64, 128, 256]
    model = ResNetSE(SEBasicBlock, [3, 4, 6, 3], num_filters, lin_neurons, **kwargs)
    return model


if __name__ =="__main__":
    
    os.environ["CUDA_VISABLE_DEVICE"] = "1"
    layers=[2, 2, 2, 2]
    num_filters = [16, 32, 64, 128]
    # num_filters = [64, 128, 256, 512]
    model = ResNetSE(layers, num_filters, lin_neurons=128, input_size=80)
    print(summary(model, torch.randn((32,80,200)), show_input=False))
