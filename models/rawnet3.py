# -*- encoding: utf-8 -*-

import torch,math,os,yaml
import torch.nn as nn
import torch.nn.functional as F
from asteroid_filterbanks import Encoder, ParamSincFB
from pytorch_model_summary import summary


class PreEmphasis(torch.nn.Module):
    def __init__(self, coef: float = 0.97) -> None:
        super().__init__()
        self.coef = coef
        # make kernel
        # In pytorch, the convolution operation uses cross-correlation. So, filter is flipped.
        self.register_buffer(
            "flipped_filter",
            torch.FloatTensor([-self.coef, 1.0]).unsqueeze(0).unsqueeze(0),
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        # print(input.size())
        # assert (
        #     len(input.size()) == 2
        # ), "The number of dimensions of input tensor must be 2!"
        # # reflect padding to match lengths of in/out
        # input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), "reflect")
        return F.conv1d(input, self.flipped_filter)


class AFMS(nn.Module):
    """
    Alpha-Feature map scaling, added to the output of each residual block[1,2].

    Reference:
    [1] RawNet2 : https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1011.pdf
    [2] AMFS    : https://www.koreascience.or.kr/article/JAKO202029757857763.page
    """

    def __init__(self, nb_dim: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones((nb_dim, 1)))
        self.fc = nn.Linear(nb_dim, nb_dim)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        y = F.adaptive_avg_pool1d(x, 1).view(x.size(0), -1)
        y = self.sig(self.fc(y)).view(x.size(0), x.size(1), -1)

        x = x + self.alpha
        x = x * y
        return x


class Bottle2neck(nn.Module):
    def __init__(
        self,
        inplanes,
        planes,
        kernel_size=None,
        dilation=None,
        scale=4,
        pool=False,
    ):

        super().__init__()

        width = int(math.floor(planes / scale))

        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)

        self.nums = scale - 1

        convs = []
        bns = []

        num_pad = math.floor(kernel_size / 2) * dilation

        for i in range(self.nums):
            convs.append(
                nn.Conv1d(
                    width,
                    width,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=num_pad,
                )
            )
            bns.append(nn.BatchNorm1d(width))

        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)

        self.relu = nn.ReLU()

        self.width = width

        self.mp = nn.MaxPool1d(pool) if pool else False
        self.afms = AFMS(planes)

        if inplanes != planes:  # if change in number of filters
            self.residual = nn.Sequential(
                nn.Conv1d(inplanes, planes, kernel_size=1, stride=1, bias=False)
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        residual = self.residual(x)

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)

        out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)

        out += residual
        if self.mp:
            out = self.mp(out)
        out = self.afms(out)

        return out


class AttentiveStatsPool(nn.Module):

    def __init__(self, in_dim, bottleneck_dim=128, context=True):
        super().__init__()
        # Use Conv1d with stride == 1 rather than Linear, then we don't need to transpose inputs.
        self.context = context
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
            global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4,max=1e4)).repeat(1,1,t)), dim=1)
        else:
            global_x = x
        alpha = torch.relu(self.linear1(global_x))
        alpha = torch.tanh(self.bn1(alpha))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        residuals = torch.sum(alpha * x ** 2, dim=2) - mean ** 2
        std = torch.sqrt(residuals.clamp(min=1e-9))
        return torch.cat([mean, std], dim=1)


class RawNet3(nn.Module):
    def __init__(self, C, lin_neurons, summed, context, log_sinc, norm_sinc):
        super().__init__()

        self.log_sinc = log_sinc
        self.norm_sinc = norm_sinc
        self.summed = summed
        self.context = context

        self.norm = nn.InstanceNorm1d(1, eps=1e-4, affine=True)

        self.conv1 = Encoder(
            ParamSincFB(
                C // 4,
                251,
                stride=10
            )
        )
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(C // 4)

        self.layer1 = Bottle2neck(
            C // 4, C, kernel_size=3, dilation=2, scale=8, pool=5
        )
        self.layer2 = Bottle2neck(
            C, C, kernel_size=3, dilation=3, scale=8, pool=3
        )
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)

        self.layer4 = nn.Conv1d(3*C, 3*C, kernel_size=1)
        attn_input = 3*C
        self.pooling = AttentiveStatsPool(attn_input, 128, context=context)        

        self.bn5 = nn.BatchNorm1d(attn_input*2)

        self.fc6 = nn.Linear(attn_input*2, lin_neurons)
        self.bn6 = nn.BatchNorm1d(lin_neurons)

        self.mp3 = nn.MaxPool1d(3)

    def forward(self, x):
        """
        :param x: input mini-batch (bs, samp)
        """

        with torch.cuda.amp.autocast(enabled=False):
            x[:, 1:] = x[:, 1:] - 0.97 * x[:, 0:-1]
            x = x.unsqueeze(1)
            x = self.norm(x)
            x = torch.abs(self.conv1(x))
            if self.log_sinc:
                x = torch.log(x + 1e-6)
            if self.norm_sinc == "mean":
                x = x - torch.mean(x, dim=-1, keepdim=True)
            elif self.norm_sinc == "mean_std":
                m = torch.mean(x, dim=-1, keepdim=True)
                s = torch.std(x, dim=-1, keepdim=True)
                s[s < 0.001] = 0.001
                x = (x - m) / s

        if self.summed:
            x1 = self.layer1(x)
            x2 = self.layer2(x1)
            x3 = self.layer3(self.mp3(x1) + x2)
        else:
            x1 = self.layer1(x)
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)

        x = self.layer4(torch.cat((self.mp3(x1), x2, x3), dim=1))
        x = self.relu(x)

        x = self.pooling(x)
        x = self.bn5(x)

        x = self.fc6(x)

        # if self.out_bn:
        #     x = self.bn6(x)

        return x


def MainModel():

    model = RawNet3(C=256, lin_neurons=256, context=True, summed=True, log_sinc=True, norm_sinc="mean")
    return model

if __name__ =="__main__":

    os.environ["CUDA_VISABLE_DEVICE"] = "1"
    model = MainModel()
    print(summary(model, torch.randn((32,64000)), show_input=False))
