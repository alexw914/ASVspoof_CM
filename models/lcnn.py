import torch, sys, os
import torch.nn as nn
import torch.nn.functional as F
from pytorch_model_summary import summary

class AttentiveStatsPool(torch.nn.Module):
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
            global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
        else:
            global_x = x
        alpha = torch.relu(self.linear1(global_x))
        alpha = torch.tanh(self.bn1(alpha))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        residuals = torch.sum(alpha * x ** 2, dim=2) - mean ** 2
        std = torch.sqrt(residuals.clamp(min=1e-9))
        return torch.cat([mean, std], dim=1)

class MaxFeatureMap2D(nn.Module):
    """ Max feature map (along 2D)
    MaxFeatureMap2D(max_dim=1)
    l_conv2d = MaxFeatureMap2D(1)
    data_in = torch.rand([1, 4, 5, 5])
    data_out = l_conv2d(data_in)
    Input:
    ------
    data_in: tensor of shape (batch, channel, ...)
    Output:
    -------
    data_out: tensor of shape (batch, channel//2, ...)
    Note
    ----
    By default, Max-feature-map is on channel dimension,
    and maxout is used on (channel ...)
    """

    def __init__(self, max_dim=1):
        super(MaxFeatureMap2D, self).__init__()
        self.max_dim = max_dim

    def forward(self, inputs):
        # suppose inputs (batchsize, channel, length, dim)

        shape = list(inputs.size())

        if self.max_dim >= len(shape):
            print("MaxFeatureMap: maximize on %d dim" % (self.max_dim))
            print("But input has %d dimensions" % (len(shape)))
            sys.exit(1)
        if shape[self.max_dim] // 2 * 2 != shape[self.max_dim]:
            print("MaxFeatureMap: maximize on %d dim" % (self.max_dim))
            print("But this dimension has an odd number of data")
            sys.exit(1)
        shape[self.max_dim] = shape[self.max_dim] // 2
        shape.insert(self.max_dim, 2)

        # view to (batchsize, 2, channel//2, ...)
        # maximize on the 2nd dim
        m, i = inputs.view(*shape).max(self.max_dim)
        return m


class LCNN(nn.Module):
    def __init__(self, num_nodes=80, enc_dim=128, num_frames = 750):
        super(LCNN, self).__init__()
        self.num_nodes = num_nodes
        self.enc_dim = enc_dim
        self.num_frames = num_frames
        self.conv1 = nn.Sequential(nn.Conv2d(1, 64, (5, 5), 1, padding=(2, 2)),
                                   MaxFeatureMap2D(),
                                   nn.MaxPool2d((2, 2), (2, 2)))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, (1, 1), 1, padding=(0, 0)),
                                   MaxFeatureMap2D(),
                                   nn.BatchNorm2d(32, affine=False))
        self.conv3 = nn.Sequential(nn.Conv2d(32, 96, (3, 3), 1, padding=(1, 1)),
                                   MaxFeatureMap2D(),
                                   nn.MaxPool2d((2, 2), (2, 2)),
                                   nn.BatchNorm2d(48, affine=False))
        self.conv4 = nn.Sequential(nn.Conv2d(48, 96, (1, 1), 1, padding=(0, 0)),
                                   MaxFeatureMap2D(),
                                   nn.BatchNorm2d(48, affine=False))
        self.conv5 = nn.Sequential(nn.Conv2d(48, 128, (3, 3), 1, padding=(1, 1)),
                                   MaxFeatureMap2D(),
                                   nn.MaxPool2d((2, 2), (2, 2)))
        self.conv6 = nn.Sequential(nn.Conv2d(64, 128, (1, 1), 1, padding=(0, 0)),
                                   MaxFeatureMap2D(),
                                   nn.BatchNorm2d(64, affine=False))
        self.conv7 = nn.Sequential(nn.Conv2d(64, 64, (3, 3), 1, padding=(1, 1)),
                                   MaxFeatureMap2D(),
                                   nn.BatchNorm2d(32, affine=False))
        self.conv8 = nn.Sequential(nn.Conv2d(32, 64, (1, 1), 1, padding=(0, 0)),
                                   MaxFeatureMap2D(),
                                   nn.BatchNorm2d(32, affine=False))
        self.conv9 = nn.Sequential(nn.Conv2d(32, 64, (3, 3), 1, padding=[1, 1]),
                                   MaxFeatureMap2D(),
                                   nn.MaxPool2d((2, 2), (2, 2)))
        # self.out = nn.Sequential(nn.Dropout(0.7),
        #                          nn.Linear((num_frames // 16) * (num_nodes // 16) * 32, 160),
        #                          MaxFeatureMap2D())
        self.bilstm = nn.Sequential(        
                    nn.LSTM((num_frames // 16) * (num_nodes // 16) * 32, (num_frames // 16) * (num_nodes // 16) * 32),
                    nn.LSTM((num_frames // 16) * (num_nodes // 16) * 32, (num_frames // 16) * (num_nodes // 16) * 32)
        )
        self.out = nn.Linear((num_frames // 16) * (num_nodes // 16) * 32, self.enc_dim)
        self.asp = AttentiveStatsPool(in_dim=num_nodes, context=False)
        self.fc =  nn.Linear(num_nodes*2, self.enc_dim)
        self.fc_out = nn.Linear(self.enc_dim,2)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        hidden_features = x.flatten(1)
        hidden_features_lstm = self.bilstm(hidden_features)

        feat = self.out((hidden_features_lstm + hidden_features).sum(1))
        feat = self.asp(feat.unsqueeze(-1))
        feat = self.fc(feat)
    

        return feat

if __name__ =="__main__":
    
    os.environ["CUDA_VISABLE_DEVICE"] = "1"
    TDNN = LCNN(num_nodes=80, enc_dim=128)
    print(summary(TDNN, torch.randn((64,80,750)), show_input=False))
