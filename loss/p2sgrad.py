from operator import index
from turtle import shape
import torch
import torch.nn as nn
from torch.nn import Parameter
from speechbrain.nnet.losses import mse_loss

###################
"""
P2SGrad-MSE
__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
"""
###################

class P2SGradLoss(nn.Module):
    """ Output layer that produces cos theta between activation vector x
    and class vector w_j

    in_dim:     dimension of input feature vectors
    output_dim: dimension of output feature vectors 
                (i.e., number of classes)
    
    Usage example:
      batchsize = 64
      input_dim = 10
      class_num = 5

      l_layer = P2SActivationLayer(input_dim, class_num)
      l_loss = P2SGradLoss()

      data = torch.rand(batchsize, input_dim, requires_grad=True)
      target = (torch.rand(batchsize) * class_num).clamp(0, class_num-1)
      target = target.to(torch.long)

      scores = l_layer(data)
      loss = l_loss(scores, target)

      loss.backward()
    """
    def __init__(self, in_dim, out_dim, smooth=0.1):
        super(P2SGradLoss, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.smooth = smooth
        
        self.weight = Parameter(torch.Tensor(in_dim, out_dim))
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)

        self.m_loss = nn.MSELoss()

    def smooth_labels(self, labels):
        factor = self.smooth

        # smooth the labels
        labels *= (1 - factor)
        labels += (factor / labels.shape[1])
        return labels

    def forward(self, input_feat, target=None, is_train=True):
        """
        Compute P2sgrad activation
        
        input:
        ------
          input_feat: tensor (batchsize, input_dim)

        output:
        -------
          tensor (batchsize, output_dim)
          
        """
        # normalize the weight (again)
        # w (feature_dim, output_dim)
        w = self.weight.renorm(2, 1, 1e-5).mul(1e5)
        
        # normalize the input feature vector
        # x_modulus (batchsize)
        # sum input -> x_modules in shape (batchsize)
        x_modulus = input_feat.pow(2).sum(1).pow(0.5)
        # w_modules (output_dim)
        # w_moduls should be 1, since w has been normalized
        w_modulus = w.pow(2).sum(0).pow(0.5)

        # W * x = ||W|| * ||x|| * cos())))))))
        # inner_wx (batchsize, output_dim)
        inner_wx = input_feat.mm(w)
        # cos_theta (batchsize, output_dim)
        cos_theta = inner_wx / x_modulus.view(-1, 1)
        cos_theta = cos_theta.clamp(-1, 1)

        # P2Sgrad MSE
        # target (batchsize, 1)
        loss=0
        if is_train:
          target = target.long() #.view(-1, 1)
          
          # filling in the target
          # index (batchsize, class_num)
          with torch.no_grad():
              index = torch.zeros_like(cos_theta)
              # index[i][target[i][j]] = 1
              index.scatter_(1, target.data.view(-1, 1), 1)
              index = self.smooth_labels(index)
    
        # MSE between \cos\theta and one-hot vectors
          loss = self.m_loss(cos_theta, index)
        # print(cos_theta[:, 0].shape)
        return loss, -cos_theta[:, 0]


if __name__=="__main__":
      batchsize = 64
      input_dim = 256
      class_num = 5

      # l_layer = P2SActivationLayer(input_dim, class_num)
      l_loss = P2SGradLoss(in_dim=256, out_dim=5)

      data = torch.rand(64, 256, requires_grad=True)
      target = (torch.rand(64) * 5).clamp(0, class_num-1)
      target = target.to(torch.long)
      print(l_loss.weight.shape)
      loss, pred = l_loss(data,None)

      print(loss)
      print(pred)
