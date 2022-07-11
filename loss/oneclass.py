import torch
import torch.nn as nn
from torch.autograd.function import Function
import torch.nn.functional as F
from torch.nn import Parameter

class OCSoftmax(nn.Module):
    def __init__(self, in_dim=2, r_real=0.9, r_fake=0.2, alpha=20.0, **kwargs):

        super(OCSoftmax, self).__init__()
        self.feat_dim = in_dim
        self.r_real = r_real
        self.r_fake = r_fake
        self.alpha = alpha
        self.center = nn.Parameter(torch.randn(1, self.feat_dim))
        nn.init.kaiming_uniform_(self.center, 0.25)
        self.softplus = nn.Softplus()

    def forward(self, x, labels=None, is_train=True):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
            is_train: check if we are in in train mode.
        """
        w = F.normalize(self.center, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)

        scores = x @ w.transpose(0, 1)
        output_scores = scores.clone()

        if is_train:
            scores[labels == 0] = self.r_real - scores[labels == 0]
            scores[labels == 1] = scores[labels == 1] - self.r_fake

        loss = self.softplus(self.alpha * scores).mean()

        return loss, -output_scores.squeeze(1)


if __name__=="__main__":
      batchsize = 64
      input_dim = 256
      class_num = 5

      l_loss = OCSoftmax()
      data = torch.rand(64, 2, requires_grad=True)
      target = (torch.rand(64) * 2).clamp(0, class_num-1)
       
      loss, pred = l_loss(data, None, is_train=False)
    
      print(loss)
      print(pred)
