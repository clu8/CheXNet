import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class PneumoniaNet(nn.Module):
    def __init__(self, use_gpu, class_counts=(None, None), verbose=False):
        """
        class_counts: (num_pos, num_neg) used for weighted loss
        """
        super(PneumoniaNet, self).__init__()
        
        self.densenet = models.densenet161(pretrained=True)
        
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_features, 1)
        
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.densenet = self.densenet.cuda()
        
        self.optimizer = torch.optim.Adam(self.densenet.parameters())
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=2, verbose=verbose)

        self.num_pos, self.num_neg = class_counts
        self.num_total = self.num_pos + self.num_neg
        
    def forward(self, img):
        return self.densenet(img)

    def loss(self, logit, label):
        weight = torch.zeros(label.size())
        if self.use_gpu:
            weight = weight.cuda()
        is_pos = label.data == 1
        weight[is_pos] = self.num_neg / self.num_total
        weight[~is_pos] = self.num_pos / self.num_total
        weight = Variable(weight, requires_grad=False)
        return F.binary_cross_entropy_with_logits(logit, label, weight)
 
    def train_step(self, img, label):
        logit = self(img)
        loss = self.loss(logit, label)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss

