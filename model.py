import torch
import torch.nn as nn
import torchvision.models as models


class PneumoniaNet(nn.Module):
    def __init__(self, use_gpu):
        super(PneumoniaNet, self).__init__()
        
        self.densenet = models.densenet161(pretrained=True)
        for param in self.densenet.parameters():
            param.requires_grad = False
        
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_features, 1)
        
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.densenet = self.densenet.cuda()
        
        self.optimizer = torch.optim.Adam(self.densenet.classifier.parameters())
        self.loss_fn = nn.BCEWithLogitsLoss()
        
    def forward(self, img):
        return self.densenet(img)
    
    def train_step(self, img, label):
        logit = self(img)
        loss = self.loss_fn(logit, label)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss
