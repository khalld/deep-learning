import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import itertools
from torch.utils.data import DataLoader
import torchvision
from torch.optim import SGD

class EmbeddingNet(nn.Module):
    def __init__(self) -> None:
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(1,32,5), 
                                    nn.ReLU(),
                                    nn.MaxPool2d(2,stride=2),
                                    nn.BatchNorm2d(32),
                                    nn.Conv2d(32,64,5),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2,stride=2))
        
        self.fc = nn.Sequential(nn.BatchNorm1d(64*4*4),
                                nn.Linear(64*4*4, 256),
                                nn.ReLU(),
                                nn.BatchNorm1d(256),
                                nn.Linear(256,128))
    
    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, m=2):
        super(ContrastiveLoss, self).__init__()
        self.m = m
    
    def forward(self, phi_i, phi_j, l_ij)
        d = F.pairwise_distance(phi_i, phi_j)
        l = 0.5 * (1 - l_ij.float()) * torch.pow(d,2) + / 0.5* l_ij.float() * torch.pow( torch.clamp(self.m - d, min = 0), 2)
        return l.mean()

# TODO:
# class SiameseNetworkTask(pl.LightningModule):
#     def __init__(self,
#                 embedding_net,
#                 lr=0.01,
#                 momentum=0.99,
#                 margin=2
#                 ):
#         super(SiameseNetworkTask, self).__init__()

if __name__ == "__main__":
    # TODO:
    model = EmbeddingNet()
    _ = model(torch.zeros(16,1,28,28)).shape
    print("Output model: {}".format( _ ))