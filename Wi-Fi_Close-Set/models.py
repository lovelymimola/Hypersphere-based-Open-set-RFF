import torch
import torch.nn as nn
import torch.nn.functional as F
from complexcnn import *
from loss_functions import AngularPenaltySMLoss

class ConvAngularPen(nn.Module):
    def __init__(self, num_classes=10, loss_type='arcface'):
        super(ConvAngularPen, self).__init__()
        self.convlayers = ConvNet()
        self.adms_loss = AngularPenaltySMLoss(512, num_classes, loss_type=loss_type)

    def forward(self, x, labels=None, embed=False):
        x = self.convlayers(x)
        if embed:
            return x
        L, logit = self.adms_loss(x, labels)
        return L, logit

class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            ComplexConv(1, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=2))
        self.layer2 = nn.Sequential(
            ComplexConv(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=2))
        self.layer3 = nn.Sequential(
            ComplexConv(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=2))
        self.layer4 = nn.Sequential(
            ComplexConv(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=2))
        self.layer5 = nn.Sequential(
            ComplexConv(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=2))
        self.layer6 = nn.Sequential(
            ComplexConv(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=2))
        self.layer7 = nn.Sequential(
            ComplexConv(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=2))
        self.layer8 = nn.Sequential(
            ComplexConv(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=2))
        self.layer9 = nn.Sequential(
            ComplexConv(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=2))

        self.fc_projection = nn.Linear(1152, 512)

    def forward(self, x, embed=False):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc_projection(x)
        return x

if __name__ == "__main__":
    model = ConvBaseline()
    input = torch.randn((10,2,6000))
    output = model(input)
