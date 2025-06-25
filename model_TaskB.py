
import torch.nn as nn
from torchvision import models

class FaceNet(nn.Module):
    def __init__(self, embedding_dim=128):
        super(FaceNet, self).__init__()
        self.base = models.resnet18(pretrained=True)
        self.base.fc = nn.Identity()
        self.embedding = nn.Linear(512, embedding_dim)

    def forward(self, x):
        x = self.base(x)
        return self.embedding(x)

class Classifier(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        return self.fc(x)
