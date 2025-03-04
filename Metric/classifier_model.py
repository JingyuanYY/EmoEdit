import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class clip_classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # self.fc = nn.Linear(512, num_classes)
        self.hidden = nn.Linear(512, 256)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return x

class resnet50_classifier(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.cnn = models.resnet50(pretrained=True)

        self.backbone = nn.Sequential(*list(self.cnn.children())[:-2])
        self.flaten = nn.Sequential(nn.AvgPool2d(kernel_size=7), nn.Flatten())
        self.fc_1 = nn.Linear(2048, 768)
        self.fc_2 = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(768, 8)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.flaten(x)
        x = self.fc_1(x)
        x = self.fc_2(x)
        return x
