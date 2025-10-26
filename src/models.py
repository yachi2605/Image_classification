

import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
    def __init__(self, layer1=512, layer2=256, out_features=10):
        super().__init__()                    
        in_features = 3 * 32 * 32            
        self.fc1 = nn.Linear(in_features, layer1)
        self.fc2 = nn.Linear(layer1, layer2)
        self.out = nn.Linear(layer2, out_features)

    def forward(self, x):
        x = x.view(x.size(0), -1)            
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)                    

class CnnModel(nn.Module):
    def __init__(self,out_class=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),    
            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),

        )
        self.classifier=nn.Linear(256,out_class)

    def forward(self,x):
        x=self.features(x)
        x=x.view(x.size(0),-1)
        x=self.classifier(x)
        return x