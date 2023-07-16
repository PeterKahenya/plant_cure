import torch
from torch import nn

class PlantDiseaseDetector(nn.Module):
    def __init__(self,number_of_classes):
        super().__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)),

            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)),

            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)),

            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)),

            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(in_features=25088,out_features=1200),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=1200,out_features=1200),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=1200,out_features=number_of_classes),
        )
    
    def forward(self,X):
        flattened = self.flatten(self.cnn(X))
        return self.fc(flattened)

