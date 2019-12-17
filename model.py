import torch
import torch.nn as nn

class DF(nn.Module):
    def __init__(self, nb_classes):
        super(DF, self).__init__()
        self.block1 = nn.Sequential(         
            nn.Conv1d(
                in_channels=1,              
                out_channels=32,            
                kernel_size=8,              
                stride=1,                   
                padding=0,                 
            ),  
            nn.BatchNorm1d(32),                   
            nn.ELU(alpha=1.0),                     
            nn.Conv1d(32, 32, 8, 1, 0),
            nn.BatchNorm1d(32),
            nn.ELU(alpha=1.0),
            nn.MaxPool1d(8, 4, 0), 
            nn.Dropout(0.1), 
        )

        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, 8, 1, 0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 8, 1, 0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(8, 4, 0),
            nn.Dropout(0.1),
        )

        self.block3 = nn.Sequential(
            nn.Conv1d(64, 128, 8, 1, 0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, 8, 1, 0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(8, 4, 0),
            nn.Dropout(0.1),
        )

        self.block4 = nn.Sequential(
            nn.Conv1d(128, 256, 8, 1, 0),
            nn.BatchNorm1d(256), 
            nn.ReLU(),
            nn.Conv1d(256, 256, 8, 1, 0),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(8, 4, 0),
            nn.Dropout(0.1),
        )

        self.fc1 = nn.Sequential(         
            nn.Flatten(),
            nn.Linear(3328,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.7),              
        )

        self.fc2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),              
        )

        self.out = nn.Sequential(
            nn.Linear(512, nb_classes),
        )   

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.fc1(x)
        x = self.fc2(x)          
        output = self.out(x)
        return output, x   


class AWF(nn.Module):
    def __init__(self, nb_classes):
        super(AWF, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Dropout(0.25),         
            nn.Conv1d(1, 32, 5, 1, 0),  
            nn.ReLU(),                     
            nn.MaxPool1d(4), 
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 32, 5, 1, 0),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )

        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(9952, 500),
            nn.ReLU(),
            nn.Linear(500, nb_classes)
        )   

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)     
        output = self.out(x)
        return output, x   


class WF(nn.Module):
    def __init__(self, nb_classes):
        super(WF, self).__init__()
        self.conv1 = nn.Sequential(       
            nn.Conv1d(1, 32, 8, 1, 0),
            nn.BatchNorm1d(32), 
            nn.ELU(),                     
            nn.MaxPool1d(8, 4, 0),
            nn.Dropout(0.1), 
        )

        self.conv2 = nn.Sequential(       
            nn.Conv1d(32, 64, 8, 1, 0),
            nn.BatchNorm1d(64), 
            nn.ELU(),                     
            nn.MaxPool1d(8, 4, 0),
            nn.Dropout(0.1), 
        )

        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(19776, 500),
            nn.ReLU(),
            nn.Linear(500, nb_classes)
        )   

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)     
        output = self.out(x)
        return output, x  