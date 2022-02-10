import torch
import torch.nn as nn
import torch.nn.functional as F

# Two networks are defined as follows

class PRLinear(nn.Module):
    def __init__(self, num_input_channels=100, num_output_channels=100, skip=True): #'nearest' zero zero reflection
        super(PRLinear, self).__init__()
        self.layer1 = nn.Linear(num_output_channels,100,bias=True)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(100,100,bias=True)
        self.layer3 = nn.Linear(100,100,bias=True)
        self.layer4 = nn.Linear(100,num_output_channels,bias=True)
        self.skip = skip
    def forward(self, input):
        x = self.layer1(input)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        x = self.relu(x)
        x = self.layer4(x)
        if self.skip:
            return input + x
        else:
            return x

class PRLinearPlus(nn.Module):
    def __init__(self, num_input_channels=100, num_output_channels=100): #'nearest' zero zero reflection
        super(PRLinearPlus, self).__init__()
        self.layer1 = nn.Linear(num_output_channels,100,bias=True)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(100,100,bias=True)
        self.layer3 = nn.Linear(100,100,bias=True)
        self.layer4 = nn.Linear(100,num_output_channels,bias=True)
        self.layer5 = nn.Linear(num_output_channels,1,bias=True)
        self.alpha = nn.Parameter(torch.ones(1,1))
    def forward(self, input):
        x = self.layer1(input)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        x = self.relu(x)
        x = self.layer4(x)
        input2 = self.alpha*(self.layer5(input))
        return input2 + x, x, self.alpha #x #0.2                  