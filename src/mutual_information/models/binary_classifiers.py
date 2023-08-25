import torch
from torch import nn
from dataclasses import dataclass


class BaseBinaryClassifier(nn.Module):

    def __init__(self, config):
        super(BaseBinaryClassifier, self).__init__()
        self.config = config
        input_size = self.config.input_size
        hidden_size = self.config.hidden_size
        output_transformation_str = self.config.output_transformation

        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, 1)
        self.output_transformation = torch.sigmoid if output_transformation_str == "sigmoid" else lambda x : x

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.output_transformation(self.layer2(x))
        return x
