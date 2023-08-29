import torch
from torch import nn
from dataclasses import dataclass


class BaseBinaryClassifier(nn.Module):

    def __init__(self, config):
        super(BaseBinaryClassifier, self).__init__()
        self.config = config
        input_size = self.config.input_size
        hidden_size = self.config.hidden_size
        self.output_transformation_str = self.config.output_transformation

        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        if self.output_transformation_str == "sigmoid":
            x = torch.sigmoid(self.layer2(x))
        elif self.output_transformation_str == "identity":
            x = self.layer2(x)
        else:
            raise Exception("Transformation not Implemented!")

        return x
