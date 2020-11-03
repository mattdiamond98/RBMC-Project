import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.PolicyNet()
        self.ValueNet()

    def forward(self, x):
        policy = self.PolicyForward(x)
        value = self.ValueForward(x)

        return policy, value

    def PolicyNet(self):
        # Convolutional Network: 2 filters, kernel size 1x1, stride 1
        self.policy_conv = nn.Conv2d(
            in_channels=1, 
            out_channels=2, 
            kernel_size=1, 
            stride=1,
            bias=False)

        # TODO: this num is prob wrong  
        # Batch normalizer
        self.policy_batchnorm = nn.BatchNorm1d(2)

        # Fully connected linear layer
        self.policy_linear = nn.Linear(2, 2, False)

    def ValueNet(self):
        # Convolutional Network: 1 filter, kernel size 1x1, stride 1
        self.value_conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=1,
            stride=1,
            bias=False)

        # Batch normalizer
        self.value_batchnorm = nn.BatchNorm1d(1)

        self.value_linear1 = nn.Linear(1, 1, False)
        self.value_linear2 = nn.Linear(1, 1, False)


    def PolicyForward(self, x):
        x = self.policy_conv(x)
        x = F.relu(self.policy_batchnorm(x))
        x = self.policy_linear(x.flatten())

        return x

    def ValueForward(self, x):
        x = self.value_conv(x)
        x = F.relu(self.value_batchnorm(x))
        x = F.relu(self.value_linear1(x.flatten()))
        x = F.tanh(self.value_linear2(x.flatten()))

        return x
