import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, in_channels, policy_size):
        super(Net, self).__init__()
        
        self.in_channels = in_channels

        self.PolicyNet(policy_size)
        self.ValueNet()

    def forward(self, x):
        policy = self.PolicyForward(x)
        value = self.ValueForward(x)

        return policy, value

    def PolicyNet(self, policy_size):
        # Convolutional Network: 2 filters, kernel size 1x1, stride 1
        self.policy_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=2,
            kernel_size=1,
            stride=1,
            bias=False)

        # Batch normalizer
        self.policy_batchnorm = nn.BatchNorm2d(2)

        # Fully connected linear layer
        self.policy_linear = nn.Linear(128, policy_size, False)

    def ValueNet(self):
        # Convolutional Network: 1 filter, kernel size 1x1, stride 1
        self.value_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=1,
            kernel_size=1,
            stride=1,
            bias=False)

        # Batch normalizer
        self.value_batchnorm = nn.BatchNorm2d(1)

        # Linear Hidden layer
        self.value_linear1 = nn.Linear(64, 1, False)

        # Linear output value layer
        self.value_linear2 = nn.Linear(1, 1, False)

    def PolicyForward(self, x):
        # Convolutional layer
        x = self.policy_conv(x)
        x = F.leaky_relu(self.policy_batchnorm(x))

        # Linear layer
        x = self.policy_linear(x.flatten())

        return x

    def ValueForward(self, x):
        # Convolutional layer
        x = self.value_conv(x)
        x = F.leaky_relu(self.value_batchnorm(x))

        # Linear layer
        x = F.leaky_relu(self.value_linear1(x.flatten()))

        # Linear layer
        x = F.tanh(self.value_linear2(x.flatten()))

        return x
