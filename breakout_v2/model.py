from torch import nn
from utills import init_weights, normalized_columns_initializer
import torch
import numpy as np
import torch.nn.functional as F


class DQN_LNN(nn.Module):
    def __init__(self, init_dim, layers, class_num, device):
        super(DQN_LNN, self).__init__()
        input_dim = init_dim
        self.device = device

        linear_layers = nn.ModuleList()
        for dim in layers:
            linear_layers.append(nn.Linear(input_dim, dim).to(device))
            linear_layers.append(nn.LeakyReLU().to(device))
            input_dim = dim
        linear_layers.append(nn.Linear(input_dim, class_num).to(device))
        self.linear_layers = linear_layers

    def _init_weights(self):
        self.apply(init_weights)
        self.fc4.weight.data = normalized_columns_initializer(self.fc4.weight.data, 0.0001)
        self.fc4.bias.data.fill_(0)
        self.fc5.weight.data = normalized_columns_initializer(self.fc5.weight.data, 0.0001)
        self.fc5.bias.data.fill_(0)

    def forward(self, x):
        for layer in self.linear_layers:
            x = layer(x)
        return x

class DQN_CNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device):
        super(DQN_CNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device
        self.conv1 = nn.Conv2d(self.input_dim, 16, kernel_size=3, stride=2)
        self.rl1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.rl2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.rl3 = nn.ReLU()

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        self.fc4 = nn.Linear(32 * 5 * 5, self.hidden_dim)
        self.rl4 = nn.ReLU()
        self.fc5 = nn.Linear(self.hidden_dim, self.output_dim)

        self._init_weights()

    def _init_weights(self):
        self.apply(init_weights)
        self.fc4.weight.data = normalized_columns_initializer(self.fc4.weight.data, 0.0001)
        self.fc4.bias.data.fill_(0)
        self.fc5.weight.data = normalized_columns_initializer(self.fc5.weight.data, 0.0001)
        self.fc5.bias.data.fill_(0)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.rl1(self.conv1(x))
        x = self.rl2(self.conv2(x))
        x = self.rl3(self.conv3(x))
        x = self.rl4(self.fc4(x.view(batch_size, -1)))
        return self.fc5(x)

class DQN(nn.Module):

    def __init__(self, hist_size, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(hist_size, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


