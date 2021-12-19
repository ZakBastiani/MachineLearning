import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, config, act=nn.Tanh()):
        super(Network, self).__init__()
