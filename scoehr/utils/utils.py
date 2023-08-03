"""General helper utility functions.
"""

import os

import torch
from torch import nn


def create(*args):
    # Code from https://github.com/CW-Huang/sdeflow-light/blob/524650bc5ad69522b3e0905672deef0650374512/lib/helpers.py#L119
    path = "/".join(a for a in args)
    if not os.path.isdir(path):
        os.makedirs(path)


def weights_init(m):
    # From https://github.com/astorfi/cor-gan/blob/b6df51a16399335bfe995c15b6951f053453fbb3/Generative/medGAN/MIMIC/pytorch/MLP/medGAN.py#L263
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def convert_to_binary(data):
    data[data >= 0.5] = 1.0
    data[data < 0.5] = 0.0
    return data
