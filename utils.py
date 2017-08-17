"""Utilities for ADDA."""

import random

import torch

from datasets import get_mnist, get_usps


def init_random_seed(manual_seed):
    """Init random seed."""
    seed = None
    if manual_seed is None:
        seed = random.randint(1, 10000)
    else:
        seed = manual_seed
    print("use random seed: {}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_data_loader(name):
    if name == "MNIST":
        return get_mnist()
    elif name == "USPS":
        return get_usps()
