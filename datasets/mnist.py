"""Dataset setting and data loader for MNIST."""


import torch
from torchvision import datasets, transforms

import params

# image pre-processing
pre_process = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize(
                                      mean=params.dataset_mean,
                                      std=params.dataset_std)])

# dataset and data loader
mnist_dataset = datasets.MNIST(root=params.data_root,
                               transform=pre_process,
                               download=True)

mnist_data_loader = torch.utils.data.DataLoader(dataset=mnist_dataset,
                                                batch_size=params.batch_size,
                                                shuffle=True)


def get_mnist():
    """Get MNIST dataset loader."""
    return mnist_data_loader
