"""Main script for ADDA."""

import params
from utils import get_data_loader, init_random_seed

if __name__ == '__main__':
    # init random seed
    init_random_seed(params.manual_seed)

    # load dataset
    src_dataset = get_data_loader("MNIST")
    tgt_dataset = get_data_loader("USPS")
