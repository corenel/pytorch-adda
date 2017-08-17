"""Main script for ADDA."""

import params
from datasets.usps import get_usps
from utils import init_random_seed

if __name__ == '__main__':
    # init random seed
    init_random_seed(params.manual_seed)

    # load dataset
    src_dataset = get_usps()
