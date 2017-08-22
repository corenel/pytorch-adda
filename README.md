# PyTorch-ADDA
A PyTorch implementation for [Adversarial Discriminative Domain Adaptation](https://arxiv.org/abs/1702.05464).

## Environment
- Python 3.6
- PyTorch 0.2.0

## Usage

I only test on MNIST -> USPS, you can just run the following command:

```shell
python3 main.py
```

## Result

|                                    | MNIST (Source) | USPS (Target) |
| ---------------------------------- | -------------- | ------------- |
| Source Encoder + Source Classifier | 99.140000%     | 83.978495%    |
| Target Encoder + Source Classifier |                | 97.634409%    |

Domain Adaptation does work (97% vs 83%).