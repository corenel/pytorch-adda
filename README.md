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

## Network

In this experiment, I use three types of network. They are very simple.

- LeNet encoder

  ```
  LeNetEncoder (
    (encoder): Sequential (
      (0): Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
      (1): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
      (2): ReLU ()
      (3): Conv2d(20, 50, kernel_size=(5, 5), stride=(1, 1))
      (4): Dropout2d (p=0.5)
      (5): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
      (6): ReLU ()
    )
    (fc1): Linear (800 -> 500)
  )
  ```

- LeNet classifier

  ```
  LeNetClassifier (
    (fc2): Linear (500 -> 10)
  )
  ```

- Discriminator

  ```
  Discriminator (
    (layer): Sequential (
      (0): Linear (500 -> 500)
      (1): ReLU ()
      (2): Linear (500 -> 500)
      (3): ReLU ()
      (4): Linear (500 -> 2)
      (5): LogSoftmax ()
    )
  )
  ```

## Result

|                                    | MNIST (Source) | USPS (Target) |
| :--------------------------------: | :------------: | :-----------: |
| Source Encoder + Source Classifier |   99.140000%   |  83.978495%   |
| Target Encoder + Source Classifier |                |  97.634409%   |

Domain Adaptation does work (97% vs 83%).