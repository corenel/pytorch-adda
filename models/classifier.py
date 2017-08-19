"""Classifier model for ADDA."""

import torch.nn.functional as F
from torch import nn


class Classifier(nn.Module):
    """Classifier model for source domain."""

    def __init__(self, num_channels, conv_dims, num_classes, fc_dims):
        """Init classifier."""
        super(Classifier, self).__init__()

        self.num_channels = num_channels
        self.conv_dims = conv_dims
        self.num_classes = num_classes
        self.fc_dims = fc_dims
        self.restored = False

        self.encoder = nn.Sequential(
            # 1st conv layer
            # input [num_channels x 28 x 28]
            # output [conv_dims x 12 x 12]
            nn.Conv2d(num_channels, conv_dims, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            # 2nd conv layer
            # input [conv_dims x 12 x 12]
            # output [(conv_dims*2) x 4 x 4]
            nn.Conv2d(conv_dims, conv_dims * 2, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
        )
        self.fc1 = nn.Linear((conv_dims * 2) * 4 * 4, fc_dims)
        self.fc2 = nn.Linear(fc_dims, num_classes)

    def forward(self, input):
        """Forward the classifier."""
        conv_out = self.encoder(input)
        feature = self.fc1(conv_out.view(-1, (self.conv_dims * 2) * 4 * 4))
        out = F.dropout(F.relu(feature), training=self.training)
        out = F.log_softmax(self.fc2(out))
        return feature, out
