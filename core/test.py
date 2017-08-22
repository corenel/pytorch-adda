"""Test script to classify target data."""

import torch
import torch.nn as nn

from utils import make_variable


def eval_tgt(model_src, model_tgt, data_loader):
    """Evaluation for target encoder by source classifier on target dataset."""
    print("=== Evaluating classifier for encoded target domain ===")
    model_src.eval()
    model_tgt.eval()
    loss = 0
    acc = 0
    criterion = nn.CrossEntropyLoss()

    for (images, labels) in data_loader:
        images = make_variable(images, volatile=True)
        labels = make_variable(labels).squeeze_()

        _, preds = model_tgt(images)
        loss += criterion(preds, labels).data[0]

        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(labels.data).cpu().sum()

    loss /= len(data_loader)
    acc /= len(data_loader.dataset)

    print("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))
