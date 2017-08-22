"""Pre-train encoder and classifier for source dataset."""

import os

import torch
import torch.nn as nn
import torch.optim as optim

import params
from utils import make_variable, save_model


def train_src(model, data_loader):
    """Train classifier for source domain."""
    print("=== Training classifier for source domain ===")
    print(model)

    model.train()
    optimizer = optim.Adam(model.parameters(),
                           lr=params.c_learning_rate,
                           betas=(params.beta1, params.beta2))
    criterion = nn.CrossEntropyLoss()

    for epoch in range(params.num_epochs_pre):
        for step, (images, labels) in enumerate(data_loader):
            images = make_variable(images)
            labels = make_variable(labels.squeeze_())

            optimizer.zero_grad()

            _, preds = model(images)
            loss = criterion(preds, labels)

            loss.backward()
            optimizer.step()

            if ((step + 1) % params.log_step == 0):
                print("Epoch [{}/{}] Step [{}/{}]: loss={}"
                      .format(epoch + 1,
                              params.num_epochs_pre,
                              step + 1,
                              len(data_loader),
                              loss.data[0]))

        if ((epoch + 1) % params.eval_step == 0):
            eval_src(model, data_loader, welcome_msg=False)

        if ((epoch + 1) % params.save_step == 0):
            save_model(model, "classifier_src-{}.pt".format(epoch + 1))

    save_model(model, "classifier_src-final.pt")

    return model


def eval_src(model, data_loader, welcome_msg=True):
    """Evaluate classifier for source domain."""
    if welcome_msg:
        print("=== Evaluating classifier for source domain ===")
        print(model)

    model.eval()
    loss = 0
    acc = 0
    criterion = nn.CrossEntropyLoss()

    for (images, labels) in data_loader:
        images = make_variable(images, volatile=True)
        labels = make_variable(labels)

        _, preds = model(images)
        loss += criterion(preds, labels).data[0]

        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(labels.data).cpu().sum()

    loss /= len(data_loader)
    acc /= len(data_loader.dataset)

    print("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))
