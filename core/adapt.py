"""Adversarial adaptation to train target encoder."""

import os

import torch
import torch.optim as optim
from torch import nn

import params
from utils import make_variable


def train_tgt(model_src, model_tgt, model_critic,
              src_data_loader, tgt_data_loader):
    """Train encoder for target domain."""
    ####################
    # 1. setup network #
    ####################

    # welcome message
    print("=== Training encoder for target domain ===")
    # set train state for Dropout and BN layers
    model_tgt.train()
    model_critic.train()
    # print model architecture
    print(model_tgt)
    print(model_critic)

    # no need to compute gradients for source model
    for p in model_src.parameters():
        p.requires_grad = False

    # setup criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer_tgt = optim.Adam(model_tgt.parameters(),
                               lr=params.c_learning_rate,
                               betas=(params.beta1, params.beta2))
    optimizer_critic = optim.Adam(model_tgt.parameters(),
                                  lr=params.d_learning_rate,
                                  betas=(params.beta1, params.beta2))
    len_data_loader = min(len(src_data_loader), len(tgt_data_loader))

    ####################
    # 2. train network #
    ####################

    for epoch in range(params.num_epochs):
        # zip source and target data pair
        data_zip = enumerate(zip(src_data_loader, tgt_data_loader))
        for step, ((images_src, _), (images_tgt, _)) in data_zip:
            ###########################
            # 2.1 train discriminator #
            ###########################

            images_src = make_variable(images_src)
            images_tgt = make_variable(images_tgt)

            optimizer_tgt.zero_grad()
            optimizer_critic.zero_grad()

            feat_src, _ = model_src(images_src)
            feat_tgt, _ = model_tgt(images_tgt)
            feat_concat = torch.cat((feat_src, feat_tgt), 0)

            label_concat = torch.cat((
                make_variable(torch.zeros(feat_concat.size(0) // 2).long()),
                make_variable(torch.ones(feat_concat.size(0) // 2).long())
            ), 0)

            pred_concat = model_critic(feat_concat)
            loss_critic = criterion(pred_concat, label_concat)
            loss_critic.backward(retain_graph=True)

            optimizer_critic.step()

            pred_cls = torch.squeeze(pred_concat.max(1)[1])
            acc = (pred_cls == label_concat).float().mean()

            # train target encoder
            optimizer_tgt.zero_grad()
            optimizer_critic.zero_grad()

            loss_tgt = criterion(
                feat_concat[feat_concat.size(0) // 2:, ...],
                make_variable(torch.ones(feat_concat.size(0) // 2).long())
            )
            loss_tgt.backward()

            optimizer_tgt.step()

            # print step info
            if ((step + 1) % params.log_step == 0):
                print("Epoch [{}/{}] Step [{}/{}]:"
                      "d_loss={:.3f} g_loss={:.3f} acc={:.3f}"
                      .format(epoch + 1,
                              params.num_epochs,
                              step + 1,
                              len_data_loader,
                              loss_critic.data[0],
                              loss_tgt.data[0],
                              acc.data[0]))

        # save model parameters
        if ((epoch + 1) % params.save_step == 0):
            if not os.path.exists(params.model_root):
                os.makedirs(params.model_root)
            torch.save(model_critic.state_dict(), os.path.join(
                params.model_root,
                "ADDA-critic-{}.pt".format(epoch + 1)))
            torch.save(model_tgt.state_dict(), os.path.join(
                params.model_root,
                "ADDA-target-{}.pt".format(epoch + 1)))
