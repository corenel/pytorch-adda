"""Main script for ADDA."""

import params
from core import eval_src, eval_tgt, train_src, train_tgt
from models import Classifier, Discriminator
from utils import get_data_loader, init_model, init_random_seed

if __name__ == '__main__':
    # init random seed
    init_random_seed(params.manual_seed)

    # load dataset
    src_data_loader = get_data_loader(params.src_dataset)
    src_data_loader_eval = get_data_loader(params.src_dataset, train=False)
    tgt_data_loader = get_data_loader(params.tgt_dataset)
    tgt_data_loader_eval = get_data_loader(params.tgt_dataset, train=False)

    # load models
    model_src = init_model(net=Classifier(num_channels=params.num_channels,
                                          conv_dims=params.c_conv_dims,
                                          num_classes=params.num_classes,
                                          fc_dims=params.c_fc_dims),
                           restore=params.src_model_restore)
    model_tgt = init_model(net=Classifier(num_channels=params.num_channels,
                                          conv_dims=params.c_conv_dims,
                                          num_classes=params.num_classes,
                                          fc_dims=params.c_fc_dims),
                           restore=params.tgt_model_restore)
    D = init_model(Discriminator(input_dims=params.d_input_dims,
                                 hidden_dims=params.d_hidden_dims,
                                 output_dims=params.d_output_dims),
                   restore=params.d_model_restore)

    # train and eval source model
    if not (model_src.restored and params.src_model_trained):
        model_src = train_src(model_src, src_data_loader)
    eval_src(model_src, src_data_loader_eval)

    # train target encoder by GAN
    if not (model_tgt.restored and params.tgt_model_trained):
        model_tgt = train_tgt(model_src, model_tgt, tgt_data_loader)

    # eval target encoder on test set of target dataset
    eval_tgt(model_src, model_tgt, tgt_data_loader_eval)
