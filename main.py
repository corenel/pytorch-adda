"""Main script for ADDA."""

import params
from core import eval_src, eval_tgt, train_src, train_tgt
from models import Discriminator, LeNetClassifier, LeNetEncoder
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
    src_encoder = init_model(net=LeNetEncoder(),
                             restore=params.src_encoder_restore)
    classifier_src = init_model(net=LeNetClassifier(),
                                restore=params.src_classifier_restore)
    tgt_encoder = init_model(net=LeNetEncoder(),
                             restore=params.tgt_encoder_restore)
    critic = init_model(Discriminator(input_dims=params.d_input_dims,
                                      hidden_dims=params.d_hidden_dims,
                                      output_dims=params.d_output_dims),
                        restore=params.d_model_restore)

    # train and eval source model
    if not (src_encoder.restored and classifier_src.restored and
            params.src_model_trained):
        model_src = train_src(src_encoder, classifier_src, src_data_loader)
    eval_src(src_encoder, classifier_src, src_data_loader_eval)

    # train target encoder by GAN
    # if not (tgt_encoder.restored and params.tgt_encoder_trained):
    #     model_tgt = train_tgt(src_encoder, tgt_encoder, critic,
    #                           src_data_loader, tgt_data_loader)

    # eval target encoder on test set of target dataset
    # eval_tgt(classifier_src, tgt_encoder, tgt_data_loader_eval)
