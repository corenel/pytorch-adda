"""Params for WGAN-GP."""

# params for dataset and data loader
data_root = "data"
dataset_mean_value = 0.5
dataset_std_value = 0.5
dataset_mean = (dataset_mean_value, dataset_mean_value, dataset_mean_value)
dataset_std = (dataset_std_value, dataset_std_value, dataset_std_value)
batch_size = 64
image_size = 64
num_classes = 10

# params for source dataset
src_dataset = "MNIST"
src_model_restore = "snapshots/classifier_src-100.pt"
src_model_trained = True

# params for target dataset
tgt_dataset = "USPS"
tgt_model_restore = None
tgt_model_trained = False

# params for setting up models
model_root = "snapshots"
num_channels = 1
c_conv_dims = 10
c_fc_dims = 50
d_input_dims = 50
d_hidden_dims = 512
d_output_dims = 2
d_model_restore = None

# params for training network
num_gpu = 1
num_epochs_pre = 100
num_epochs = 500
log_step = 20
save_step = 100
manual_seed = None

# params for optimizing models
d_learning_rate = 1e-4
c_learning_rate = 1e-4
beta1 = 0.5
beta2 = 0.9
