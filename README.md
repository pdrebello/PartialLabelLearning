train.py arguments

dump_dir: Directory where output logs, models will be written

dataset_folder: Folder where datasets are located

datasets: String of comma separated names of datasets. e.g 'lost,MSRCv2,Yahoo! News'

fold_no: Fold number

technique: {cc_loss, iexplr_loss, min_loss, naive_loss, linear_rl, exponential_rl, weighted_cc_loss_y, weighted_cc_loss_xy, weighted_cc_loss_xy_lstm, weighted_iexplr_loss_y, weighted_iexplr_loss_xy, weighted_iexplr_loss_xy_lstm}
model: Model for Prediciton Network {1layer, 3layer}

lr: Learning Rate {float}

weight_decay: Weight decay {float}

optimizer: Optimizer {SGD, Adam}

batch_size: Batch Size for mini batch training {float}

val_metric: Metric for early stopping {'acc', 'loss'}

lambd: Regularization parameter for cc_loss + lambda * min_loss {float}

pretrain_p: Pretrain Prediction Network in RL {0,1}

pretrain_p_perc: Percentage of Baseline Accuracy to train P Network to {int}

pretrain_q: Pretrain Selection Network in RL {0,1}

pretrain: Pretrain Prediction Network in Generative Scheme {0,1}

neg_sample: Negative Sampling while pretraining Generative Model {0,1}

freeze50: Freeze prediction model for first 50 epochs of generative training {0,1}

-----
Dataset Manipulation and Loading is done in: dataset.py

-----
All models and networks are in: networks.py
