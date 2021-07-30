----
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

-----
OUTPUT STRUCTURE:

You have given the following as arguments: dump_dir, dataset, model, technique, fold_no
The output for this run will be written to: dump_dir/dataset/model/technique/fold_no
Two folders:
1. models: This will contain train_<epoch>.pth files, along with train_best.pth file. Check "save_checkpoint" function in train.py to see what is stored.
2. logs: This will contain log.json
    log.json will have a list of dictionaries, one for each epoch, with stats. Test epoch has epoch = -1
-----
PRETRAINING:

For RL: Ensure you have run techinque = "cc_loss"
For Generative Modeeling: Ensure you have technique = cc_loss_<optimizer>_<lr>_<weight_decay>

-----
EXAMPLE RUNS:

A. LOSS FUNCTIONS:

python train.py   --technique cc_loss --datasets "lost,MSRCv2,BirdSong,Soccer Player" --fold_no 3 --dump_dir results/test --model 3layer --dataset_folder datasets

python train.py   --technique min_loss --datasets "Yahoo! News" --fold_no 3 --dump_dir results/test --model 1layer --dataset_folder datasets --lr 0.1
--optimzer SGD

python train.py   --technique iexplr_loss --datasets "lost" --fold_no 6 --dump_dir results/test --model 1layer --dataset_folder datasets --optimzer SGD --weight_decay 0.0001

B. REINFORCEMENT LEARNING:

python train.py   --technique linear_rl --datasets "lost,MSRCv2,BirdSong,Soccer Player" --fold_no 3 --dump_dir results/test --model 3layer --dataset_folder datasets --pretrain_p 1 --pretrain_p_perc 90

C. GENERATIVE MODELLING:

LSTM:
python train.py   --technique weighted_loss_xy_lstm_iexplr_Adam_0.01_1e-06 --datasets "BirdSong,Soccer Player" --fold_no 0 --dump_dir results/test --model 3layer --freeze50 1 --pretrain_p 1  --pretrain_p_perc 100 --pretrain_g 1 --optimizer Adam --lr 0.01 --weight_decay 1e-06 --batch_size 64 --dataset_folder datasets

Y DEPENDENCE (MATRIX MODEL):
python train.py   --technique weighted_loss_y_Adam_0.01_1e-06 --datasets "BirdSong,Soccer Player" --fold_no 0 --dump_dir results/test --model 3layer --pretrain_p 1  --pretrain_p_perc 100 --optimizer Adam --lr 0.01 --weight_decay 1e-06 --batch_size 64 --dataset_folder datasets

-----
ANALYSIS

notebooks/Analyis.ipynb will read log files