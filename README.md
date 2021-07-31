----
**train.py arguments**
```
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
```

-----
Dataset Manipulation and Loading is done in: dataset.py

-----
All models and networks are in: networks.py

-----
OUTPUT STRUCTURE:

You have given the following as arguments: dump_dir, dataset, model, technique, fold_no

The output for this run will be written to: dump_dir/dataset/model/technique/fold_no

Inside this, you will find two folders:
1. models: This will contain train_<epoch>.pth files, along with train_best.pth file. Check "save_checkpoint" function in train.py to see what is stored.
2. logs: This will contain log.json
    log.json will have a list of dictionaries, one for each epoch, with stats. Test epoch has epoch = -1
-----

**PRETRAINING:**

*For RL:* Training a selector module with RL assumes that we have already pre-trained a prediction model with technique = `cc_loss`. 
    The corresponding pre-trained checkpoint should be present under `dump_dir/dataset/model/cc_loss/<fold_no>`
    
*For Generative Modeling:* Similar to trainingg a selector module with RL, training a generative model also assumes that the prediction network has been pretrained.
    Corresponding checkpoint should be present under: `dump_dir/dataset/model/cc_loss_<optimizer>_<lr>_<weight_decay>/<fold_no>`

-----
EXAMPLE RUNS:

A. LOSS FUNCTIONS:
```
python train.py   --technique cc_loss --datasets "lost,MSRCv2,BirdSong,Soccer Player" --fold_no 3 --dump_dir results/test --model 3layer --dataset_folder datasets

python train.py   --technique min_loss --datasets "Yahoo! News" --fold_no 3 --dump_dir results/test --model 1layer --dataset_folder datasets --lr 0.1
--optimzer SGD

python train.py   --technique iexplr_loss --datasets "lost" --fold_no 6 --dump_dir results/test --model 1layer --dataset_folder datasets --optimzer SGD --weight_decay 0.0001
```
B. REINFORCEMENT LEARNING:
```
python train.py   --technique linear_rl --datasets "lost,MSRCv2,BirdSong,Soccer Player" --fold_no 3 --dump_dir results/test --model 3layer --dataset_folder datasets --pretrain_p 1 --pretrain_p_perc 90
```
Note that when pretrain_p_perc == 100, we load the saved cc_loss from from file. However when pretrain_p_perc < 100, we instead read the log file of cc_loss, determine the epoch to which to train, and do the pretraining within the run.
    
C. GENERATIVE MODELLING:

LSTM:
```
python train.py   --technique weighted_loss_xy_lstm_iexplr_Adam_0.01_1e-06 --datasets "BirdSong,Soccer Player" --fold_no 0 --dump_dir results/test --model 3layer --freeze50 1 --pretrain_p 1  --pretrain_p_perc 100 --pretrain_g 1 --optimizer Adam --lr 0.01 --weight_decay 1e-06 --batch_size 64 --dataset_folder datasets
```
    
Y DEPENDENCE (MATRIX MODEL):
 ```
python train.py   --technique weighted_loss_y_Adam_0.01_1e-06 --datasets "BirdSong,Soccer Player" --fold_no 0 --dump_dir results/test --model 3layer --pretrain_p 1  --pretrain_p_perc 100 --optimizer Adam --lr 0.01 --weight_decay 1e-06 --batch_size 64 --dataset_folder datasets
```
-----
JOBS

Inside `hpc_run` folder:

Modify `create.py` to include jobs required.

Then run:
```python create.py  -num_task_per_process <num_processes> -num_process_per_job <num_per_jobs> -task_script <train_script> -global_time <global_time> -dump_dir <dump_dir> -jobs_dir <jobs_dir>  -multi_header <multinode_header>```

Example Run:
```python create.py  -num_task_per_process 3 -num_process_per_job 6 -task_script /home/cse/phd/csz178057/pratheek/PartialLabelLearning/train.py -global_time 2 -dump_dir /home/cse/phd/csz178057/hpcscratch/unification/pll/results/test -jobs_dir test  -multi_header multinode_header.sh
```
    
-----
ANALYSIS

Inside `notebooks` folder:
    
**Analysis.ipynb**
  
will read log files, extract the test epoch and put into a pandas dataframe. 

**Input:** <dump_dir>
    
`log.json` should be present at the appropriate location, i.e. `<dump_dir>/<dataset>/<model>/<technique>/<fold_no>/logs/`
It reads the meta info from the path name, i.e. dataset name, model,  technique, and the fold number.


**Output:** 
The script computes various statistics of the following variables:
- 'real_test_acc',
- 'real_val_acc',
- 'real_train_acc', 
- 'surrogate_test_acc', 
- 'surrogate_val_acc', 
- 'surrogate_train_acc', 
- 'test_confidence' : max probability of prediction network, averaged across the test dataset,
- 'val_confidence' : max probability of prediction network, averaged across the val dataset, 
- 'train_confidence' : max probability of prediction network, averaged across the train dataset,

Further manipulation can be done in notebook.
    
**LSTM.ipynb**
    
This notebook adds Intersection over Union (IOU) scores for the LSTM model. You must input the directory from which to load the lstm model from, and it will compute and append IOUs to the logfile. Currenlty it is not rewriting the log.json, instead creating a new file called log.json_lstm.
    
The following are computed.
- 'train_IOU',
- 'train_IOU_neg',
- 'val_IOU',
- 'val_IOU_neg',
- 'test_IOU',
- 'test_IOU_neg',

Where train/val/test_IOU is computed on feeding the gold label to the LSTM.
    
While train/val/test_IOU_neg on feeding non-gold label to the LSTM.
