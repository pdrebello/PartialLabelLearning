 
module load apps/anaconda/3
module load apps/pytorch/1.5.0/gpu/anaconda3
cd /home/cse/phd/csz178057/pratheek/PartialLabelLearning
touch /home/cse/phd/csz178057/hpcscratch/unification/hpc_run/misc-scripts/hpc/create_git/test/JACK_2
rm /home/cse/phd/csz178057/hpcscratch/unification/hpc_run/misc-scripts/hpc/create_git/test/JACK_2
export PATH="$(pwd)"/third_party/Jacinle/bin:$PATH
python /home/cse/phd/csz178057/pratheek/PartialLabelLearning/train.py   --technique weighted_loss_xy_lstm_iexplr_SGD_0.01_1e-05 --datasets "lost,MSRCv2,BirdSong,Soccer Player" --fold_no 6 --dump_dir /home/cse/phd/csz178057/hpcscratch/unification/pll/results/test --model 3layer --freeze50 1 --pretrain_p 1  --pretrain_p_perc 100 --pretrain_g 1 --optimizer SGD --lr 0.01 --weight_decay 1e-05 --batch_size 64 --dataset_folder /home/cse/phd/csz178057/hpcscratch/unification/pll/datasets &
pids[0]=$!
python /home/cse/phd/csz178057/pratheek/PartialLabelLearning/train.py   --technique weighted_loss_xy_lstm_iexplr_SGD_0.01_1e-05 --datasets "lost,MSRCv2,BirdSong,Soccer Player" --fold_no 4 --dump_dir /home/cse/phd/csz178057/hpcscratch/unification/pll/results/test --model 3layer --freeze50 1 --pretrain_p 1  --pretrain_p_perc 100 --pretrain_g 1 --optimizer SGD --lr 0.01 --weight_decay 1e-05 --batch_size 64 --dataset_folder /home/cse/phd/csz178057/hpcscratch/unification/pll/datasets &
pids[1]=$!
python /home/cse/phd/csz178057/pratheek/PartialLabelLearning/train.py   --technique linear_rl --datasets "lost,MSRCv2,BirdSong,Soccer Player" --fold_no 0 --dump_dir /home/cse/phd/csz178057/hpcscratch/unification/pll/results/test --model 3layer --dataset_folder /home/cse/phd/csz178057/hpcscratch/unification/pll/datasets &
pids[2]=$!
for pid in ${pids[*]}; do 
         wait $pid 
done

touch /home/cse/phd/csz178057/hpcscratch/unification/hpc_run/misc-scripts/hpc/create_git/test/JACK_2
