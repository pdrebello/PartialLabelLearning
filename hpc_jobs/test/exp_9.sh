 
module load apps/anaconda/3
module load apps/pytorch/1.5.0/gpu/anaconda3
cd /home/cse/phd/csz178057/pratheek/PartialLabelLearning
touch /home/cse/phd/csz178057/hpcscratch/unification/hpc_run/misc-scripts/hpc/create_git/test/JACK_9
rm /home/cse/phd/csz178057/hpcscratch/unification/hpc_run/misc-scripts/hpc/create_git/test/JACK_9
export PATH="$(pwd)"/third_party/Jacinle/bin:$PATH
python /home/cse/phd/csz178057/pratheek/PartialLabelLearning/train.py   --technique cc_loss_Adam_0.01_1e-06 --datasets "lost,MSRCv2,BirdSong,Soccer Player" --fold_no 2 --dump_dir /home/cse/phd/csz178057/hpcscratch/unification/pll/results/test --model 3layer --dataset_folder /home/cse/phd/csz178057/hpcscratch/unification/pll/datasets --optimizer Adam --lr 0.01 --weight_decay 1e-06 &
pids[0]=$!
python /home/cse/phd/csz178057/pratheek/PartialLabelLearning/train.py   --technique cc_loss_Adam_0.01_1e-06 --datasets "lost,MSRCv2,BirdSong,Soccer Player" --fold_no 3 --dump_dir /home/cse/phd/csz178057/hpcscratch/unification/pll/results/test --model 3layer --dataset_folder /home/cse/phd/csz178057/hpcscratch/unification/pll/datasets --optimizer Adam --lr 0.01 --weight_decay 1e-06 &
pids[1]=$!
python /home/cse/phd/csz178057/pratheek/PartialLabelLearning/train.py   --technique cc_loss_Adam_0.01_1e-06 --datasets "lost,MSRCv2,BirdSong,Soccer Player" --fold_no 0 --dump_dir /home/cse/phd/csz178057/hpcscratch/unification/pll/results/test --model 3layer --dataset_folder /home/cse/phd/csz178057/hpcscratch/unification/pll/datasets --optimizer Adam --lr 0.01 --weight_decay 1e-06 &
pids[2]=$!
for pid in ${pids[*]}; do 
         wait $pid 
done

touch /home/cse/phd/csz178057/hpcscratch/unification/hpc_run/misc-scripts/hpc/create_git/test/JACK_9
