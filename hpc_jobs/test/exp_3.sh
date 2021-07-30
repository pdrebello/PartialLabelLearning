 
module load apps/anaconda/3
module load apps/pytorch/1.5.0/gpu/anaconda3
cd /home/cse/phd/csz178057/pratheek/PartialLabelLearning
touch /home/cse/phd/csz178057/hpcscratch/unification/hpc_run/misc-scripts/hpc/create_git/test/JACK_3
rm /home/cse/phd/csz178057/hpcscratch/unification/hpc_run/misc-scripts/hpc/create_git/test/JACK_3
export PATH="$(pwd)"/third_party/Jacinle/bin:$PATH
python /home/cse/phd/csz178057/pratheek/PartialLabelLearning/train.py   --technique linear_rl --datasets "lost,MSRCv2,BirdSong,Soccer Player" --fold_no 1 --dump_dir /home/cse/phd/csz178057/hpcscratch/unification/pll/results/test --model 3layer --dataset_folder /home/cse/phd/csz178057/hpcscratch/unification/pll/datasets &
pids[0]=$!
python /home/cse/phd/csz178057/pratheek/PartialLabelLearning/train.py   --technique linear_rl --datasets "lost,MSRCv2,BirdSong,Soccer Player" --fold_no 2 --dump_dir /home/cse/phd/csz178057/hpcscratch/unification/pll/results/test --model 3layer --dataset_folder /home/cse/phd/csz178057/hpcscratch/unification/pll/datasets &
pids[1]=$!
python /home/cse/phd/csz178057/pratheek/PartialLabelLearning/train.py   --technique linear_rl --datasets "lost,MSRCv2,BirdSong,Soccer Player" --fold_no 3 --dump_dir /home/cse/phd/csz178057/hpcscratch/unification/pll/results/test --model 3layer --dataset_folder /home/cse/phd/csz178057/hpcscratch/unification/pll/datasets &
pids[2]=$!
for pid in ${pids[*]}; do 
         wait $pid 
done

touch /home/cse/phd/csz178057/hpcscratch/unification/hpc_run/misc-scripts/hpc/create_git/test/JACK_3
