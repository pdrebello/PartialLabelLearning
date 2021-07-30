#PBS -N test_multi_job_2
#PBS -l walltime=2:00:00
#Name of job
#Dep name , project name
#PBS -P cse 

#PBS -j oe
#PBS -m bea
### Specify email address to use for notification.
#PBS -M $USER@iitd.ac.in
#PBS -l select=3:ngpus=2:ncpus=6
## SPECIFY JOB NOW

CURTIME=$(date +%Y%m%d%H%M%S)
module load apps/pythonpackages/3.6.0/pytorch/0.4.1/gpu
## Change to dir from where script was launched




count=12


declare -a var
init_count=$count 
while read p; do
      echo $p
      script="source /usr/share/Modules/3.2.10/init/bash && CUDA_VISIBLE_DEVICES=0 nohup /home/cse/phd/csz178057/hpcscratch/unification/hpc_run/misc-scripts/hpc/create_git/test/exp_${count}.sh > /home/cse/phd/csz178057/hpcscratch/unification/hpc_run/misc-scripts/hpc/create_git/test/LOG_${count} 2>&1 &"
      echo $script
      ssh -o StrictHostKeyChecking=no -n -f ${USER}@$p $script
      var[`expr $count - $init_count`]=/home/cse/phd/csz178057/hpcscratch/unification/hpc_run/misc-scripts/hpc/create_git/test/JACK_$count  
      count=`expr $count + 1`

      script="source /usr/share/Modules/3.2.10/init/bash && CUDA_VISIBLE_DEVICES=1 nohup /home/cse/phd/csz178057/hpcscratch/unification/hpc_run/misc-scripts/hpc/create_git/test/exp_${count}.sh > /home/cse/phd/csz178057/hpcscratch/unification/hpc_run/misc-scripts/hpc/create_git/test/LOG_${count} 2>&1 &"
      echo $script
      ssh -o StrictHostKeyChecking=no -n -f ${USER}@$p $script  
      var[`expr $count - $init_count`]=/home/cse/phd/csz178057/hpcscratch/unification/hpc_run/misc-scripts/hpc/create_git/test/JACK_$count  
      count=`expr $count + 1`
  
done <$PBS_NODEFILE

for i in "${var[@]}" 
do 
	echo $i 
    until [ -f $i ]
    do
        sleep 10
    done

done 



