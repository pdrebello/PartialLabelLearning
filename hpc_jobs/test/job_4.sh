#PBS -N job_4_3layer_lin
#PBS -l walltime=2:00:00
#Name of job
#Dep name , project name
#PBS -P mausam.p2.27
#PBS -j oe
#PBS -l select=1:ngpus=1:ncpus=6
## SPECIFY JOB NOW

JOBNAME=HPRSR
CURTIME=$(date +%Y%m%d%H%M%S)
cd $PBS_O_WORKDIR
module load apps/pythonpackages/3.6.0/pytorch/0.4.1/gpu
module load apps/pytorch/1.5.0/gpu/anaconda3
## Change to dir from where script was launched




bash exp_4.sh
