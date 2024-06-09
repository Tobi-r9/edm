#!/bin/bash
#SBATCH --account=obdifflearn    
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks=4                # Number of tasks
#SBATCH --cpus-per-task=8        # Number of CPU cores per task (adjust as needed)
#SBATCH --gres=gpu:1                  # Number of GPUs per node
#SBATCH --time=00:15:00               # Time limit hrs:min:sec


jutil env activate -p obdifflearn
source $PROJECT/thoeppe/envs/sc_venv_template/activate.sh
cd $PROJECT/thoeppe/edm

srun --ntasks=1 --gres=gpu:1 --cpus-per-task=2 python classifier/cosine_eval.py --data_path $PROJECT/thoeppe/edm/samples/reconstructions/1.9233398370400518/data.zip --model_path $PROJECT/thoeppe/edm/classifier/models/classifier.pth --batch_size 256 &
srun --ntasks=1 --gres=gpu:1 --cpus-per-task=2 python classifier/cosine_eval.py --data_path $PROJECT/thoeppe/edm/samples/reconstructions/80.0/data.zip --model_path $PROJECT/thoeppe/edm/classifier/models/classifier.pth --batch_size 256 & 
wait
