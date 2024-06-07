#!/bin/bash
#SBATCH --account=obdifflearn    
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks=4                # Number of tasks
#SBATCH --cpus-per-task=8        # Number of CPU cores per task (adjust as needed)
#SBATCH --gres=gpu:1                  # Number of GPUs per node
#SBATCH --time=01:00:00               # Time limit hrs:min:sec


jutil env activate -p obdifflearn
source $PROJECT/thoeppe/envs/sc_venv_template/activate.sh
cd $PROJECT/thoeppe/edm

srun --ntasks=1 --gres=gpu:1 --cpus-per-task=2 python classifier/eval_edm.py --data_path $PROJECT/thoeppe/edm/datasets/cifar/test.zip --model_path $PROJECT/thoeppe/edm/classifier/models/sgd/step --start_iteration 7500 --max_iteration 8600 &
srun --ntasks=1 --gres=gpu:1 --cpus-per-task=2 python classifier/eval_edm.py --data_path $PROJECT/thoeppe/edm/datasets/cifar/test.zip --model_path $PROJECT/thoeppe/edm/classifier/models/sgd/step/1e-3_step --start_iteration 9000 --max_iteration 10900 &

srun --ntasks=1 --gres=gpu:1 --cpus-per-task=2 python classifier/eval_edm.py --data_path $PROJECT/thoeppe/edm/datasets/cifar/test.zip --model_path $PROJECT/thoeppe/edm/classifier/models/sgd/cosine --start_iteration 7500 --max_iteration 8600 &
srun --ntasks=1 --gres=gpu:1 --cpus-per-task=2 python classifier/eval_edm.py --data_path $PROJECT/thoeppe/edm/datasets/cifar/test.zip --model_path $PROJECT/thoeppe/edm/classifier/models/sgd/cosine/1e-3_cosine --start_iteration 9500 --max_iteration 10900 &

wait