#!/bin/bash
#SBATCH --account=obdifflearn         
#SBATCH --job-name=hf_accelerate_job  # Job name
#SBATCH --gres=gpu:4                  # Number of GPUs per node
#SBATCH --time=02:00:00               # Time limit hrs:min:sec


jutil env activate -p obdifflearn
source $PROJECT/thoeppe/envs/sc_venv_template/activate.sh
cd $PROJECT/thoeppe/edm
srun --ntasks=1 --gres=gpu:4 --cpus-per-task=8 python classifier/train_edm.py --train_data_path $PROJECT/thoeppe/edm/datasets/cifar/train.zip --val_data_path $PROJECT/thoeppe/edm/datasets/cifar/test.zip --model_save_path $PROJECT/thoeppe/edm/classifier/models --opt sgd --scheduler_name warmup