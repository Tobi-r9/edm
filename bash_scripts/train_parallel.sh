#!/bin/bash
#SBATCH --account=obdifflearn    
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks=2                # Number of tasks
#SBATCH --cpus-per-task=10        # Number of CPU cores per task (adjust as needed)
#SBATCH --gres=gpu:2              # Number of GPUs per task (2 GPUs per task)
#SBATCH --time=02:30:00           # Time limit hrs:min:sec

jutil env activate -p obdifflearn
source $PROJECT/thoeppe/envs/sc_venv_template/activate.sh
cd $PROJECT/thoeppe/edm

# Run the first model with SGD
srun --ntasks=1 --gres=gpu:2 --cpus-per-task=5 python classifier/train_edm.py --train_data_path $PROJECT/thoeppe/edm/datasets/cifar/train.zip --val_data_path $PROJECT/thoeppe/edm/datasets/cifar/test.zip --model_save_path $PROJECT/thoeppe/edm/classifier/models --opt sgd --scheduler_name step  &

# Run the second model with Adam
srun --ntasks=1 --gres=gpu:2 --cpus-per-task=5 python classifier/train_edm.py --train_data_path $PROJECT/thoeppe/edm/datasets/cifar/train.zip --val_data_path $PROJECT/thoeppe/edm/datasets/cifar/test.zip --model_save_path $PROJECT/thoeppe/edm/classifier/models --opt sgd --scheduler_name cosine &

wait