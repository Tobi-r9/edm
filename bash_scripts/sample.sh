#!/bin/bash

#SBATCH --job-name=hf_accelerate_job  # Job name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=1                    # Number of tasks
#SBATCH --cpus-per-task=8             # Number of CPU cores per task
#SBATCH --gres=gpu:4                  # Number of GPUs per node
#SBATCH --mem=32G                     # Memory per node
#SBATCH --time=02:00:00               # Time limit hrs:min:sec
#SBATCH --output=output_%j.log        # Standard output and error log

jutil env activate -p obdifflearn
source $PROJECT/thoeppe/envs/edm/activate.sh
salloc --gres=gpu:4 --partition=booster --time=00:10:00 --account obdifflearn
srun --ntasks=1 --ntasks-per-node=1 --cpus-per-task=8 torchrun --standalone --nproc_per_node=4 generate.py --outdir=$PROJECT/thoeppe/edm/samples --seeds=0 --batch=4 --network=$PROJECT/thoeppe/edm/models/edm-cifar10-32x32-uncond-vp.pkl
srun --ntasks=1 --ntasks-per-node=1 --cpus-per-task=8 torchrun --standalone --nproc_per_node=1 reconstruction/noise.py --save_path=$PROJECT/thoeppe/edm/samples/reconstructions --model_path=$PROJECT/thoeppe/edm/models/edm-cifar10-32x32-uncond-vp.pkl --data_path=$PROJECT/thoeppe/edm/datasets/cifar/train.zip


# multi-node
srun --ntasks=1 --ntasks-per-node=1 --cpus-per-task=8 torchrun --standalone --nproc_per_node=4 generate.py --outdir=$PROJECT/thoeppe/edm/samples --seeds=0 --batch=8 --network=$PROJECT/thoeppe/edm/models/edm-cifar10-32x32-uncond-vp.pkl
srun --ntasks --ntasks-per-node=1 --cpus-per-task=8 torchrun --standalone --nproc_per_node=4 train.py --outdir=$PROJECT/thoeppe/edm/test_run --data=$PROJECT/thoeppe/edm/datasets/cifar/train.zip --cond=0 --arch=ddpmpp

python classifier/train_edm.py --train_data_path $PROJECT/thoeppe/edm/datasets/cifar/train.zip --val_data_path $PROJECT/thoeppe/edm/datasets/cifar/test.zip --model_save_path $PROJECT/thoeppe/edm/classifier/models

srun --ntasks=1 --ntasks-per-node=1 --cpus-per-task=8 torchrun --standalone --nproc_per_node=1 classifier/train_edm.py --train_data_path $PROJECT/thoeppe/edm/datasets/cifar/train.zip --val_data_path $PROJECT/thoeppe/edm/datasets/cifar/test.zip --model_save_path $PROJECT/thoeppe/edm/classifier/models