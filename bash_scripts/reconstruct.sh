#!/bin/bash
#SBATCH --account=obdifflearn    
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks=4                # Number of tasks
#SBATCH --cpus-per-task=8        # Number of CPU cores per task (adjust as needed)
#SBATCH --gres=gpu:1              # Number of GPUs per task (2 GPUs per task)
#SBATCH --time=01:00:00           # Time limit hrs:min:sec

jutil env activate -p obdifflearn
source $PROJECT/thoeppe/envs/sc_venv_template/activate.sh
cd $PROJECT/thoeppe/edm


SAMPLING_PARAMETERS="--batch_size 512 --num_steps 18 --rho 7 --max_files 10000"
srun --ntasks=1 --gres=gpu:1 --cpus-per-task=2 torchrun --standalone --nproc_per_node=1 reconstruction/reconstruct.py --image_path=$PROJECT/thoeppe/edm/samples/reconstructions --model_path=$PROJECT/thoeppe/edm/models/edm-cifar10-32x32-uncond-vp.pkl --data_path=$PROJECT/thoeppe/edm/datasets/cifar/train.zip $SAMPLING_PARAMETERS --start_step 0 --skip 4 &
srun --ntasks=1 --gres=gpu:1 --cpus-per-task=2 torchrun --standalone --nproc_per_node=1 reconstruction/reconstruct.py --image_path=$PROJECT/thoeppe/edm/samples/reconstructions --model_path=$PROJECT/thoeppe/edm/models/edm-cifar10-32x32-uncond-vp.pkl --data_path=$PROJECT/thoeppe/edm/datasets/cifar/train.zip $SAMPLING_PARAMETERS --start_step 1 --skip 4 &
srun --ntasks=1 --gres=gpu:1 --cpus-per-task=2 torchrun --standalone --nproc_per_node=1 reconstruction/reconstruct.py --image_path=$PROJECT/thoeppe/edm/samples/reconstructions --model_path=$PROJECT/thoeppe/edm/models/edm-cifar10-32x32-uncond-vp.pkl --data_path=$PROJECT/thoeppe/edm/datasets/cifar/train.zip $SAMPLING_PARAMETERS --start_step 2 --skip 4 &
srun --ntasks=1 --gres=gpu:1 --cpus-per-task=2 torchrun --standalone --nproc_per_node=1 reconstruction/reconstruct.py --image_path=$PROJECT/thoeppe/edm/samples/reconstructions --model_path=$PROJECT/thoeppe/edm/models/edm-cifar10-32x32-uncond-vp.pkl --data_path=$PROJECT/thoeppe/edm/datasets/cifar/train.zip $SAMPLING_PARAMETERS --start_step 3 --skip 4 &
wait

