#!/bin/bash
#SBATCH --account=obdifflearn    
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks=4                # Number of tasks
#SBATCH --cpus-per-task=8        # Number of CPU cores per task (adjust as needed)
#SBATCH --gres=gpu:1                  # Number of GPUs per node
#SBATCH --time=00:10:00               # Time limit hrs:min:sec


jutil env activate -p obdifflearn
source $PROJECT/thoeppe/envs/sc_venv_template/activate.sh
cd $PROJECT/thoeppe/edm

srun --ntasks=1 --gres=gpu:1 --cpus-per-task=2 python classifier/cosine_eval.py --data_path $PROJECT/thoeppe/edm/samples/reconstructions/0.05994731123547158/data.zip --model_path $PROJECT/thoeppe/edm/classifier/models/classifier.pth --get_activations 0 --reference_path $PROJECT/thoeppe/edm/samples/reconstructions/0/activations &
srun --ntasks=1 --gres=gpu:1 --cpus-per-task=2 python classifier/cosine_eval.py --data_path $PROJECT/thoeppe/edm/samples/reconstructions/0.2964422844791578/data.zip --model_path $PROJECT/thoeppe/edm/classifier/models/classifier.pth --get_activations 0 --reference_path $PROJECT/thoeppe/edm/samples/reconstructions/0/activations & 
srun --ntasks=1 --gres=gpu:1 --cpus-per-task=2 python classifier/cosine_eval.py --data_path $PROJECT/thoeppe/edm/samples/reconstructions/1.088170636545279/data.zip --model_path $PROJECT/thoeppe/edm/classifier/models/classifier.pth --get_activations 0 --reference_path $PROJECT/thoeppe/edm/samples/reconstructions/0/activations &
srun --ntasks=1 --gres=gpu:1 --cpus-per-task=2 python classifier/cosine_eval.py --data_path $PROJECT/thoeppe/edm/samples/reconstructions/1.9233398370400518/data.zip --model_path $PROJECT/thoeppe/edm/classifier/models/classifier.pth --get_activations 0 --reference_path $PROJECT/thoeppe/edm/samples/reconstructions/0/activations &
wait

srun --ntasks=1 --gres=gpu:1 --cpus-per-task=2 python classifier/cosine_eval.py --data_path $PROJECT/thoeppe/edm/samples/reconstructions/28.37458460415684/data.zip --model_path $PROJECT/thoeppe/edm/classifier/models/classifier.pth --get_activations 0 --reference_path $PROJECT/thoeppe/edm/samples/reconstructions/0/activations &
srun --ntasks=1 --gres=gpu:1 --cpus-per-task=2 python classifier/cosine_eval.py --data_path $PROJECT/thoeppe/edm/samples/reconstructions/40.78557379650796/data.zip --model_path $PROJECT/thoeppe/edm/classifier/models/classifier.pth --get_activations 0 --reference_path $PROJECT/thoeppe/edm/samples/reconstructions/0/activations & 
srun --ntasks=1 --gres=gpu:1 --cpus-per-task=2 python classifier/cosine_eval.py --data_path $PROJECT/thoeppe/edm/samples/reconstructions/57.58598472124816/data.zip --model_path $PROJECT/thoeppe/edm/classifier/models/classifier.pth --get_activations 0 --reference_path $PROJECT/thoeppe/edm/samples/reconstructions/0/activations &
srun --ntasks=1 --gres=gpu:1 --cpus-per-task=2 python classifier/cosine_eval.py --data_path $PROJECT/thoeppe/edm/samples/reconstructions/8.400935309099825/data.zip --model_path $PROJECT/thoeppe/edm/classifier/models/classifier.pth --get_activations 0 --reference_path $PROJECT/thoeppe/edm/samples/reconstructions/0/activations &
wait

srun --ntasks=1 --gres=gpu:1 --cpus-per-task=2 python classifier/cosine_eval.py --data_path $PROJECT/thoeppe/edm/samples/reconstructions/0.0075280199627840785/data.zip --model_path $PROJECT/thoeppe/edm/classifier/models/classifier.pth --get_activations 0 --reference_path $PROJECT/thoeppe/edm/samples/reconstructions/0/activations &
srun --ntasks=1 --gres=gpu:1 --cpus-per-task=2 python classifier/cosine_eval.py --data_path $PROJECT/thoeppe/edm/samples/reconstructions/0.13951646873101678/data.zip --model_path $PROJECT/thoeppe/edm/classifier/models/classifier.pth --get_activations 0 --reference_path $PROJECT/thoeppe/edm/samples/reconstructions/0/activations & 
srun --ntasks=1 --gres=gpu:1 --cpus-per-task=2 python classifier/cosine_eval.py --data_path $PROJECT/thoeppe/edm/samples/reconstructions/0.5853481231945422/data.zip --model_path $PROJECT/thoeppe/edm/classifier/models/classifier.pth --get_activations 0 --reference_path $PROJECT/thoeppe/edm/samples/reconstructions/0/activations &
srun --ntasks=1 --gres=gpu:1 --cpus-per-task=2 python classifier/cosine_eval.py --data_path $PROJECT/thoeppe/edm/samples/reconstructions/12.910082380757322/data.zip --model_path $PROJECT/thoeppe/edm/classifier/models/classifier.pth --get_activations 0 --reference_path $PROJECT/thoeppe/edm/samples/reconstructions/0/activations &
wait

srun --ntasks=1 --gres=gpu:1 --cpus-per-task=2 python classifier/cosine_eval.py --data_path $PROJECT/thoeppe/edm/samples/reconstructions/19.352452980325225/data.zip --model_path $PROJECT/thoeppe/edm/classifier/models/classifier.pth --get_activations 0 --reference_path $PROJECT/thoeppe/edm/samples/reconstructions/0/activations &
srun --ntasks=1 --gres=gpu:1 --cpus-per-task=2 python classifier/cosine_eval.py --data_path $PROJECT/thoeppe/edm/samples/reconstructions/3.256821519765537/data.zip --model_path $PROJECT/thoeppe/edm/classifier/models/classifier.pth --get_activations 0 --reference_path $PROJECT/thoeppe/edm/samples/reconstructions/0/activations & 
srun --ntasks=1 --gres=gpu:1 --cpus-per-task=2 python classifier/cosine_eval.py --data_path $PROJECT/thoeppe/edm/samples/reconstructions/5.315194521796383/data.zip --model_path $PROJECT/thoeppe/edm/classifier/models/classifier.pth --get_activations 0 --reference_path $PROJECT/thoeppe/edm/samples/reconstructions/0/activations &
srun --ntasks=1 --gres=gpu:1 --cpus-per-task=2 python classifier/cosine_eval.py --data_path $PROJECT/thoeppe/edm/samples/reconstructions/80.0/data.zip --model_path $PROJECT/thoeppe/edm/classifier/models/classifier.pth --get_activations 0 --reference_path $PROJECT/thoeppe/edm/samples/reconstructions/0/activations &
wait
