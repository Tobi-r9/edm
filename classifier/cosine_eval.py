import sys
sys.path.insert(0, '/p/project/obdifflearn/thoeppe/edm')
import torch as th
import numpy as np
import os
from argparse import ArgumentParser
from reconstruction.image_datasets import load_data
from cifar10_models.resnet import resnet18
from torch.nn import DataParallel


def get_activations(model, data_loader, batch_size, device, save_path):
    num_files = 0
    activations = {}
    for i in range(6):
        activations[f"act_{i}"] = []
    with th.no_grad():
        while num_files < 10000:
            num_files += batch_size
            images, _ = next(data_loader)
            images = images.to(device)
            act_5, act_4, act_3, act_2, act_1, act_0 = model(images)
            for i, act in enumerate([act_0, act_1, act_2, act_3, act_4, act_5]):
                activations[f"act_{i}"].append(act.detach().cpu().numpy())

    for i in range(6):
        act = np.array(activations[f"act_{i}"])
        it, b = act.shape[:2]
        new_shape = (it*b, ) + act.shape[2:]
        act = act.reshape(new_shape)
        np.savez(os.path.join(save_path, f"act_{i}"), act)

def load_activations(path):
    activations = np.load(path)["arr_0"]
    n = activations.shape[0]
    activations = activations.reshape(n, -1)
    return activations

def compute_cosine_similarity(act1, act2):
    # Normalize the arrays
    norm1 = np.linalg.norm(act1, axis=1)
    norm2 = np.linalg.norm(act2, axis=1)
    # Compute dot product and cosine similarity
    dot_product = np.sum(act1 * act2, axis=1)
    cosine_similarity = dot_product / (norm1 * norm2)
    return cosine_similarity


def main(model_path, data_path, batch_size, get_activations=True, reference_path=""):
    dirname = os.path.dirname(data_path)
    timestep = dirname.split("/")[-1]
    if get_activations:
    
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        data_loader = load_data(
            data_dir=data_path,
            batch_size=batch_size,
            image_size=32,
            class_cond=True,
            random_crop=False,
            random_flip=False,
            deterministic=True,
        )

        model = resnet18(pretrained=False)
        model = DataParallel(model)
        state_dict = th.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        save_path = os.path.join(dirname, "activations")
        os.makedirs(save_path, exist_ok=True)
        get_activations(model, data_loader, batch_size, device, save_path)

    else:
        activation_folder = os.path.join(dirname, "activations")
        save_dir = os.path.join(reference_path, f"cosine_similarities")
        os.makedirs(save_dir, exist_ok=True)
        for i in range(6):
            # load reference
            reference_file = os.path.join(reference_path, f"act_{i}.npz")
            reference_activations = load_activations(reference_file)

            # load reconstructions
            activations_file = os.path.join(activation_folder, f"act_{i}.npz")
            activations = load_activations(activations_file)
            
            #compute cosine
            cosine_similarity = compute_cosine_similarity(activations, reference_activations)
            np.savez(os.path.join(save_dir, f"css_{timestep}_{i}"), cosine_similarity)




if __name__ == "__main__":

    parser = ArgumentParser()

    # Add arguments
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Training and validation batch size"
    )
    parser.add_argument(
        "--get_activations", type=int, default=1, help="get activations or just compute cosine"
    )
    parser.add_argument("--data_path", type=str, help="Path to validation data")
    parser.add_argument("--model_path", type=str, help="Path to save model")
    parser.add_argument("--reference_path", type=str, help="Path to reference activations")

    # Parse arguments
    args = parser.parse_args()

    main(**vars(args))
