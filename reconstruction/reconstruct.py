import sys
sys.path.insert(0, '/proj/berzelius-2021-89/users/x_tohop/edm')
import torch
import os
import numpy as np
import pickle
from image_datasets import load_data
from PIL import Image
import concurrent.futures
import torch_utils
import dnnlib


def log_gpu_usage():
    os.system("nvidia-smi")

def edm_sampler(
    net, images, t_steps, class_labels=None, S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, randn_like=torch.randn_like
):

    num_steps = len(t_steps) - 1

    # Main sampling loop.
    x_next = images + torch.randn_like(images) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

def get_discretization_steps(model, device, num_steps=18, sigma_min=0.002, sigma_max=80, rho=7):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, model.sigma_min)
    sigma_max = min(sigma_max, model.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([model.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
    return t_steps

def save_image(image, save_path):
    pil_img = Image.fromarray(image)
    pil_img.save(save_path)

def save_images_parallel(img_arr, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    save_paths = [os.path.join(save_dir, f'image_{i}.png') for i in range(len(img_arr))]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(save_image, img_arr, save_paths)


def main(model_path, data_path, image_path, batch_size=256, num_steps=18, rho=7, max_files=10000, seed=None, start_step=0, skip=1):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)['ema'].to(device)
    model.eval()

    t_steps = get_discretization_steps(model=model, device=device, num_steps=num_steps, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho)
    for step in range(start_step, num_steps - 1, skip):
        dataloader = load_data(
            data_dir=data_path,
            batch_size=batch_size,
            image_size=32,
            class_cond=False,
            random_crop=False,
            random_flip=False,
            deterministic=True,
        )
        num_files = 0
        reconstructions = []
        save_dir = os.path.join(image_path, f"{round(t_steps[step].item(), 2)}")
        if seed is not None:
            torch.manual_seed(seed)
        while num_files < max_files:
            num_files += batch_size
            images, _ = next(dataloader)
            denoised = edm_sampler(model, images.to(device), t_steps[step:])
            # log_gpu_usage()
            denoised = (denoised * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            reconstructions.append(denoised)

        reconstructions = np.array(reconstructions)
        n, b, d1, d2, d3 = reconstructions.shape
        reconstructions = reconstructions.reshape(-1, d1, d2, d3)
        save_images_parallel(reconstructions, save_dir)
        print(f"finished step: {step}")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Hyperparameters for sampling')
    parser.add_argument('--model_path', type=str, required=True, help='the path to model')
    parser.add_argument('--data_path', type=str, required=True, help='path to data')
    parser.add_argument('--image_path', type=str, required=True, help='path to save denoised images')
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_steps", type=int, default=18, help="number of sampling steps for edm")
    parser.add_argument("--rho", type=int, default=7, help="parameter for step distribution")
    parser.add_argument("--max_files", type=int, default=10000, help="maximum number of images to reconstruct")
    parser.add_argument("--seed", type=int, default=None, help="fix seed to get the same reconstructions")
    parser.add_argument("--start_step", type=int, default=0, help="only change for parallel computing")
    parser.add_argument("--skip", type=int, default=1, help="only change for parallel computing")
    args = parser.parse_args()

    main(**vars(args))