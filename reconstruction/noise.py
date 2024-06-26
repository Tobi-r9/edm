import sys
sys.path.insert(0, '/p/project/obdifflearn/thoeppe/edm')
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle
from image_datasets import load_data
import PIL
import torch_utils
import dnnlib


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

def create_image_grid(original_image, denoised_images, ratio, image_shape=(32, 32, 3), grid_size=(4, 4), save_path='image_grid.png'):
    fig, axes = plt.subplots(*grid_size, figsize=(grid_size[1]*2, grid_size[0]*2))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    num_images = len(denoised_images)
    for i, ax in enumerate(axes.flat):
        if i == grid_size[0] * grid_size[1] - 1: 
            ax.imshow(original_image)
            rect = patches.Rectangle((0, 0), image_shape[1], image_shape[0], linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
        elif i < num_images:
            ax.imshow(denoised_images[i])
        ax.axis('off')
    plt.suptitle(f'Red box indicates original image - ratio t/T: {ratio:.2f}', fontsize=16)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def main(model_path, data_path, save_path):
    iterations=15
    num_steps=25
    sigma_min=0.002
    sigma_max=80
    rho=5
    batch_size=12
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)['ema'].to(device)
    model.eval()

    data = load_data(
        data_dir=data_path,
        batch_size=batch_size,
        image_size=32,
        class_cond=False,
    )

    
    images, _ = next(data)
    original_img = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    t_steps = get_discretization_steps(model=model, device=device, num_steps=num_steps, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho)
    for step in range(num_steps - 1):
        ratio = t_steps[step] / sigma_max
        if ratio > 0.005 and ratio < 0.15:
            collection = []
            for _ in range(iterations):
                denoised = edm_sampler(model, images.to(device), t_steps[step:])
                denoised = (denoised * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
                collection.append(denoised)
            collection = np.array(collection)
            print(collection.shape)
            collection = np.swapaxes(collection,0,1)
            print(f"{collection.shape} \n")
            for b, denoised_img in enumerate(collection):
                os.makedirs(f"{save_path}/{b}", exist_ok=True)
                create_image_grid(original_img[b], denoised_img, ratio=ratio, save_path=f"{save_path}/{b}/{ratio}.png")






if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Hyperparameters for sampling')
    parser.add_argument('--model_path', required=True, help='the path to model')
    parser.add_argument('--data_path', required=True, help='path to data')
    parser.add_argument('--save_path', required=True, help='path to save denoised images')
    args = parser.parse_args()
    main(model_path=args.model_path, data_path=args.data_path, save_path=args.save_path)