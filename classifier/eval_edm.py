import sys

sys.path.insert(0, "/p/project/obdifflearn/thoeppe/edm")
import torch as th
import os
from reconstruction.image_datasets import load_data
from cifar10_models.resnet import resnet18
from argparse import ArgumentParser
from torchmetrics import Accuracy
from torch.nn import DataParallel


def main(model_path, data_path, batch_size, start_iteration=7500, max_iteration=8600):
    data_loader = load_data(
        data_dir=data_path,
        batch_size=batch_size,
        image_size=32,
        class_cond=True,
        random_crop=False,
        random_flip=False,
        deterministic=True,
    )
    model_name = model_path.split("/")[-1]
    for iter in range(start_iteration, max_iteration + 1, 100):
        model = resnet18(pretrained=False)
        model = DataParallel(model)
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        model_iter_path = os.path.join(model_path, f"model_{iter}.pth")
        state_dict = th.load(model_iter_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        accuracy = Accuracy(task="multiclass", num_classes=10).to(device)

        num_files = 0
        with th.no_grad():
            while num_files < 10000:
                num_files += batch_size
                images, labels = next(data_loader)
                images = images.to(device)
                labels = labels["y"].to(device)
                logits, _, _, _, _, _ = model(images)
                accuracy.update(logits, labels)

        val_accuracy = accuracy.compute().item()
        print(f"Val_acc {model_name}_{iter}: {val_accuracy} \n")


if __name__ == "__main__":

    parser = ArgumentParser()

    # Add arguments
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Training and validation batch size"
    )
    parser.add_argument("--start_iteration", type=int, default=7500, help="start model")
    parser.add_argument("--max_iteration", type=int, default=8600, help="last model")
    parser.add_argument("--data_path", type=str, help="Path to validation data")
    parser.add_argument("--model_path", type=str, help="Path to save model")

    # Parse arguments
    args = parser.parse_args()

    main(**vars(args))
