import sys

sys.path.insert(0, "/p/project/obdifflearn/thoeppe/edm")
import os
from reconstruction.image_datasets import load_data
from cifar10_models.resnet import resnet18
from argparse import ArgumentParser
from torchmetrics import Accuracy
import torch as th
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from schduler import WarmupCosineLR
from torch.nn import DataParallel


def log_gpu_usage():
    os.system("nvidia-smi")


def model_validation(model, val_data, device, val_iterations):

    model.eval()
    accuracy = Accuracy(task="multiclass", num_classes=10).to(device)
    with th.no_grad():
        for _ in range(val_iterations):
            images, labels = next(val_data)
            images = images.to(device)
            labels = labels["y"].to(device)
            logits, _, _, _, _, _ = model(images)
            accuracy.update(logits, labels)

    val_accuracy = accuracy.compute().item()
    return val_accuracy


def training_loop(
    model,
    train_data,
    iterations,
    device,
    optimizer,
    scheduler,
    model_save_path,
    val_data=None,
    val_iter=50,
    save_iter=100,
):
    criterion = th.nn.CrossEntropyLoss()
    opt_str = model_save_path.split("/")[-1]

    for iter in range(iterations):
        images, labels = next(train_data)
        images = images.to(device=device)
        labels = labels["y"].to(device=device)

        logits, _, _, _, _, _ = model(images)

        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the loss every 10 iterations
        if iter % 10 == 0:
            print(f"\nModel {opt_str}")
            print(f"Iteration {iter}/{iterations}, Loss: {loss.item()}")
            # log_gpu_usage()

            if iter % val_iter == 0 and val_data is not None:
                val_acc = model_validation(model, val_data, device, val_iterations=20)
                model.train()
                print(f"Iteration {iter}/{iterations}, Val_acc: {val_acc} \n")

            if iter % save_iter == 0:
                model_iter_save_path = os.path.join(
                    model_save_path, f"model_{iter}.pth"
                )
                th.save(model.state_dict(), model_iter_save_path)
                print(f"Model saved to {model_iter_save_path}")
        if scheduler is not None:
            scheduler.step()


def main(
    train_data_path,
    batch_size,
    iterations,
    model_save_path,
    val_data_path=None,
    val_iter=50,
    save_iter=100,
    opt="sgd",
    scheduler_name="step",
):

    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    model = resnet18(pretrained=False)
    if th.cuda.device_count() > 1:
        print(f"Using {th.cuda.device_count()} GPUs")
        model = DataParallel(model)
    model.to(device)

    if opt == "sgd":
        optimizer = SGD(
            model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-2, nesterov=True
        )
        step_size = iterations // 3
        if scheduler_name == "step":
            scheduler = StepLR(optimizer, step_size=step_size, gamma=0.1)
        elif scheduler_name == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=iterations,
                eta_min=1e-8,
            )
        elif scheduler_name == "warmup":
            print("warmup")
            scheduler = WarmupCosineLR(
                optimizer, warmup_epochs=iterations * 0.3, max_epochs=iterations
            )
    elif opt == "adam":
        optimizer = Adam(model.parameters(), lr=1e-3)
        scheduler = None
    model_save_path = os.path.join(model_save_path, opt)
    if scheduler is not None:
        model_save_path = os.path.join(model_save_path, scheduler_name)
    os.makedirs(model_save_path, exist_ok=True)

    train_data = load_data(
        data_dir=train_data_path,
        batch_size=batch_size,
        image_size=32,
        class_cond=True,
        random_crop=True,
        random_flip=True,
    )
    if val_data_path is not None:
        val_data = load_data(
            data_dir=val_data_path,
            batch_size=batch_size,
            image_size=32,
            class_cond=True,
            random_crop=False,
            random_flip=False,
        )
    else:
        val_data = None

    training_loop(
        model,
        train_data,
        iterations,
        device,
        optimizer,
        scheduler,
        model_save_path,
        val_data,
        val_iter,
        save_iter,
    )


if __name__ == "__main__":

    parser = ArgumentParser()

    # Add arguments
    parser.add_argument("--opt", type=str, default="sgd", help="Optimizer to use")
    parser.add_argument(
        "--scheduler_name", type=str, default="step", help="schedueler to use"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20001,
        help="Number of iterations to run the training loop",
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Training and validation batch size"
    )
    parser.add_argument(
        "--val_iter", type=int, default=50, help="Frequency of evaluation"
    )
    parser.add_argument(
        "--save_iter", type=int, default=100, help="Frequency of saving"
    )
    parser.add_argument("--train_data_path", type=str, help="Path to training data")
    parser.add_argument("--val_data_path", type=str, help="Path to validation data")
    parser.add_argument("--model_save_path", type=str, help="Path to save model")

    # Parse arguments
    args = parser.parse_args()

    main(**vars(args))
