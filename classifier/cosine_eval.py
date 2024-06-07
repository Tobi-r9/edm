import torch as th
import numpy as np


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
            for act in [act_5, act_4, act_3, act_2, act_1, act_0]:
                activations[f"act_{i}"].append(act.detach().cpu().numpy())

    for i in range(6):
        act = np.array(activations[f"act_{i}"])
        # TODO: Save stuff
