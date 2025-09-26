import torch
from torch.utils.data import DataLoader, random_split
import os


def load_model(model_class, model_path=None, device='cuda'):
    model = model_class().to(device)

    if model_path is not None and os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded model weights from {model_path}")

    else:
        print("No pretrained weights provided, using randomly initialized model")

    return model


def split_dataset(dataset, val_ratio=0.2, seed=42):
    n_val = int(len(dataset) * val_ratio)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        gengerator=torch.Generator().manual_seed(seed)
    )
    return train_set, val_set


def create_dataloaders(train_set, val_set, batch_size=8, num_workers=4):
    loader_args = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    return train_loader, val_loader