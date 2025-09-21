import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from classification.dataset import ImagesDataset


def create_data_splits(csv_file, train_split=0.8, random_state=42):
    """
    将 CSV 文件中的数据按比例划分为训练集和验证集。

    Args:
        csv_file (str): CSV 文件路径，包含数据集的 image_id 和标签列。
        train_split (float, optional): 训练集占比，默认 0.8。
        random_state (int, optional): 随机种子，保证划分可复现，默认 42。

    Returns:
        tuple:
            train_indices (list of int): 训练集样本索引。
            val_indices (list of int): 验证集样本索引。
    """

    df = pd.read_csv(csv_file)

    indices = list(range(len(df)))

    train_indices, val_indices = train_test_split(
        indices,
        train_size=train_split,
        random_state=random_state,
        shuffle=True,
        stratify=None,
    )
    print(f"\n总数据量: {len(df)}")
    print(f"训练集: {len(train_indices)} ({len(train_indices) / len(df) * 100:.1f}%)")
    print(f"验证集: {len(val_indices)} ({len(val_indices) / len(df) * 100:.1f}%)")

    return train_indices, val_indices


def create_data_loaders(csv_file, image_dir, train_indices, val_indices,
                        batch_size=32, num_workers=4, image_size=224, multi_label=False):
    """
    根据索引创建训练集和验证集的 DataLoader。

    Args:
        csv_file (str): CSV 文件路径，包含 image_id 和标签。
        image_dir (str): 图像所在目录。
        train_indices (list of int): 训练集样本索引。
        val_indices (list of int): 验证集样本索引。
        batch_size (int, optional): 批次大小，默认 32。
        num_workers (int, optional): DataLoader 使用的子进程数，默认 4。
        image_size (int, optional): 图像尺寸，默认 224。
        multi_label (bool, optional): 是否为多标签任务，默认 False。

    Returns:
        tuple:
            train_loader (DataLoader): 训练集 DataLoader。
            val_loader (DataLoader): 验证集 DataLoader。
    """

    train_dataset = ImagesDataset(
        csv_file=csv_file,
        image_dir=image_dir,
        indices=train_indices,
        image_size=image_size,
        is_training=True,
        multi_label=multi_label
    )

    val_dataset = ImagesDataset(
        csv_file=csv_file,
        image_dir=image_dir,
        indices=val_indices,
        image_size=image_size,
        is_training=False,
        multi_label=multi_label
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader