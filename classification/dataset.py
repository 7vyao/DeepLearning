import os

import pandas as pd
import torch
from PIL.Image import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


def get_transforms(is_training=True, image_size=224):
    """
    获取图像的数据增强和预处理变换。

    Args:
        is_training (bool, optional): 是否为训练模式。训练模式会使用随机裁剪、翻转、旋转和颜色扰动。默认 True。
        image_size (int, optional): 输出图像的尺寸（宽和高）。默认 224。

    Returns:
        torchvision.transforms.Compose: 可直接用于 Dataset 的图像变换组合。
    """

    if is_training:
        return transforms.Compose([
            transforms.Resize((int(image_size * 1.14), int(image_size * 1.14))),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])



class ImagesDataset(Dataset):
    """
    自定义 PyTorch 数据集，用于从 CSV 文件加载图像及其标签。

    Args:
        csv_file (str): CSV 文件路径，包含数据集的 image_id 和标签列。
        image_dir (str): 图像存放目录。
        transform (callable, optional): 数据增强或预处理操作。默认 None，会使用类内 get_transforms 函数生成。
        indices (list or array-like, optional): 仅使用 CSV 文件中的指定行索引。默认 None。
        image_size (int, optional): 图像尺寸，用于默认 transform。默认 224。
        label_names (list of str, optional): 指定标签列名。默认 None，会自动获取 CSV 中除 image_id 外的列。
        multi_label (bool, optional): 是否为多标签分类。默认 False。
        default_label (int, optional): 如果没有有效标签，使用的默认标签。默认 0。
        skip_invalid (bool, optional): 是否跳过没有有效标签的样本。默认 False。
        is_training (bool, optional): 是否为训练模式，会影响默认 transform 的增强策略。默认 True。

    Returns:
        tuple:
            image (torch.Tensor): 经过 transform 后的图像张量。
            label_tensor (torch.Tensor): 标签张量，单标签为 long 类型，多标签为 float 类型。
            image_name (str): 对应图片文件名，便于调试或后处理。
    """

    def __init__(self, csv_file, image_dir, transform=None, indices=None, image_size=224, label_names=None,
                 multi_label=False, default_label=0, skip_invalid=False, is_training=True):

        self.df = pd.read_csv(csv_file)
        if indices is not None:
            self.df = self.df.iloc[indices].reset_index(drop=True)
        self.image_dir = image_dir
        self.multi_label = multi_label
        self.default_label = default_label
        self.skip_invalid = skip_invalid

        if label_names is None:                                         # 自动取除 image_id 外的所有列,获取标签
            self.label_names = [c for c in self.df.columns if c != 'image_id']
        else:                                                           # 使用传入的标签名
            self.label_names = label_names

        self.transform = transform or get_transforms(is_training=is_training, image_size=image_size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_name = self.df.iloc[idx]['image_id'] + '.jpg'             # 索引CSV文件中的第idx行的image_id列，拼接处完整的imagename
        image_path = os.path.join(self.image_dir, image_name)           # 获取图像路径

        try:                                                            # 用PIL打开图片并转换成RGB
            image = Image.open(image_path).convert('RGB')
        except Exception as e:                                          # 如果读取失败就加载一张全黑图像
            print(f"Error loading image {image_path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        label_values = row[self.label_names].values                     # 读取标签
        if self.multi_label:
            label_tensor = torch.tensor(label_values, dtype=torch.float)
        else:
            label_tensor = torch.tensor(int(label_values.argmax())
                                        if label_values.sum() > 0
                                        else self.default_label,
                                        dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label_tensor, image_name
