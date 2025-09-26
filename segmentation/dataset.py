import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from PIL import Image
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm
from os.path import splitext, isfile, join


def load_image(filename):
    """
    加载图像文件并转为PIL格式。

    Args:
        filename (str or Path): 图像文件路径。

    Return:
        PIL.Image.Image: 转换为 PIL 格式的图像。
    """
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    """
    获取指定掩码文件的唯一像素值。

    Args:
        idx (str): 图像/掩码 ID（不含扩展名）。
        mask_dir (Path): 掩码文件夹路径。
        mask_suffix (str): 掩码文件名后缀。

    Return:
        numpy.ndarray: 掩码中的唯一像素值。如果是多通道彩色掩码，返回唯一颜色的数组；如果是灰度掩码，返回唯一灰度值。

    """
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicSegmentationDataset(Dataset):
    """
    通用语义分割数据集.
    该数据集类会自动读取图像与对应的掩码文件，并进行预处理.

    Args:
        images_dir (str): 图像文件夹路径。
        mask_dir (str): 掩码文件夹路径。
        scale (float, optional): 图像缩放比例，必须在 (0, 1] 之间。默认 1.0。
        mask_suffix (str, optional): 掩码文件名的附加后缀。默认 ''。

    Attributes:
        ids (list[str]): 数据集中所有样本的 ID（不含扩展名）。
        mask_values (list): 掩码中的唯一像素值集合。
        images_dir (Path): 图像文件夹路径（Pathlib 格式）。
        mask_dir (Path): 掩码文件夹路径（Pathlib 格式）。
        scale (float): 缩放比例。
        mask_suffix (str): 掩码文件名附加后缀。

    Example:
        >>> dataset = BasicSegmentationDataset("data/images", "data/masks", scale=0.5)
        >>> sample = dataset[0]
        >>> image, mask = sample["image"], sample["mask"]
        >>> image.shape
        torch.Size([3, 256, 256])
        >>> mask.shape
        torch.Size([256, 256])
    """
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)

        assert 0 < scale <= 1, 'Scale must be between 0 and 1'              # 断言，保证图片缩放比例在0到1之间
        self.scale = scale
        self.mask_suffix = mask_suffix

        # 获取所有图像 id（不含扩展名）
        self.ids = [
            splitext(file)[0]                                               # 获取文件名（删除后缀）作为ID
            for file in os.listdir(images_dir)
            if isfile(join(images_dir, file)) and not file.startswith('.')
        ]
        if not self.ids:                                                    # 空文件检测
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        with Pool() as p:                                                   # 多线程并行处理，获取所有mask的唯一值
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir,  # 固定参数，只传ids，并遍历每个ids
                               mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, target_size=(256, 256), is_mask=False):
        pil_img = pil_img.resize(target_size, resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:                                                         # 处理mask
            mask = np.zeros(target_size, dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:                                           # 灰度mask直接比较像素值
                    mask[img == v] = i
                else:                                                       # 彩色mask需逐通道相等才算一类
                    mask[(img == v).all(-1)] = i
            return mask

        else:                                                               # 处理image
            if img.ndim == 2:                                               # 灰度图，维度+1
                img = img[np.newaxis, ...]
            else:                                                           # RGB图，把通道维度放在第一位
                img = img.transpose((2, 0, 1))
            if (img > 1).any():                                             # 归一化
                img = img / 255.0
            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        # 断言，保证每个 ID 只有一张图像和一张 mask
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'

        # 加载mask和image
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        # 断言，确保 mask 和图像原始大小一致。
        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, target_size=(256, 256), is_mask=False)
        mask = self.preprocess(self.mask_values, mask, target_size=(256, 256), is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


