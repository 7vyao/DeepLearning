import torch.nn.functional as F
import torch.optim.lr_scheduler
from matplotlib import pyplot as plt
from torch import optim
from tqdm import tqdm

from segmentation.utils import DiceLoss, FocalLoss


def train_epoch(model, train_loader, criterion, optimizer, device, num_classes, grad_clip_value=None):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    dices, ious, accs = [], [], []

    progress_bar = tqdm(train_loader, desc="Training")
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)

        loss.backward()

        if grad_clip_value is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)

        optimizer.step()

        total_loss += loss.item()

        preds = outputs.argmax(dim=1) if num_classes > 1 else (torch.sigmoid(outputs) > 0.5).long()
        dice, iou, acc = _compute_metrics(outputs, masks, num_classes)

        dices.append(dice)
        ious.append(iou)
        accs.append(acc)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(masks.cpu().numpy())

        progress_bar.set_postfix({"Loss": loss.item(), "Dice": dice, "IoU": iou})

    avg_loss = total_loss / len(train_loader)
    return avg_loss, np.mean(dices), np.mean(ious), np.mean(accs), np.array(all_preds), np.array(all_labels)




def validate_epoch(model, val_loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    dices, ious, accs = [], [], []

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Validating"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()

            preds = outputs.argmax(dim=1) if num_classes > 1 else (torch.sigmoid(outputs) > 0.5).long()
            dice, iou, acc = _compute_metrics(outputs, masks, num_classes)

            dices.append(dice)
            ious.append(iou)
            accs.append(acc)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(masks.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    return avg_loss, np.mean(dices), np.mean(ious), np.mean(accs), np.array(all_preds), np.array(all_labels)


import torch
import torch.nn as nn
import numpy as np


def train_model(model, train_loader, val_loader,
                num_epochs=50, lr=1e-4, weight_decay=0.01,
                loss_name="crossentropy", optimizer_name="adamw", scheduler_name="cosine",
                device='cuda', patience=10, min_delta=1e-4,
                grad_clip_value=None, display_interval=10, model_save_path='best_model.pth'):
    """
    分割任务的训练主函数（结构与分类版本统一）

    Args:
        model (torch.nn.Module): 分割模型
        train_loader (DataLoader): 训练数据
        val_loader (DataLoader): 验证数据
        num_epochs (int): 最大训练轮数
        lr (float): 学习率
        weight_decay (float): 权重衰减
        loss_name (str): 损失函数
        optimizer_name (str): 优化器
        scheduler_name (str): 学习率调度器
        device (str): 设备
        patience (int): 早停阈值
        min_delta (float): Dice 提升最小阈值
        grad_clip_value (float): 梯度裁剪
        display_interval (int): 指标平准打印间隔
        model_save_path (str): 最佳模型保存路径

    Returns:
        best_model_state (dict): 最优模型 state_dict
        metrics (dict): 训练过程的指标
    """

    criterion, optimizer, scheduler = _build_training_config(
        model.parameters(), loss_name=loss_name,
        optimizer_name=optimizer_name,
        scheduler_name=scheduler_name,
        lr=lr, weight_decay=weight_decay,
        num_epochs=num_epochs, patience=patience
    )

    train_losses, val_losses = [], []
    train_dices, val_dices = [], []
    train_ious, val_ious = [], []
    train_accs, val_accs = [], []

    best_val_dice = 0
    best_model_state = None
    patience_counter = 0

    epoch_group_metrics = {
        'train_loss': [], 'val_loss': [],
        'train_dice': [], 'val_dice': [],
        'train_iou': [], 'val_iou': [],
        'train_acc': [], 'val_acc': []
    }

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)

        train_loss, train_dice, train_iou, train_acc, _, _ = train_epoch(
            model, train_loader, criterion, optimizer, device,
            grad_clip_value=grad_clip_value
        )

        val_loss, val_dice, val_iou, val_acc, _, _ = validate_epoch(
            model, val_loader, criterion, device
        )

        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_dices.append(train_dice)
        val_dices.append(val_dice)
        train_ious.append(train_iou)
        val_ious.append(val_iou)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        epoch_group_metrics['train_loss'].append(train_loss)
        epoch_group_metrics['val_loss'].append(val_loss)
        epoch_group_metrics['train_dice'].append(train_dice)
        epoch_group_metrics['val_dice'].append(val_dice)
        epoch_group_metrics['train_iou'].append(train_iou)
        epoch_group_metrics['val_iou'].append(val_iou)
        epoch_group_metrics['train_acc'].append(train_acc)
        epoch_group_metrics['val_acc'].append(val_acc)

        # --- 打印 ---
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Train Dice: {train_dice:.4f}, Val Dice: {val_dice:.4f}")
        print(f"Train IoU: {train_iou:.4f}, Val IoU: {val_iou:.4f}")
        print(f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        # --- 平均指标打印 ---
        if (epoch + 1) % display_interval == 0:
            print("\n" + "=" * 60)
            print(f"AVERAGE METRICS FOR EPOCHS {epoch - display_interval + 2}-{epoch + 1}")
            print("=" * 60)
            print(f"Avg Train Loss: {np.mean(epoch_group_metrics['train_loss']):.4f} "
                  f"| Avg Val Loss: {np.mean(epoch_group_metrics['val_loss']):.4f}")
            print(f"Avg Train Dice: {np.mean(epoch_group_metrics['train_dice']):.4f} "
                  f"| Avg Val Dice: {np.mean(epoch_group_metrics['val_dice']):.4f}")
            print(f"Avg Train IoU: {np.mean(epoch_group_metrics['train_iou']):.4f} "
                  f"| Avg Val IoU: {np.mean(epoch_group_metrics['val_iou']):.4f}")
            print(f"Avg Train Acc: {np.mean(epoch_group_metrics['train_acc']):.4f} "
                  f"| Avg Val Acc: {np.mean(epoch_group_metrics['val_acc']):.4f}")
            print("=" * 60)

            # 清空分组指标
            for key in epoch_group_metrics:
                epoch_group_metrics[key] = []

        if val_dice > best_val_dice + min_delta:
            best_val_dice = val_dice
            best_model_state = model.state_dict().copy()
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_dice': best_val_dice,
                'val_iou': val_iou,
                'val_acc': val_acc,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_dices': train_dices,
                'val_dices': val_dices,
                'train_ious': train_ious,
                'val_ious': val_ious,
                'train_accs': train_accs,
                'val_accs': val_accs
            }, model_save_path)
            print(f"🎉 New best model saved with Dice: {best_val_dice:.4f}")
        else:
            patience_counter += 1
            print(f"⏳ No improvement for {patience_counter} epochs")

        if patience_counter >= patience:
            print(f"\n🛑 Early stopping triggered after {patience} epochs without improvement")
            break

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED!")
    print("=" * 60)
    print(f"Best Validation Dice: {best_val_dice:.4f}")
    print(f"Total Epochs: {epoch + 1}")

    _plot_segmentation_results(train_losses, val_losses,  # 这里要改成画 Dice/IoU/Acc 曲线的函数
                               train_dices, val_dices,
                               train_ious, val_ious,
                               train_accs, val_accs)

    return best_model_state, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_dices': train_dices,
        'val_dices': val_dices,
        'train_ious': train_ious,
        'val_ious': val_ious,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_val_dice': best_val_dice
    }


def _build_training_config(model_params, loss_name="crossentropy", optimizer_name="adamw",
                           scheduler_name=None, lr=1e-5, weight_decay=1e-4, num_epochs=50, patience=5):
    """
    构建训练配置：损失函数、优化器和学习率调度器。

    Args:
        model_params: 模型参数，供优化器使用。
        loss_name: 损失函数类型
        optimizer_name: 优化器类型
        scheduler_name: 学习率调度器类型
        lr (float, optional): 学习率，默认 1e-5。
        weight_decay (float, optional): 权重衰减系数，默认 1e-4。
        num_epochs (int, optional): 总训练轮数，用于部分调度器，默认 50。
        patience (int, optional): 验证集停滞时降低学习率的等待轮数，仅对 ReduceLROnPlateau 有效，默认 5。

    Returns:
        tuple:
            loss_fn (torch.nn.Module): 选定的损失函数实例。
            optimizer (torch.optim.Optimizer): 选定的优化器实例。
            scheduler (torch.optim.lr_scheduler._LRScheduler or None): 选定的学习率调度器实例，若 scheduler_name 为 None 则返回 None。
    """

    # 损失函数
    loss_map = {
        # 🎯 多分类分割标准
        "crossentropy": nn.CrossEntropyLoss(),
        # ✅ 二分类 / 多标签
        "bce": nn.BCEWithLogitsLoss(),
        # 🧩 Dice 损失（需自定义实现）
        "dice": DiceLoss(),
        # 🔗 Dice + CE
        "dice_ce": lambda: DiceLoss() + nn.CrossEntropyLoss(),
        # 🔗 Dice + BCE
        "dice_bce": lambda: DiceLoss() + nn.BCEWithLogitsLoss(),
        # 🎯 类别不平衡时常用
        "focal": FocalLoss(),
    }
    loss_fn = loss_map[loss_name]

    # 优化器
    opt_map = {
        # ⚙️ Transformer / 分割常用
        "adamw": optim.AdamW(model_params, lr=lr, weight_decay=weight_decay),
        # ⚡ 快速收敛
        "adam": optim.Adam(model_params, lr=lr, weight_decay=weight_decay),
        # 🏃 稳定收敛，大数据集常用
        "sgd": optim.SGD(model_params, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True),
        # 🌊 UNet 原版常用
        "rmsprop": optim.RMSprop(model_params, lr=lr, alpha=0.9, weight_decay=weight_decay),
    }
    optimizer = opt_map[optimizer_name]

    # 学习率调度器
    sched_map = {
        # 🌊 余弦退火
        "cosine": optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs),
        # 🔥🌊 余弦 + 热重启
        "cosine_warmup": optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2),
        # 🛑 验证集停滞时降低 lr
        "reduce_on_plateau": optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=patience),
        # 📉 PolyLR（分割论文常见）
        "poly": torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: (1 - epoch / num_epochs) ** 0.9)
    }
    scheduler = sched_map[scheduler_name] if scheduler_name else None

    return loss_fn, optimizer, scheduler


def _plot_segmentation_results(train_losses, val_losses,
                               train_dices, val_dices,
                               train_ious, val_ious,
                               train_accs, val_accs,
                               save_path='segmentation_results.png'):
    """
    绘制分割任务训练过程的曲线 (Loss, Dice, IoU, Acc)

    Args:
        train_losses (list[float]): 训练 Loss
        val_losses (list[float]): 验证 Loss
        train_dices (list[float]): 训练 Dice
        val_dices (list[float]): 验证 Dice
        train_ious (list[float]): 训练 IoU
        val_ious (list[float]): 验证 IoU
        train_accs (list[float]): 训练 Acc
        val_accs (list[float]): 验证 Acc
        save_path (str, optional): 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # ---- Loss ----
    axes[0, 0].plot(train_losses, label='Train Loss', color='blue')
    axes[0, 0].plot(val_losses, label='Val Loss', color='orange')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()

    # ---- Dice ----
    axes[0, 1].plot(train_dices, label='Train Dice', color='blue')
    axes[0, 1].plot(val_dices, label='Val Dice', color='orange')
    axes[0, 1].set_title('Dice Coefficient')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Dice')
    axes[0, 1].legend()

    # ---- IoU ----
    axes[1, 0].plot(train_ious, label='Train IoU', color='blue')
    axes[1, 0].plot(val_ious, label='Val IoU', color='orange')
    axes[1, 0].set_title('Intersection over Union (IoU)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('IoU')
    axes[1, 0].legend()

    # ---- Accuracy ----
    axes[1, 1].plot(train_accs, label='Train Accuracy', color='blue')
    axes[1, 1].plot(val_accs, label='Val Accuracy', color='orange')
    axes[1, 1].set_title('Pixel Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def _compute_metrics(pred, target, num_classes):
    """
    计算分割任务的 Dice, IoU, Acc
    Args:
        pred (torch.Tensor): 模型输出，形状 [B, C, H, W] (logits)
        target (torch.Tensor): 真实标签，形状 [B, H, W] (int)
        num_classes (int): 类别数
    Returns:
        dice (float), iou (float), acc (float)
    """
    if num_classes > 1:  # 多分类
        preds = pred.argmax(dim=1)  # [B, H, W]
        pred_onehot = F.one_hot(preds, num_classes).permute(0, 3, 1, 2)  # [B, C, H, W]
        target_onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).to(pred.device)
    else:  # 二分类
        preds = (torch.sigmoid(pred) > 0.5).long()  # [B, 1, H, W]
        pred_onehot = preds
        target_onehot = target.unsqueeze(1).long()

    # 转 float 方便计算
    pred_onehot = pred_onehot.float()
    target_onehot = target_onehot.float()

    # Dice & IoU
    intersection = (pred_onehot * target_onehot).sum(dim=(0, 2, 3))  # 按类别算
    union = pred_onehot.sum(dim=(0, 2, 3)) + target_onehot.sum(dim=(0, 2, 3))

    dice = ((2 * intersection + 1e-6) / (union + 1e-6)).mean().item()
    iou = ((intersection + 1e-6) / (union - intersection + 1e-6)).mean().item()

    # Accuracy
    acc = (preds == target).float().mean().item()

    return dice, iou, acc