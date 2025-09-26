import copy

import numpy as np
import torch.nn.utils
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, classification_report
from torch import nn, optim
from tqdm import tqdm


def train_epoch(model, train_loader, criterion, optimizer, device, grad_clip_value=None):
    """
    对模型执行一个训练轮次（epoch）。

    Args:
        model (torch.nn.Module): 要训练的模型。
        train_loader (torch.utils.data.DataLoader): 包含训练数据的 DataLoader，每个批次返回 (images, labels)。
        criterion (torch.nn.Module): 损失函数。
        optimizer (torch.optim.Optimizer): 优化器。
        device (torch.device or str): 模型和数据所在的设备。
        grad_clip_value (float, optional): 梯度剪裁阈值，超过该值的梯度将被裁剪。默认 None 表示不裁剪。

    Returns:
        avg_loss (float): 该轮训练的平均损失。
        accuracy (float): 该轮训练的整体准确率（%）。
        f1 (float): 该轮训练的宏平均 F1 分数。
        all_preds (np.ndarray): 本轮训练所有样本的预测标签。
        all_labels (np.ndarray): 本轮训练所有样本的真实标签。
    """

    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(train_loader, desc='Training')
    for batch_idx, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()                           # 清空上次迭代的梯度
        outputs = model(images)                         # 前向传播，获得预测值
        loss = criterion(outputs, labels)               # 计算损失

        loss.backward()                                 # 反向传播

        if grad_clip_value is not None:                 # 梯度剪裁
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)

        optimizer.step()                                # 更新模型参数

        total_loss += loss.item()                       # 计算总损失

        _, predicted = torch.max(outputs.data, 1)       # 获取预测类别
        total += labels.size(0)
        correct += (predicted == labels).sum().item()   # 统计预测正确数量

        all_preds.extend(predicted.cpu().numpy())       # 收集单批次所有的预测类别
        all_labels.extend(labels.cpu().numpy())         # 收集单批次所有的真实类别

        progress_bar.set_postfix({'Loss': loss.item(),
                                  'Acc': 100. * correct / total})

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    f1 = f1_score(all_labels, all_preds,
                  average='macro', zero_division=0)

    return avg_loss, accuracy, f1, np.array(all_preds), np.array(all_labels)


def validate_epoch(model, val_loader, criterion, device):
    """
    对模型执行一个验证轮次（epoch），计算损失、准确率和 F1 分数。

    Args:
        model (torch.nn.Module): 要验证的模型。
        val_loader (torch.utils.data.DataLoader): 验证数据的 DataLoader，每个批次返回 (images, labels)。
        criterion (torch.nn.Module): 损失函数。
        device (torch.device or str): 模型和数据所在的设备。

    Returns:
        avg_loss (float): 该轮验证的平均损失。
        accuracy (float): 该轮验证的整体准确率（%）。
        f1 (float): 该轮验证的宏平均 F1 分数。
        all_preds (np.ndarray): 本轮验证所有样本的预测标签。
        all_labels (np.ndarray): 本轮验证所有样本的真实标签。
    """

    model.eval()                                        # 评估模式，固定模型参数，禁止反向传播
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():                               # without梯度信息
        for images, labels in tqdm(val_loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(val_loader)             # 平均损失：总损失 / batch数量
    accuracy = 100. * correct / total                   # 准确率：预测正确样本数 / 总样本数 * 100%
    f1 = f1_score(all_labels, all_preds,                # 使用所有预测类别和真实类别，调用sklearn 的函数计算 F1
                  average='macro', zero_division=0)

    return avg_loss, accuracy, f1, np.array(all_preds), np.array(all_labels)


class Trainer:
    """
    通用模型训练器，封装训练、验证、指标记录、早停、最佳模型保存和可视化功能。

    Args:
        Args:
        model (torch.nn.Module): 需要训练的模型。
        train_loader (DataLoader): 训练数据的 DataLoader。
        val_loader (DataLoader): 验证数据的 DataLoader。
        num_epochs (int, optional): 最大训练轮数，默认 50。
        lr (float, optional): 学习率，默认 1e-4。
        weight_decay (float, optional): 权重衰减系数，默认 0.01。
        loss_name (str, optional): 损失函数名称，默认 "crossentropy"。
        optimizer_name (str, optional): 优化器名称，默认 "adamw"。
        scheduler_name (str, optional): 学习率调度器名称，默认 "cosine"。
        device (str or torch.device, optional): 训练设备，默认 "cuda"。
        patience (int, optional): 早停轮数阈值，默认 10。
        min_delta (float, optional): 精度提升阈值，用于早停判定，默认 1e-4。
        grad_clip_value (float, optional): 梯度裁剪阈值，None 表示不裁剪，默认 None。
        display_interval (int, optional): 打印平均指标的间隔轮数，默认 10。
        model_save_path (str, optional): 保存最佳模型的路径，默认 'best_model'。
        class_names (list[str], optional): 分类任务的类别名称列表，默认 None。

    Attributes:
        model (torch.nn.Module): 需要训练的模型。
        train_loader (DataLoader): 训练数据的 DataLoader。
        val_loader (DataLoader): 验证数据的 DataLoader。
        num_epochs (int): 最大训练轮数。
        device (torch.device): 模型训练设备。
        patience (int): 早停轮数阈值。
        min_delta (float): 验证指标提升最小阈值，用于早停判定。
        grad_clip_value (float or None): 梯度裁剪阈值，None 表示不裁剪。
        display_interval (int): 打印平均指标的间隔轮数。
        model_save_path (str): 保存最佳模型的路径。
        class_names (list[str] or None): 分类任务的类别名称列表。
        criterion (torch.nn.Module): 损失函数。
        optimizer (torch.optim.Optimizer): 优化器。
        scheduler (torch.optim.lr_scheduler._LRScheduler): 学习率调度器。
        history (dict): 训练过程中保存的指标历史，包括损失、准确率和 F1 分数。
        best_model_state (dict or None): 验证集上性能最优的模型权重。

    Example:
        Example:
        >>>
        >>> train_dataset = MyDataset(train=True)
        >>> val_dataset = MyDataset(train=False)
        >>> train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        >>> val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        >>>
        >>> model = MyModel(num_classes=5)
        >>> trainer = Trainer(
        ...     model=model,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     num_epochs=50,
        ...     lr=1e-4,
        ...     patience=10,
        ...     class_names=['类A', '类B', '类C', '类D', '类E']
        ... )
        >>> best_model_state, history = trainer.train()
        >>> history['train_losses'][-1]
        0.1234
        >>> history['val_accuracies'][-1]
        92.56
    """

    def __init__(self, model, train_loader, val_loader,
                 num_epochs=50, lr=1e-4, weight_decay=0.01,
                 loss_name="crossentropy", optimizer_name="adamw", scheduler_name="cosine",
                 device='cuda', patience=10, min_delta=1e-4,
                 grad_clip_value=None, display_interval=10, model_save_path='best_model', class_names=None):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.patience = patience
        self.min_delta = min_delta
        self.grad_clip_value = grad_clip_value
        self.display_interval = display_interval
        self.model_save_path = model_save_path
        self.class_names = class_names

        self.criterion, self.optimizer, self.scheduler = _build_training_config(
            model.parameters(), loss_name=loss_name,
            optimizer_name=optimizer_name, scheduler_name=scheduler_name,
            lr=lr, weight_decay=weight_decay, num_epochs=num_epochs, patience=patience
        )

        self.history = {
            'train_losses': [], 'val_losses': [],
            'train_accuracies': [], 'val_accuracies': [],
            'train_f1s': [], 'val_f1s': []
        }
        self.best_val_acc = 0
        self.best_model_state = None

    def _plot_training_results(self, save_path='training_results.png'):

        train_losses, val_losses = self.history['train_losses'], self.history['val_losses']
        train_accuracies, val_accuracies = self.history['train_accuracies'], self.history['val_accuracies']
        train_f1s, val_f1s = self.history['train_f1s'], self.history['val_f1s']

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 损失曲线
        axes[0, 0].plot(train_losses, label='Train Loss', color='blue')
        axes[0, 0].plot(val_losses, label='Val Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # 准确率曲线
        axes[0, 1].plot(train_accuracies, label='Train Accuracy', color='blue')
        axes[0, 1].plot(val_accuracies, label='Val Accuracy', color='red')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # F1分数曲线
        axes[1, 0].plot(train_f1s, label='Train Macro F1', color='blue')
        axes[1, 0].plot(val_f1s, label='Val Macro F1', color='red')
        axes[1, 0].set_title('Training and Validation Macro F1')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Macro F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # 最后十个epoch的详细指标
        recent_epochs = min(10, len(train_losses))
        epochs_range = range(len(train_losses) - recent_epochs, len(train_losses))

        axes[1, 1].plot(epochs_range, train_accuracies[-recent_epochs:], 'o-', label='Train Acc', color='blue')
        axes[1, 1].plot(epochs_range, val_accuracies[-recent_epochs:], 'o-', label='Val Acc', color='red')
        axes[1, 1].set_title('Last 10 Epochs - Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def train(self):

        model = self.model.to(self.device)
        criterion, optimizer, scheduler = self.criterion, self.optimizer, self.scheduler

        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []
        train_f1s, val_f1s = [], []

        best_val_acc = 0
        best_model_state = None
        patience_counter = 0

        epoch_group_metrics = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_f1': [], 'val_f1': []
        }

        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print("-" * 50)

            train_loss, train_acc, train_f1, train_preds, train_labels = train_epoch(
                model, self.train_loader, criterion, optimizer, self.device, self.grad_clip_value
            )

            val_loss, val_acc, val_f1, val_preds, val_labels = validate_epoch(
                model, self.val_loader, criterion, self.device
            )

            scheduler.step()  # 更新学习率

            train_losses.append(train_loss)  # 保存指标结果
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            train_f1s.append(train_f1)
            val_f1s.append(val_f1)

            epoch_group_metrics['train_loss'].append(train_loss)
            epoch_group_metrics['val_loss'].append(val_loss)
            epoch_group_metrics['train_acc'].append(train_acc)
            epoch_group_metrics['val_acc'].append(val_acc)
            epoch_group_metrics['train_f1'].append(train_f1)
            epoch_group_metrics['val_f1'].append(val_f1)

            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Train Accuracy: {train_acc:.2f}%, Val Accuracy: {val_acc:.2f}%")
            print(f"Train Macro F1: {train_f1:.4f}, Val Macro F1: {val_f1:.4f}")
            print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

            if (epoch + 1) % self.display_interval == 0:  # 每display_interval个epoch打印一次平准指标
                print("\n" + "=" * 60)
                print(f"AVERAGE METRICS FOR EPOCHS {epoch - self.display_interval + 2}-{epoch + 1}")
                print("=" * 60)
                avg_train_loss = np.mean(epoch_group_metrics['train_loss'])
                avg_val_loss = np.mean(epoch_group_metrics['val_loss'])
                avg_train_acc = np.mean(epoch_group_metrics['train_acc'])
                avg_val_acc = np.mean(epoch_group_metrics['val_acc'])
                avg_train_f1 = np.mean(epoch_group_metrics['train_f1'])
                avg_val_f1 = np.mean(epoch_group_metrics['val_f1'])

                print(f"Average Train Loss: {avg_train_loss:.4f} | Average Val Loss: {avg_val_loss:.4f}")
                print(f"Average Train Accuracy: {avg_train_acc:.2f}% | Average Val Accuracy: {avg_val_acc:.2f}%")
                print(f"Average Train Macro F1: {avg_train_f1:.4f} | Average Val Macro F1: {avg_val_f1:.4f}")
                print("=" * 60)

                for key in epoch_group_metrics:  # 清空累积指标
                    epoch_group_metrics[key] = []

            if val_acc > best_val_acc + self.min_delta:  # 保存最好的模型权重
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_accuracy': best_val_acc,
                    'val_f1': val_f1,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'train_accuracies': train_accuracies,
                    'val_accuracies': val_accuracies,
                    'train_f1s': train_f1s,
                    'val_f1s': val_f1s
                }, self.model_save_path)
                print(f"🎉 New best model saved with accuracy: {best_val_acc:.2f}%")
            else:
                patience_counter += 1
                print(f"⏳ No improvement for {patience_counter} epochs")

            if patience_counter >= self.patience:  # 早停机制
                print(f"\n🛑 Early stopping triggered after {self.patience} epochs without improvement")
                break

        print("\n" + "=" * 60)
        print("TRAINING COMPLETED!")
        print("=" * 60)
        print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
        print(f"Total Epochs: {epoch + 1}")

        # 更新历史指标
        self.history['train_losses'] = train_losses
        self.history['val_losses'] = val_losses
        self.history['train_accuracies'] = train_accuracies
        self.history['val_accuracies'] = val_accuracies
        self.history['train_f1s'] = train_f1s
        self.history['val_f1s'] = val_f1s

        self._plot_training_results()

        if best_model_state is not None:  # 使用最佳模型进行最终验证
            model.load_state_dict(best_model_state)
            model.eval()

            final_val_loss, final_val_acc, final_val_f1, final_val_preds, final_val_labels = validate_epoch(
                model, self.val_loader, criterion, self.device
            )

            print("\nFinal Classification Report:")
            print("=" * 50)
            print(classification_report(final_val_labels, final_val_preds,
                                        target_names=self.class_names, digits=4))

        return best_model_state, {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'train_f1s': train_f1s,
            'val_f1s': val_f1s,
            'best_val_accuracy': best_val_acc
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
        # 🎯 标配多分类损失
        "crossentropy": nn.CrossEntropyLoss(),
        # 📉 负对数似然（log_softmax 搭配）
        "nll": nn.NLLLoss(),
        # 🧊 标签平滑，防过拟合
        "label_smoothing": nn.CrossEntropyLoss(label_smoothing=0.1),
        # ✅ 二分类稳定首选
        "bce": nn.BCEWithLogitsLoss(),
    }
    loss_fn = loss_map[loss_name]

    # 优化器
    opt_map = {
        # 🏃‍♂️ 基础版，慢但稳
        "sgd": optim.SGD(model_params, lr=lr, momentum=0.9, weight_decay=weight_decay),
        # 🏹 SGD加速版
        "sgd_nesterov": optim.SGD(model_params, lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay),
        # ⚡ 常见自适应优化器
        "adam": optim.Adam(model_params, lr=lr, weight_decay=weight_decay),
        # ⚙️ Transformer & SOTA模型常用
        "adamw": optim.AdamW(model_params, lr=lr, weight_decay=weight_decay),
        # 🌊 RNN / LSTM 常用
        "rmsprop": optim.RMSprop(model_params, lr=lr, alpha=0.99, weight_decay=weight_decay),
        # 📚 稀疏数据友好
        "adagrad": optim.Adagrad(model_params, lr=lr, weight_decay=weight_decay),
        # # 🧩 Adam变体，适合大模型
        "adamax": optim.Adamax(model_params, lr=lr, weight_decay=weight_decay),
        #   # 🚀 Adam + Nesterov 动量
        "nadam": optim.NAdam(model_params, lr=lr, weight_decay=weight_decay),
    }
    optimizer = opt_map[optimizer_name]

    # 学习率调度器
    sched_map = {
        # ⏳ 每隔 step_size 降 lr
        "steplr": optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1),
        # 🪜 在指定 epoch 点衰减 lr
        "multistep": optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1),
        # 📉 指数衰减
        "exp": optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95),
        #  🌊 余弦退火，常用于分类
        "cosine": optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs),
        # 🔥🌊 余弦 + 热重启
        "cosine_warmup": optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2),
        # 🛑 验证集停滞时自动降低 lr
        "reduce_on_plateau": optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=patience),
    }
    scheduler = sched_map[scheduler_name] if scheduler_name else None

    return loss_fn, optimizer, scheduler





