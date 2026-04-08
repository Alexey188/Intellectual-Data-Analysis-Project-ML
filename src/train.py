import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import amp
from utils import calc_confusion_matrix, plot_results
from model import UNet
from dataset import DAGMKaggleDataset

# Set device to GPU if available, otherwise use CPU
# Установка устройства: GPU, если доступно, иначе CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_weights(m):
    # Initialize convolutional layers with Kaiming normal and BatchNorm with constants
    # Инициализация сверточных слоев методом Kaiming normal, а BatchNorm — константами
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Apply sigmoid and flatten tensors for overlap calculation
        # Применение сигмоиды и выравнивание тензоров для расчета пересечения
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

        return 1.0 - dice


def calculate_dice(preds, targets):
    # Calculate Dice score after thresholding predictions at 0.5
    # Расчет коэффициента Дайса после применения порога 0.5 к предсказаниям
    preds = (torch.sigmoid(preds) > 0.5).float()
    intersection = (preds * targets).sum()
    dice = (2. * intersection + 1e-6) / (preds.sum() + targets.sum() + 1e-6)
    return dice.item()


def train_model(model, train_loader, val_loader, epochs=50):
    # Setup loss, optimizer, and automatic mixed precision scaler
    # Настройка функции потерь, оптимизатора и скалера для смешанной точности (AMP)
    criterion = DiceLoss(smooth=1.0)

    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_val_dice = 0.0
    scaler = amp.GradScaler('cuda')

    # Data structures to store training progress and the best confusion matrix
    # Структуры данных для хранения прогресса обучения и лучшей матрицы ошибок
    history = {'train_loss': [], 'val_loss': [], 'val_dice': []}
    best_conf_matrix = (0, 0, 0, 0)

    for epoch in range(epochs):
        model.train()
        t_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=True)

        # Training loop with mixed precision and gradient clipping
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            with amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            t_loss += loss.item()

        model.eval()
        v_loss, v_dice = 0, 0
        ep_tp, ep_fp, ep_fn, ep_tn = 0, 0, 0, 0

        # Evaluation loop: calculate validation loss and metrics
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                with amp.autocast('cuda'):
                    outputs = model(images)
                    loss_val = criterion(outputs, masks)
                v_loss += loss_val.item()
                v_dice += calculate_dice(outputs, masks)

                # Update cumulative confusion matrix for the current epoch
                tp, fp, fn, tn = calc_confusion_matrix(outputs, masks)
                ep_tp += tp;
                ep_fp += fp;
                ep_fn += fn;
                ep_tn += tn

        # Calculate average metrics for the epoch
        # Расчет средних значений метрик за эпоху
        avg_train_loss = t_loss / len(train_loader)
        avg_val_loss = v_loss / len(val_loader)
        avg_val_dice = v_dice / len(val_loader)


        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_dice'].append(avg_val_dice)

        print(
            f"Result: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Dice: {avg_val_dice:.4f}")

        # Update scheduler based on validation Dice score
        # Обновление планировщика на основе коэффициента Дайса на валидации
        scheduler.step(avg_val_dice)

        # Checkpointing: save model if validation Dice improved
        # Сохранение чекпоинта: запись модели, если коэффициент Дайса улучшился
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            best_conf_matrix = (ep_tp, ep_fp, ep_fn, ep_tn)
            save_path = '../outputs/checkpoints/test6_dice_only_512.pth'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"Saved Best Dice: {best_val_dice:.4f}")

        print("-" * 20)

    # Generate final training curves and confusion matrix plot
    plot_results(history, best_conf_matrix)
    return


if __name__ == '__main__':
    # Initialize datasets and data loaders
    ROOT_DIR = "../data/DAGM_KaggleUpload"
    train_dataset = DAGMKaggleDataset(ROOT_DIR, train=True)
    val_dataset = DAGMKaggleDataset(ROOT_DIR, train=False)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=6)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=6)

    # Create model, apply weights initialization, and start training
    model = UNet().to(device)
    model.apply(init_weights)

    train_model(model, train_loader, val_loader, epochs=50)