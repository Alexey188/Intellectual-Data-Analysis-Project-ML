import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def calc_confusion_matrix(preds, targets):
    """
    Calculates pixel-wise counts for the confusion matrix (TP, FP, FN, TN)
    Считает пиксели для матрицы ошибок (TP, FP, FN, TN)
    """
    # Threshold predictions at 0.5 to get a binary mask
    # Применение порога 0.5 к предсказаниям для получения бинарной маски
    preds = (torch.sigmoid(preds) > 0.5).float()

    # Sum up True Positives, False Positives, False Negatives, and True Negatives
    # Суммирование истинно положительных, ложноположительных, ложноотрицательных и истинно отрицательных пикселей
    tp = (preds * targets).sum().item()
    fp = (preds * (1 - targets)).sum().item()
    fn = ((1 - preds) * targets).sum().item()
    tn = ((1 - preds) * (1 - targets)).sum().item()

    return tp, fp, fn, tn


def plot_results(history, conf_matrix, save_dir='../outputs/results'):
    """
    Visualizes training history curves and the confusion matrix
    """
    tp, fp, fn, tn = conf_matrix

    plt.figure(figsize=(16, 5))

    # 1. Loss Curve: Plot training and validation loss over epochs

    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='blue', linewidth=2)
    plt.plot(history['val_loss'], label='Val Loss', color='red', linewidth=2)
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 2. Dice Score Curve: Visualize the primary segmentation metric progress
    # 2. График метрики Dice: Визуализация прогресса основной метрики сегментации
    plt.subplot(1, 3, 2)
    plt.plot(history['val_dice'], label='Val Dice', color='green', linewidth=2)
    plt.title('Dice Score Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 3. Confusion Matrix: Create a 2x2 matrix for pixel classification results
    # 3. Матрица ошибок: Создание матрицы 2x2 для результатов классификации пикселей
    plt.subplot(1, 3, 3)
    matrix = np.array([[tn, fp],
                       [fn, tp]])

    # Render a heatmap using Seaborn with labels for 'Background' and 'Defect'
    # Отрисовка тепловой карты с помощью Seaborn с метками 'Background' и 'Defect'
    sns.heatmap(matrix, annot=True, fmt='.0f', cmap='Blues',
                xticklabels=['Background', 'Defect'],
                yticklabels=['Background', 'Defect'])
    plt.title('Confusion Matrix (Pixels)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    plt.tight_layout()

    # Create the output directory if it doesn't exist and save the final figure
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'training_metrics.png')
    plt.savefig(save_path, dpi=300)
    print(f"Training curves and Confusion Matrix saved to '{save_path}'")
    plt.close()