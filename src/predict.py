import os
import torch
import matplotlib.pyplot as plt
from model import UNet
from dataset import DAGMKaggleDataset
from torch.utils.data import DataLoader


# Function to visualize model predictions grouped by their respective classes
def visualize_by_classes(model, loader, device, model_path, imgs_per_class=7):
    # Set the directory for saving results in the project root
    save_dir = "../outputs/results/0.8627 40 epoch 512x512"
    os.makedirs(save_dir, exist_ok=True)

    # Set model to evaluation mode and load trained weights
    model.eval()

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded successfully. Saving results to: {save_dir}")
    else:
        print(f"Weight file not found: {model_path}")
        return

    # Dictionary to track how many images have been saved for each class
    class_counts = {}
    dataset = loader.dataset

    # Disable gradient calculation for faster inference and lower memory usage
    with torch.no_grad():
        for idx in range(len(dataset)):
            # Extract class name from the image file path
            img_path = dataset.samples[idx]['image']
            class_name = os.path.normpath(img_path).split(os.sep)[-3]
            if "Class4" in class_name:
                continue
            # Initialize class counter and skip if limit is reached
            if class_name not in class_counts: class_counts[class_name] = 0
            if class_counts[class_name] >= imgs_per_class: continue

            image, mask = dataset[idx]

            # Focus only on samples that contain actual defects (mask is not empty)
            if mask.max() > 0:
                # Forward pass: add batch dimension, move to device, and apply threshold
                output = model(image.unsqueeze(0).to(device))
                pred = (torch.sigmoid(output) > 0.5).float().cpu().numpy()[0][0]

                # Create a comparison plot with 3 subplots: Original, Ground Truth, and AI Prediction
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                fig.suptitle(f"{class_name} | Sample #{class_counts[class_name] + 1}")

                # Denormalize image for correct gray-scale display
                axes[0].imshow(image[0] * 0.5 + 0.5, cmap='gray')
                axes[0].set_title("Original")

                axes[1].imshow(mask[0], cmap='gray')
                axes[1].set_title("Label")

                axes[2].imshow(pred, cmap='gray')
                axes[2].set_title("AI Predict")

                for ax in axes: ax.axis('off')

                # Save the visualization and close the figure to free up RAM
                plt.savefig(f"{save_dir}/{class_name}_{class_counts[class_name] + 1}.png")
                plt.close(fig)

                class_counts[class_name] += 1
                print(f"Saved: {class_name} ({class_counts[class_name]}/{imgs_per_class})")


if __name__ == '__main__':
    # Initialize hardware acceleration and model architecture
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)

    # Initialize the test dataset
    test_ds = DAGMKaggleDataset("../data/DAGM_KaggleUpload", train=False)

    if len(test_ds) > 0:
        # Create a data loader and run the visualization process
        loader = DataLoader(test_ds, batch_size=1)
        visualize_by_classes(model, loader, device, '../outputs/checkpoints/0.8627 40 epoch 512x512.pth')
    else:
        print("No images found. Please check your data directory path.")