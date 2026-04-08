import os
import glob
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


# Custom dataset class for the DAGM Kaggle dataset

class DAGMKaggleDataset(Dataset):
    def __init__(self, root_dir, train=True, img_size=512):
        self.img_size = img_size
        self.train = train
        self.samples = []

        # Set directory based on split and find all PNG images in Class folders
        split_dir = "Train" if train else "Test"
        image_paths = glob.glob(os.path.join(root_dir, "Class*", split_dir, "*.PNG"))

        # Map each image to its corresponding label mask path
        for img_path in image_paths:
            img_name = os.path.basename(img_path)
            label_dir = os.path.join(os.path.dirname(img_path), "Label")
            mask_path = os.path.join(label_dir, img_name.replace(".PNG", "_label.PNG"))
            if not os.path.exists(mask_path):
                mask_path = os.path.join(label_dir, img_name)
            self.samples.append({
                "image": img_path,
                "mask": mask_path if os.path.exists(mask_path) else None
            })

    def __len__(self):
        # Return total number of samples
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load and resize grayscale image
        image = Image.open(sample["image"]).convert("L")
        image = TF.resize(image, (self.img_size, self.img_size))

        # Load mask if it exists, otherwise create an empty black mask
        if sample["mask"] is not None:
            mask = Image.open(sample["mask"]).convert("L")
            mask = TF.resize(mask, (self.img_size, self.img_size), interpolation=Image.NEAREST)
        else:
            mask = Image.new("L", (self.img_size, self.img_size), 0)

        # Apply random data augmentations during training (flips and 90-degree rotations)
        if self.train:
            if random.random() > 0.5: image, mask = TF.hflip(image), TF.hflip(mask)
            if random.random() > 0.5: image, mask = TF.vflip(image), TF.vflip(mask)
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                image, mask = TF.rotate(image, angle), TF.rotate(mask, angle)

        # Convert to tensors, binarize the mask, and normalize image pixels
        image = TF.to_tensor(image)
        mask = (TF.to_tensor(mask) > 0).float()
        image = TF.normalize(image, [0.5], [0.5])

        return image, mask