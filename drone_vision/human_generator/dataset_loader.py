import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class DroneTopDownDataset(Dataset):
    def __init__(self, image_dir, label_dir, image_size=(512, 512)):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_filenames = sorted([
            f for f in os.listdir(image_dir) if f.endswith(".png") or f.endswith(".jpg")
        ])
        self.image_size = image_size
        self.transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Wraparound for safe skipping
        original_idx = idx
        attempts = 0
        while True:
            img_name = self.image_filenames[idx]
            img_path = os.path.join(self.image_dir, img_name)
            label_path = os.path.join(self.label_dir, img_name.replace(".jpg", ".txt").replace(".png", ".txt"))

            if os.path.exists(label_path):
                break

            print(f"[WARN] Skipping image with no label: {img_name}")
            idx = (idx + 1) % len(self.image_filenames)
            attempts += 1

            if attempts >= len(self.image_filenames):
                raise RuntimeError("No images with labels found in dataset!")

        # Load image
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(image)

        # Load bounding boxes
        boxes = []
        with open(label_path, "r") as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                if len(parts) == 5:
                    _, x, y, w, h = parts
                    x1 = (x - w / 2) * self.image_size[0]
                    y1 = (y - h / 2) * self.image_size[1]
                    x2 = (x + w / 2) * self.image_size[0]
                    y2 = (y + h / 2) * self.image_size[1]
                    boxes.append([x1, y1, x2, y2])

        boxes_tensor = torch.tensor(boxes) if boxes else torch.zeros((0, 4))

        return {
            "pixel_values": image_tensor,
            "labels": boxes_tensor
        }
