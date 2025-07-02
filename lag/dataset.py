import os
from PIL import Image
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {}

        class_names = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
        # print("Found Classes: ", class_names)
        
        for idx, class_name in enumerate(class_names):
            self.class_to_idx[class_name] = idx
            class_dir = os.path.join(data_path, class_name)
            for image_file in os.listdir(class_dir):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(class_dir, image_file))
                    self.labels.append(idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
