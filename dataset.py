import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, Subset


class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.images = []
        self.labels = []
        for label in os.listdir(data_path):
            if label == 'cat':
                class_label = 0
            elif label == 'dog':
                class_label = 1
            else:
                continue
            for image_file in os.listdir(os.path.join(data_path, label)):
                self.images.append(os.path.join(data_path, label, image_file))
                self.labels.append(class_label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        # Ensure that the image is in RGB format
        image = image.convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# Define the transformations
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
])
