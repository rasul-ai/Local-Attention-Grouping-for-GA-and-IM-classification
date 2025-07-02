import os
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import CustomDataset

class preprocess_dataset:
    def __init__(self, root_dir='/teamspace/studios/this_studio/Local-Attention-Grouping-for-GA-and-IM-classification/dataset'):
        self.root_dir = root_dir

        # Image size
        image_size = (720, 576)

        # Augmentations for training
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
        ])

        # Light transform for val/test (no augmentation)
        test_val_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])

        # Paths for train, val, test
        self.train_path = os.path.join(root_dir, 'train')
        self.val_path = os.path.join(root_dir, 'val')
        self.test_path = os.path.join(root_dir, 'test')

        # Load datasets
        self.training_dataset = CustomDataset(self.train_path, transform=train_transform)
        self.validation_dataset = CustomDataset(self.val_path, transform=test_val_transform)
        self.test_dataset = CustomDataset(self.test_path, transform=test_val_transform)

        print("Train_samples:", len(self.training_dataset))
        print("Validation_samples:", len(self.validation_dataset))
        print("Test_samples:", len(self.test_dataset))

        # Create dataloaders
        self.train_dataloader = DataLoader(self.training_dataset, batch_size=32, shuffle=True)
        self.validation_dataloader = DataLoader(self.validation_dataset, batch_size=32, shuffle=False)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=32, shuffle=False)
