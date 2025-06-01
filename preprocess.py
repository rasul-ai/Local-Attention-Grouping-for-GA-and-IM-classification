import os
import random
import torch
from dataset import CustomDataset
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, Subset
from collections import defaultdict

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((720, 576)),
    transforms.ToTensor(),
])


class preprocess_dataset():
    data_path = './dataset/end_val/resized'
    
    dataset = CustomDataset(data_path, transform=transform)

    random_seed = 42
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Define the split ratio for validation and test sets
    validation_ratio = 0.1  # 10% of the data for validation
    test_ratio = 0.2        # 20% of the data for testing

    # Get total samples and class-wise indices
    total_samples = len(dataset)
    class_indices = defaultdict(list)

    for idx, label in enumerate(dataset.labels):
        class_indices[label].append(idx)

    validation_indices = []
    test_indices = []
    train_indices = []

    for label, indices in class_indices.items():
        random.shuffle(indices)

        num_val = int(len(indices) * validation_ratio)
        num_test = int(len(indices) * test_ratio)

        validation_indices.extend(indices[:num_val])
        test_indices.extend(indices[num_val:num_val + num_test])
        train_indices.extend(indices[num_val + num_test:])

    # Create subsets
    training_dataset = Subset(dataset, train_indices)
    validation_dataset = Subset(dataset, validation_indices)
    test_dataset = Subset(dataset, test_indices)

    print("Train_samples:", len(training_dataset))
    print("Validation_samples:", len(validation_dataset))
    print("Test_samples:", len(test_dataset))

    # Create dataloaders
    train_dataloader = DataLoader(training_dataset, batch_size=1, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)




# import random
# import torch
# from dataset import CustomDataset
# from torchvision import transforms
# from torch.utils.data import DataLoader, Dataset, Subset

# # Define the transformations
# transform = transforms.Compose([
#     transforms.Resize((1024, 1024)),
#     transforms.ToTensor(),
# ])


# class preprocess_dataset():
#     data_path = './dataset/demo'
    
#     dataset = CustomDataset(data_path, transform=transform)

#     random_seed = 42
#     random.seed(random_seed)
#     torch.manual_seed(random_seed)

#     # Define the split ratio for validation and test sets
#     validation_ratio = 0.1  # 20% of the data for validation
#     test_ratio = 0.2       # 20% of the data for testing

#     # Calculate the number of samples for validation and test sets
#     total_samples = len(dataset)
#     num_validation_samples = min(int(validation_ratio * total_samples), total_samples // 2)
#     num_test_samples = min(int(test_ratio * total_samples), total_samples // 2)

#     # Create an index list for shuffling
#     indices_3 = list(range(total_samples))
#     random.shuffle(indices_3)

#     # Split the indices into validation and test sets while maintaining class balance
#     train_indices = []
#     validation_indices = []
#     test_indices = []
#     indices_2 = []
#     v_num_cat = 0
#     v_num_dog = 0
#     t_num_cat = 0
#     t_num_dog = 0

#     for idx in indices_3:
#         label = dataset.labels[idx]
#         if label == 0 and v_num_cat < num_validation_samples / 2:
#             validation_indices.append(idx)
#             v_num_cat += 1
#         elif label == 1 and v_num_dog < num_validation_samples / 2:
#             validation_indices.append(idx)
#             v_num_dog += 1
#         else:
#             indices_2.append(idx)

#     for idx in indices_2:
#         label = dataset.labels[idx]
#         if label == 0 and t_num_cat < num_test_samples / 2:
#             test_indices.append(idx)
#             t_num_cat += 1
#         elif label == 1 and t_num_dog < num_test_samples / 2:
#             test_indices.append(idx)
#             t_num_dog += 1
#         else:
#             train_indices.append(idx)

#     # Create custom datasets for validation and test sets
#     training_dataset = Subset(dataset, train_indices)
#     validation_dataset = Subset(dataset, validation_indices)
#     test_dataset = Subset(dataset, test_indices)

#     print("Train_samples: " + str(len(training_dataset)))
#     print("Validation_samples: " + str(len(validation_dataset)))
#     print("Test_samples: " + str(len(test_dataset)))


#     # Create data loaders for validation and test sets
#     train_dataloader = DataLoader(training_dataset, batch_size=1, shuffle=True)
#     validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=True)
#     test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    



