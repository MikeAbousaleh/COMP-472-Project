import torch
from torchvision import datasets, transforms, models
import numpy as np
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Subset

# Hyperparameters
NUM_CLASSES = 10
TRAIN_IMAGES_PER_CLASS = 500
TEST_IMAGES_PER_CLASS = 100

#Resizing input
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

# Load Cifar dataset
train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Function to extract limited samples per class
def extract_samples_per_class(dataset, samples_per_class):
    class_indices = {i: [] for i in range(NUM_CLASSES)}
    for idx, (_, label) in enumerate(dataset):
        if len(class_indices[label]) < samples_per_class:
            class_indices[label].append(idx)
    indices = [idx for indices in class_indices.values() for idx in indices]
    return Subset(dataset, indices)

# Extract 500 images per class for training and 100 for testing
train_subset = extract_samples_per_class(train_data, TRAIN_IMAGES_PER_CLASS)
test_subset = extract_samples_per_class(test_data, TEST_IMAGES_PER_CLASS)

# DataLoader for batching
train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=128, shuffle=False)

# Load pre-trained data
resnet = models.resnet18(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  
resnet.eval() 

# Extract features from images
def extract_features(loader, model):
    features, labels = [], []
    with torch.no_grad():
        for images, targets in loader:
            output = model(images)
            features.append(output.view(output.size(0), -1).numpy())
            labels.append(targets.numpy())
    return np.concatenate(features), np.concatenate(labels)

# Extract features for train and test datasets
train_features, train_labels = extract_features(train_loader, resnet)
test_features, test_labels = extract_features(test_loader, resnet)

# Apply PCA to reduce features from 512 to 50 dimensions
pca = PCA(n_components=50)
train_features_pca = pca.fit_transform(train_features)
test_features_pca = pca.transform(test_features)  
