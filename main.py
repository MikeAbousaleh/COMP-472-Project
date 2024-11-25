import torch
import numpy as np
from NaiveBayes import NaiveBayes
from DecisionTree import DecisionTree
from MLP import MLP, train_mlp
from VG11 import VGG11, train_vgg11
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
import time

#Load the Cifar dataset
def load_cifar10_data():
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    # Download CIFAR-10 data
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Extract image data and labels
    train_images = torch.stack([trainset[i][0] for i in range(500)])  # 500 training samples
    train_labels = np.array([trainset[i][1] for i in range(500)])

    test_images = torch.stack([testset[i][0] for i in range(100)])  # 100 test samples
    test_labels = np.array([testset[i][1] for i in range(100)])

    # Flatten images Naive Bayes, Decision Tree and MLP
    train_features = train_images.view(len(train_images), -1).numpy()  # Flatten to (500, 150528)
    test_features = test_images.view(len(test_images), -1).numpy()  # Flatten to (100, 150528)

    #Reduce dimensionality for some models
    train_features = train_features[:, :50] 
    test_features = test_features[:, :50]  

    return train_features, train_labels, test_features, test_labels, train_images, test_images

#Plot and show the confusion matrix for each model
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(10), yticklabels=np.arange(10))
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

#Train models from the cifar dataset and then evaluate them
def main():
    train_features, train_labels, test_features, test_labels, train_images, test_images = load_cifar10_data()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Train Naive Bayes
    nb_model = NaiveBayes()
    print("\nTraining Naive Bayes model...")
    nb_model.fit(train_features, train_labels)
    nb_predictions_train = nb_model.predict(train_features)
    nb_predictions_test = nb_model.predict(test_features)

    nb_train_accuracy = (nb_predictions_train.numpy() == train_labels).mean() * 100
    nb_test_accuracy = (nb_predictions_test.numpy() == test_labels).mean() * 100
    print(f"Naive Bayes Training Accuracy: {nb_train_accuracy:.2f}%")
    print(f"Naive Bayes Testing Accuracy: {nb_test_accuracy:.2f}%")

    #Train DecisionTree
    dt_model = DecisionTree(max_depth=10)
    print("\nTraining Decision Tree model...")
    start_time = time.time()
    dt_model.fit(train_features, train_labels)
    elapsed_time = time.time() - start_time
    print(f"Decision Tree training completed in {elapsed_time:.2f} seconds.")

    dt_predictions_train = dt_model.predict(train_features)
    dt_predictions_test = dt_model.predict(test_features)

    dt_train_accuracy = (dt_predictions_train == train_labels).mean() * 100
    dt_test_accuracy = (dt_predictions_test == test_labels).mean() * 100
    print(f"Decision Tree Training Accuracy: {dt_train_accuracy:.2f}%")
    print(f"Decision Tree Testing Accuracy: {dt_test_accuracy:.2f}%")

    #Train MLP
    print("\nTraining MLP model...")
    start_time = time.time()
    mlp_predictions_test, mlp_train_accuracy, mlp_test_accuracy = train_mlp(
    train_features, train_labels, test_features, test_labels, device, epochs=10, batch_size=32, lr=0.01
    )

    elapsed_time = time.time() - start_time
    print(f"MLP training completed in {elapsed_time:.2f} seconds.")
    print(f"MLP Training Accuracy: {mlp_train_accuracy:.2f}%")
    print(f"MLP Testing Accuracy: {mlp_test_accuracy:.2f}%")

    #Train VGG11
    print("\nTraining VGG11 model...")
    vgg11_model, vgg11_train_accuracy, vgg11_test_accuracy, vgg11_predictions_test = train_vgg11(
    train_images, train_labels, test_images, test_labels, device, return_accuracies=True
    )
    print(f"VGG11 Training Accuracy: {vgg11_train_accuracy:.2f}%")
    print(f"VGG11 Testing Accuracy: {vgg11_test_accuracy:.2f}%")

    #Evaluate and generate confusion matrices
    print("\nGenerating confusion matrices...")
    plot_confusion_matrix(test_labels, nb_predictions_test, 'Naive Bayes')
    plot_confusion_matrix(test_labels, dt_predictions_test, 'Decision Tree')
    plot_confusion_matrix(test_labels, mlp_predictions_test, 'MLP')
    plot_confusion_matrix(test_labels, vgg11_predictions_test, 'VGG11')

    #Create performance metrics for each model 
    print("\nNaive Bayes Classification Report:")
    print(classification_report(test_labels, nb_predictions_test))
    
    print("\nDecision Tree Classification Report:")
    print(classification_report(test_labels, dt_predictions_test))
    
    print("\nMLP Classification Report:")
    print(classification_report(test_labels, mlp_predictions_test))
    
    print("\nVGG11 Classification Report:")
    print(classification_report(test_labels, vgg11_predictions_test))

if __name__ == '__main__':
    main()

