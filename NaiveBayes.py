import torch
import numpy as np

#Definie the NaiveBayes model
class NaiveBayes:

    #Initializing the model
    def __init__(self):
        self.class_means = None
        self.class_vars = None
        self.class_probs = None
        self.is_trained = False  

    #Train the model
    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))          
        self.class_means = torch.zeros(n_classes, n_features)
        self.class_vars = torch.zeros(n_classes, n_features)
        self.class_probs = torch.zeros(n_classes)
        
        print("Training Naive Bayes model...")
        
        # Calculate the mean, variance, and prior for each class
        for idx in range(n_classes):
            X_class = X[y == idx]
            X_class = torch.tensor(X_class, dtype=torch.float32)  
            self.class_means[idx] = torch.mean(X_class, axis=0)
            self.class_vars[idx] = torch.var(X_class, axis=0)
            self.class_probs[idx] = X_class.shape[0] / n_samples  
            print(f"Class {idx + 1}/{n_classes} processed.")
        
        print("Naive Bayes training completed.")        
        self.is_trained = True
    
    #Predict the class labels 
    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Please train the model first.")
        
        n_samples = X.shape[0]
        predictions = torch.zeros(n_samples, dtype=torch.int64)     
        X = torch.tensor(X, dtype=torch.float32)

        for i in range(n_samples):
            posteriors = []
            for idx in range(len(self.class_means)):
                mean = self.class_means[idx]
                var = self.class_vars[idx]
                prior = self.class_probs[idx]
                
                # Gaussian formula
                likelihood = torch.exp(-0.5 * ((X[i] - mean) ** 2) / (var + 1e-6)) / torch.sqrt(2 * np.pi * (var + 1e-6))
                posterior = torch.prod(likelihood) * prior
                posteriors.append(posterior)
            
            predictions[i] = torch.argmax(torch.tensor(posteriors))
        
        return predictions

    #Evaluate the model and return accuracies
    def evaluate(self, X, y, dataset_name="Dataset"):
        """Evaluate the model and print accuracy."""
        y_pred = self.predict(X)
        y = torch.tensor(y, dtype=torch.int64)  # Convert ground truth to tensor
        accuracy = (y_pred == y).sum().item() / len(y) * 100  # Compute accuracy
        print(f"Accuracy on {dataset_name}: {accuracy:.2f}%")
        return accuracy
    
        
