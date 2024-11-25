import torch
import torch.nn as nn
import numpy as np

#Define the MLP class
class MLP(nn.Module):

    #Initialize MLP model
    def __init__(self, input_size=50, hidden_size=128, output_size=10):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    #Defining forward pass
    def forward(self, x):
        return self.layers(x)

#Train and evaluate MLP
def train_mlp(train_features, train_labels, test_features, test_labels, device, epochs=10, batch_size=32, lr=0.01):
    train_features = torch.tensor(train_features, dtype=torch.float32).to(device)
    train_labels = torch.tensor(train_labels, dtype=torch.long).to(device)
    test_features = torch.tensor(test_features, dtype=torch.float32).to(device)
    test_labels = torch.tensor(test_labels, dtype=torch.long).to(device)

    input_size = train_features.shape[1]
    output_size = len(np.unique(train_labels))  
    model = MLP(input_size=input_size, hidden_size=128, output_size=output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    #Training loop for MLP
    print("Training MLP model...")
    for epoch in range(epochs):
        model.train()
        indices = np.arange(len(train_features))
        np.random.shuffle(indices)  

        for start in range(0, len(indices), batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            batch_features = train_features[batch_indices]
            batch_labels = train_labels[batch_indices]

            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    # Evaluation of MLP with training and testing accuracies
    print("Evaluating MLP model...")
    model.eval()
    with torch.no_grad():
        train_outputs = model(train_features)
        test_outputs = model(test_features)

        train_predictions = torch.argmax(train_outputs, axis=1)
        test_predictions = torch.argmax(test_outputs, axis=1)

        train_accuracy = (train_predictions == train_labels).sum().item() / len(train_labels) * 100
        test_accuracy = (test_predictions == test_labels).sum().item() / len(test_labels) * 100

    print(f"Train Accuracy: {train_accuracy:.2f}%")
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    return test_predictions.cpu().numpy(), train_accuracy, test_accuracy
