import numpy as np

#DecisionTree class
class DecisionTree:

    #Initializing decision tree as an object
    def __init__(self, max_depth=50):
        self.max_depth = max_depth
        self.tree = None

    #Find gini impurity from labels
    def gini(self, y):
        unique, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)

    #Find the best split that has lowest gini impurity
    def best_split(self, X, y):
        best_gini = float('inf')
        best_split = None
        n_samples, n_features = X.shape
        
        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask
                left_y = y[left_mask]
                right_y = y[right_mask]
                gini = (len(left_y) * self.gini(left_y) + len(right_y) * self.gini(right_y)) / len(y)
                
                if gini < best_gini:
                    best_gini = gini
                    best_split = (feature_index, threshold)
        
        return best_split

    #Making the decision tree
    def build_tree(self, X, y, depth=0):
        if depth % 5 == 0 or depth == self.max_depth:
            print(f"Building tree at depth {depth}...")

        if len(np.unique(y)) == 1 or depth >= self.max_depth:
            return np.unique(y)[0]  # Leaf node: return the class label
        
        feature_index, threshold = self.best_split(X, y)
        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask
        
        print(f"Depth {depth}: Splitting on feature {feature_index} at threshold {threshold:.4f}.")
        
        left_tree = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self.build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return (feature_index, threshold, left_tree, right_tree)

    #Builds desicion tree then trains it
    def fit(self, X, y):
        print("Training Decision Tree model...")
        self.tree = self.build_tree(X, y)
        print("Decision Tree training completed.")

    #Predicts sample from the decision tree
    def predict_sample(self, sample, tree):
        if not isinstance(tree, tuple):
            return tree 
        
        feature_index, threshold, left_tree, right_tree = tree
        if sample[feature_index] <= threshold:
            return self.predict_sample(sample, left_tree)
        else:
            return self.predict_sample(sample, right_tree)

    #Predict from multiple samples the labels
    def predict(self, X):
        print("Making predictions...")
        predictions = np.array([self.predict_sample(sample, self.tree) for sample in X])
        print("Predictions completed.")
        return predictions

    #Evaluate decision tree and displays accuracy
    def evaluate(self, X, y, dataset_name="Dataset"):
        """Evaluate the model and print accuracy."""
        y_pred = self.predict(X)
        accuracy = (y_pred == y).sum() / len(y) * 100  # Compute accuracy
        print(f"Accuracy on {dataset_name}: {accuracy:.2f}%")
        return accuracy
