import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, 
                 value=None, gini=None, samples=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.gini = gini
        self.samples = samples
    
    def is_leaf_node(self):
        return self.value is not None

class DecisionTreeClassifier:
    
    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.root = None
        self.n_features = None
        self.n_classes = None
        self.feature_importances_ = None
    
    def _gini_impurity(self, y):
        if len(y) == 0:
            return 0
        counts = np.bincount(y, minlength=self.n_classes)
        probabilities = counts / len(y)
        return 1.0 - np.sum(probabilities ** 2)
    
    def _find_best_split(self, X, y):
        best_ig = -1
        best_feature = None
        best_threshold = None
        
        n_samples, n_features = X.shape
        
        if self.max_features is not None:
            features = np.random.choice(n_features, self.max_features, replace=False)
        else:
            features = range(n_features)
        
        gini_parent = self._gini_impurity(y)
        n = len(y)
        
        for feature in features:
            feature_values = X[:, feature]
            unique_values = np.unique(feature_values)
            
            if len(unique_values) > 50:
                percentiles = np.linspace(0, 100, 50)
                thresholds = np.percentile(unique_values, percentiles)
                thresholds = np.unique(thresholds)
            else:
                thresholds = unique_values
            
            for threshold in thresholds:
                left_mask = feature_values <= threshold
                n_left = np.sum(left_mask)
                n_right = n - n_left
                
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                
                y_left = y[left_mask]
                y_right = y[~left_mask]
                
                gini_left = self._gini_impurity(y_left)
                gini_right = self._gini_impurity(y_right)
                weighted_gini = (n_left / n) * gini_left + (n_right / n) * gini_right
                ig = gini_parent - weighted_gini
                
                if ig > best_ig:
                    best_ig = ig
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_ig
    
    def _build_tree(self, X, y, depth=0):
        n_samples = X.shape[0]
        n_classes = len(np.unique(y))
        
        gini = self._gini_impurity(y)
        most_common_class = np.bincount(y, minlength=self.n_classes).argmax()
        
        if gini < 0.01:
            return Node(value=most_common_class, gini=gini, samples=n_samples)
        
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or
            n_classes == 1):
            return Node(value=most_common_class, gini=gini, samples=n_samples)
        
        best_feature, best_threshold, best_ig = self._find_best_split(X, y)
        
        if best_feature is None:
            return Node(value=most_common_class, gini=gini, samples=n_samples)
        
        left_mask = X[:, best_feature] <= best_threshold
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[~left_mask], y[~left_mask]
        
        left_child = self._build_tree(X_left, y_left, depth + 1)
        right_child = self._build_tree(X_right, y_right, depth + 1)
        
        return Node(
            feature=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child,
            gini=gini,
            samples=n_samples
        )
    
    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.n_classes = len(np.unique(y))
        
        start = time.time()
        self.root = self._build_tree(X, y)
        elapsed = time.time() - start
        
        print(f"Training completed in {elapsed:.2f} seconds")
        
        self._compute_feature_importances(X, y)
        return self
    
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def _compute_feature_importances(self, X, y):
        importances = np.zeros(self.n_features)
        
        def traverse(node):
            if node.is_leaf_node():
                return
            importances[node.feature] += node.samples
            traverse(node.left)
            traverse(node.right)
        
        traverse(self.root)
        self.feature_importances_ = importances / np.sum(importances)
    
    def get_depth(self):
        def _depth(node):
            if node.is_leaf_node():
                return 0
            return 1 + max(_depth(node.left), _depth(node.right))
        return _depth(self.root)
    
    def count_nodes(self):
        def _count(node):
            if node is None:
                return 0
            if node.is_leaf_node():
                return 1
            return 1 + _count(node.left) + _count(node.right)
        return _count(self.root)