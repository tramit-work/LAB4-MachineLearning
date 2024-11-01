import numpy as np

def split_node(column, threshold_split):  
    left_node = column[column <= threshold_split].index 
    right_node = column[column > threshold_split].index  
    return left_node, right_node 

def entropy(y_target):  
    values, counts = np.unique(y_target, return_counts=True) 
    result = -np.sum([(count / len(y_target)) * np.log2(count / len(y_target)) for count in counts])
    return result  

def info_gain(column, target, threshold_split):  
    entropy_start = entropy(target)  
    left_node, right_node = split_node(column, threshold_split)  
    n_target = len(target)  
    n_left = len(left_node)  
    n_right = len(right_node) 
    entropy_left = entropy(target[left_node])  
    entropy_right = entropy(target[right_node])  
    weight_entropy = (n_left / n_target) * entropy_left + (n_right / n_target) * entropy_right
    ig = entropy_start - weight_entropy
    return ig

def best_split(dataX, target, feature_id):  
    best_ig = -1  
    best_feature = None 
    best_threshold = None
    for _id in feature_id:
        column = dataX.iloc[:, _id]
        thresholds = set(column)
        for threshold in thresholds:
            ig = info_gain(column, target, threshold)
            if ig > best_ig:
                best_ig = ig
                best_feature = dataX.columns[_id]
                best_threshold = threshold
    return best_feature, best_threshold

def most_value(y_target):  
    value = y_target.value_counts().idxmax()
    return value

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None): 
        self.feature = feature  
        self.threshold = threshold  
        self.left = left  
        self.right = right  
        self.value = value  

    def is_leaf_node(self):  
        return self.value is not None  

class DecisionTreeClass:
    def __init__(self, min_samples_split=2, max_depth=10, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None
        self.n_features = n_features

    def grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_classes = len(y.unique())

        if depth >= self.max_depth or n_samples < self.min_samples_split or n_classes == 1:
            leaf_value = most_value(y)
            return Node(value=leaf_value)

        feature_id = np.random.choice(n_feats, self.n_features, replace=False)
        best_feature, best_threshold = best_split(X, y, feature_id)
        left_node = X[best_feature] <= best_threshold
        right_node = X[best_feature] > best_threshold
        left = self.grow_tree(X.loc[left_node], y.loc[left_node], depth + 1)
        right = self.grow_tree(X.loc[right_node], y.loc[right_node], depth + 1)
        return Node(best_feature, best_threshold, left, right)

    def fit(self, X, y):
        self.n_features = X.shape[1] if self.n_features is None else min(X.shape[1], self.n_features)
        self.root = self.grow_tree(X, y)

    def traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self.traverse_tree(x, node.left)
        else:
            return self.traverse_tree(x, node.right)

    def predict(self, X):
        return np.array([self.traverse_tree(x, self.root) for index, x in X.iterrows()])

def print_tree(node, indent=""):
    if node.is_leaf_node():
        print(f"{indent}Leaf: {node.value}")
        return
    print(f"{indent}Node: If {node.feature} <= {node.threshold:.2f}")
    print(f"{indent}  True:")
    print_tree(node.left, indent + "    ")
    print(f"{indent}  False:")
    print_tree(node.right, indent + "    ")

def accuracy(y_actual, y_pred):
    correct_predictions = np.sum(y_actual == y_pred)
    total_samples = len(y_actual)
    acc = correct_predictions / total_samples
    return acc * 100

y_actual = np.array([1, 0, 1, 1, 0])
y_pred = np.array([1, 1, 1, 0, 0])

print(f"Accuracy: {accuracy(y_actual, y_pred)}%")
