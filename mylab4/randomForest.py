from .decisionTree import DecisionTreeClass, most_value
import numpy as np
import pandas as pd

def bootstrap(X, y):
    n_sample = X.shape[0]
    _id = np.random.choice(n_sample, n_sample, replace=True)
    return X.iloc[_id], y.iloc[_id]

class RandomForest:
    def __init__(self, n_trees=5, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTreeClass(min_samples_split=self.min_samples_split, max_depth=self.max_depth, n_features=self.n_features)
            X_sample, y_sample = bootstrap(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        arr_pred = np.array([tree.predict(X) for tree in self.trees])
        final_pred = []
        for i in range(arr_pred.shape[1]):
            sample_pred = arr_pred[:, i]
            final_pred.append(most_value(pd.Series(sample_pred)))
        return np.array(final_pred)
