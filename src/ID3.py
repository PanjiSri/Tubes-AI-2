import numpy as np
import pandas as pd
from pprint import pprint
import pickle

class ID3Scratch:
    def __init__(self, num_cols=[]):
        self.dict_num_bound = {col: None for col in num_cols}
        self.dict_cat = {}
        self.tree = {}
        self.most_common_label = None

    def fit(self, X, y):
        self.most_common_label = y.mode()[0]

        data = X.copy()
        data['label'] = y

        for col in self.dict_num_bound.keys():
            data = self._discretize_numerical_feature(data, col)

        for col in data.columns[:-1]: 
            self.dict_cat[col] = data[col].unique()

        self.tree = self._build_tree(data)

    def predict(self, X):
        X = X.copy()
        for feature, threshold in self.dict_num_bound.items():
            if threshold is not None:
                X[feature] = np.where(X[feature] < threshold, f'<{threshold}', f'>={threshold}')

        predictions = X.apply(lambda row: self._classify(row, self.tree), axis=1)
        return predictions

    def _entropy(self, data):
        labels = data['label']
        probs = labels.value_counts(normalize=True)
        return -np.sum(probs * np.log2(probs + 1e-9))  

    def _information_gain(self, data, feature):
        total_entropy = self._entropy(data)
        values, counts = np.unique(data[feature], return_counts=True)
        weighted_entropy = 0
        for i in range(len(values)):
            subset = data[data[feature] == values[i]]
            weighted_entropy += (counts[i] / np.sum(counts)) * self._entropy(subset)
        return total_entropy - weighted_entropy

    def _best_feature(self, data):
        features = data.columns[:-1]  # Semua kecuali kolom label
        info_gains = {feature: self._information_gain(data, feature) for feature in features}
        return max(info_gains, key=info_gains.get)

    def _build_tree(self, data):
        if len(data['label'].unique()) == 1:
            return data['label'].iloc[0]

        if data.shape[1] == 1:
            return data['label'].mode()[0]

        best_feature = self._best_feature(data)
        tree = {best_feature: {}}

        for value in self.dict_cat.get(best_feature, []):
            subset = data[data[best_feature] == value].drop(columns=[best_feature])
            if subset.empty:
                tree[best_feature][value] = data['label'].mode()[0]
            else:
                tree[best_feature][value] = self._build_tree(subset)

        return tree

    def _classify(self, row, tree):
        if not isinstance(tree, dict):
            return tree

        feature = next(iter(tree))
        value = row.get(feature)

        if value not in tree[feature]:
            return self.most_common_label

        return self._classify(row, tree[feature][value])

    def _discretize_numerical_feature(self, data, feature):
        sorted_data = data.sort_values(by=feature)
        unique_values = sorted_data[feature].unique()

        best_split = None
        best_gain = -np.inf

        for i in range(1, len(unique_values)):
            threshold = (unique_values[i - 1] + unique_values[i]) / 2
            data_copy = data.copy()
            data_copy[feature] = np.where(data_copy[feature] < threshold, f'<{threshold}', f'>={threshold}')

            gain = self._information_gain(data_copy, feature)
            if gain > best_gain:
                best_gain = gain
                best_split = threshold

        if best_split is not None:
            self.dict_num_bound[feature] = best_split
            data[feature] = np.where(data[feature] < best_split, f'<{best_split}', f'>={best_split}')

        return data

    def display_tree(self):
        pprint(self.tree)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)