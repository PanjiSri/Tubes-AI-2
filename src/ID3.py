import numpy as np
import pandas as pd

class ID3Scratch:
    def __init__(self):
        self.tree = {}

    def fit(self, X, y):
        data = X.copy()
        data['label'] = y
        self.tree = self._build_tree(data)
    
    def predict(self, X):
        predictions = X.apply(lambda row: self._classify(row, self.tree), axis=1)
        return predictions

    def _entropy(self, data):
        labels = data['label']
        probs = labels.value_counts(normalize=True)
        entropy = -np.sum(probs * np.log2(probs))
        return entropy

    def _information_gain(self, data, feature):
        total_entropy = self._entropy(data)
        values, counts = np.unique(data[feature], return_counts=True)
        weighted_entropy = np.sum(
            (counts[i] / np.sum(counts)) * self._entropy(data[data[feature] == values[i]])
            for i in range(len(values))
        )
        info_gain = total_entropy - weighted_entropy
        return info_gain

    def _best_feature(self, data):
        features = data.columns[:-1]  # Exclude the target column
        info_gains = {feature: self._information_gain(data, feature) for feature in features}
        return max(info_gains, key=info_gains.get)

    def _build_tree(self, data, tree=None):
        if len(data['label'].unique()) == 1:
            return data['label'].iloc[0]
        
        if data.shape[1] == 1:
            return data['label'].mode()[0]
        
        best_feature = self._best_feature(data)
        tree = {best_feature: {}}
        for value in data[best_feature].unique():
            subset = data[data[best_feature] == value].drop(columns=[best_feature])
            subtree = self._build_tree(subset)
            tree[best_feature][value] = subtree
        return tree

    def _classify(self, row, tree):
        if not isinstance(tree, dict):
            return tree
        feature = next(iter(tree))
        if row[feature] in tree[feature]:
            return self._classify(row, tree[feature][row[feature]])
        else:
            return None  # Handle unknown feature values

    def display_tree(self):
        from pprint import pprint
        pprint(self.tree)


# Contoh data
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)
X = df[['Outlook', 'Temperature', 'Humidity', 'Wind']]
y = df['PlayTennis']

# Training model
model = ID3Scratch()
model.fit(X, y)

# Display the decision tree
model.display_tree()

# Predicting
test_data = pd.DataFrame({
    'Outlook': ['Sunny', 'Rain', 'lol'],
    'Temperature': ['Cool', 'Mild', 'Hot'],
    'Humidity': ['High', 'Normal', 'Normal'],
    'Wind': ['Strong', 'Weak', 'Weak']
})
predictions = model.predict(test_data)
print(predictions)
