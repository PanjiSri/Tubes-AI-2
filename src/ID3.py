import numpy as np
import pandas as pd

class ID3Scratch:
    def __init__(self, num_cols=[]):
        self.dict_num_bound = dict.fromkeys(num_cols)
        self.dict_cat = {}
        self.tree = {}

    def fit(self, X, y):
        data = X.copy()
        data['label'] = y

        for col in self.dict_num_bound.keys():
            data = self.discretize_numerical_feature(data, col)

        for col in data.columns[:-1]:
            self.dict_cat[col] = data[col].unique()

        self.tree = self._build_tree(data)
    
    def predict(self, X):
        predictions = X.apply(lambda row: self._classify(row, self.tree), axis=1)
        return predictions

    def _get_rules(self):
        rules = []
        def traverse(node, rule=[]):
            if not isinstance(node, dict):
                rules.append(rule + [node])
            else:
                feature = next(iter(node))
                for value in node[feature]:
                    traverse(node[feature][value], rule + [feature + '=' + value])
        traverse(self.tree)
        return rules

    def _entropy(self, data):
        labels = data['label']
        probs = labels.value_counts(normalize=True)
        entropy = -np.sum(probs * np.log2(probs))
        return entropy

    def _information_gain(self, data, feature):
        total_entropy = self._entropy(data)
        values, counts = np.unique(data[feature], return_counts=True)
        weighted_entropy = sum(
            (counts[i] / np.sum(counts)) * self._entropy(data[data[feature] == values[i]])
            for i in range(len(values))
        )
        info_gain = total_entropy - weighted_entropy
        return info_gain
    
    def _gain_ratio(self, data, feature):
        info_gain = self._information_gain(data, feature)
        split_info = self._entropy(data[[feature, 'label']])
        return info_gain / split_info

    def _best_feature(self, data):
        features = data.columns[:-1]  # Exclude the target column
        info_gains = {feature: self._information_gain(data, feature) for feature in features}
        avg_gain = sum(info_gains.values()) / len(info_gains)
        for feature in info_gains:
            if info_gains[feature] > avg_gain:
                info_gains[feature] = self._gain_ratio(data, feature)
        return max(info_gains, key=info_gains.get)

    def _build_tree(self, data, tree=None):
        if len(data['label'].unique()) == 1:
            return data['label'].iloc[0]
        
        if data.shape[1] == 1:
            return data['label'].mode()[0]
        
        best_feature = self._best_feature(data)
        tree = {best_feature: {}}
        for value in self.dict_cat[best_feature]:
            subset = data[data[best_feature] == value].drop(columns=[best_feature])
            if (subset.empty):
                tree[best_feature][value] = data['label'].mode()[0]
            else:
                subtree = self._build_tree(subset)
                tree[best_feature][value] = subtree
        return tree

    def _classify(self, row, tree):
        if not isinstance(tree, dict):
            return tree
        feature = next(iter(tree))
        if (feature in self.dict_num_bound):
            num_bound = self.dict_num_bound[feature]
            if (row[feature] < num_bound):
                row[feature] = ('<'+str(num_bound))
            else:
                row[feature] = ('>='+str(num_bound))
        if row[feature] in tree[feature]:
            return self._classify(row, tree[feature][row[feature]])
        else:
            return None  # Handle unknown feature values

    def display_tree(self):
        from pprint import pprint
        pprint(self.tree)

    def discretize_numerical_feature(self, data, feature):
        # Sort the values of the feature
        sorted_values = data[[feature, 'label']].sort_values(by=feature)
        
        # Calculate midpoints between consecutive values
        candidates = []
        prev_value = sorted_values.loc[0, 'label']
        for i in range(len(sorted_values) - 1):
            curr_value = sorted_values.loc[i, 'label']
            if (curr_value != prev_value):
                midpoint = (float(sorted_values.loc[i-1, feature]) + float(sorted_values.loc[i, feature])) / 2

                # Calculate information gain
                temp_data = data.copy()
                temp_data[feature] = np.where(temp_data[feature] < midpoint, '<'+str(midpoint), '>='+str(midpoint))
                gain = self._information_gain(sorted_values, feature)

                # Store the candidate
                candidates.append([midpoint, gain])
                prev_value = curr_value
        
        # Find the best candidate
        best_candidate = max(candidates, key=lambda x: x[1])[0]
        # Assign labels based on the best candidate
        self.dict_num_bound[feature] = best_candidate
        data[feature] = np.where(data[feature] < best_candidate, '<'+str(best_candidate), '>='+str(best_candidate))
        
        return data


# Contoh data
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Angka': [13, 15, 19, 10, 19, 15, 19, 14, 14, 14, 14, 15, 13, 15],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)
X = df[['Outlook', 'Temperature', 'Humidity', 'Angka']]
y = df['PlayTennis']

# Training model
model = ID3Scratch(['Angka'])
model.fit(X, y)

# Display the decision tree
model.display_tree()

# Predicting
test_data = pd.DataFrame({
    'Outlook': ['Sunny', 'Rain', 'Rain'],
    'Temperature': ['Cool', 'Mild', 'Hot'],
    'Humidity': ['High', 'Normal', 'Normal'],
    'Angka': [7, 20, 1]
})
predictions = model.predict(test_data)
print(predictions)
