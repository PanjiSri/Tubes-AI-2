import pandas as pd
import numpy as np
import pickle

class NaiveBayesFromScratch:
    def __init__(self):
        self.feature_means = {}
        self.feature_vars = {}
        self.class_probs = {}
        self.classes = None
        self.n_features = None
        
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        
        self.classes = np.unique(y)
        self.n_features = X.shape[1]
        n_samples = len(y)
        
        for cls in self.classes:
            self.class_probs[cls] = np.mean(y == cls)
            
            X_class = X[y == cls]
            
            self.feature_means[cls] = np.mean(X_class, axis=0)
            self.feature_vars[cls] = np.var(X_class, axis=0)
            
            self.feature_vars[cls] = self.feature_vars[cls] + 1e-10
            
        return self
    
    def _calculate_likelihood(self, x, mean, var):
        exponent = -0.5 * np.log(2 * np.pi * var) - ((x - mean) ** 2 / (2 * var))
        return np.exp(exponent)
    
    def predict_proba(self, X):

        X = np.asarray(X)
        probas = np.zeros((len(X), len(self.classes)))
        
        for i, cls in enumerate(self.classes):
            log_prob = np.log(self.class_probs[cls])
            
            for j in range(self.n_features):
                mean = self.feature_means[cls][j]
                var = self.feature_vars[cls][j]
                likelihood = self._calculate_likelihood(X[:, j], mean, var)
                log_prob += np.log(likelihood + 1e-10)
            
            probas[:, i] = log_prob

        probas = np.exp(probas - np.max(probas, axis=1, keepdims=True))
        probas = probas / np.sum(probas, axis=1, keepdims=True)
        
        return probas
    
    def predict(self, X):
        probas = self.predict_proba(X)
        return self.classes[np.argmax(probas, axis=1)]
    

    def save_model(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load_model(file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)