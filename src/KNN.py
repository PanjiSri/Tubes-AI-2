import numpy as np

class KNNScratch:
    def __init__(self, k_neighbours=3, distance_type='euclidean', p = 3):
        self.k_neighbours = k_neighbours
        self.distance_type = distance_type
        self.p = p
        
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
    def calculate_distance(self, x1, x2):
        if self.distance_type == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.distance_type == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        elif self.distance_type == 'minkowski':
            return np.sum(np.abs(x1 - x2) ** self.p) ** (1/self.p)
        else:
            raise ValueError("Tipe tidak diketahui: pilih 'euclidean', 'manhattan', atau 'minkowski'")
    
    def predict(self, X_test):
        predictions = []
        for x in X_test:
            result = self.predict_one(x)
            predictions.append(result)
        return np.array(predictions)
    
    def predict_one(self, x):
        # Jarak antara x dan semua titik train
        distances = [self.calculate_distance(x, x_train) for x_train in self.X_train]
        
        # Indeks k tetangga terdekat
        k_indices = np.argsort(distances)[:self.k_neighbours]
        
        # Label dari k tetangga terdekat
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Label yang paling umum di antara k tetangga terdekat
        most_common_label = max(set(k_nearest_labels), key=k_nearest_labels.count)
        
        return most_common_label

    