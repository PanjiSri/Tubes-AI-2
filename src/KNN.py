import numpy as np

class KNNScratch:
    def __init__(self, k_neighbours=3, distance_type='euclidean', p=3):
        self.k_neighbours = k_neighbours
        self.distance_type = distance_type
        self.p = p
        print(f"[DEBUG] KNN initialized with k_neighbours={k_neighbours}, distance_type={distance_type}, p={p}")
        
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        print(f"[DEBUG] Model fitted with {len(X_train)} training samples.")
        
    def calculate_distance(self, x1, x2):
        if self.distance_type == 'euclidean':
            distance = np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.distance_type == 'manhattan':
            distance = np.sum(np.abs(x1 - x2))
        elif self.distance_type == 'minkowski':
            distance = np.sum(np.abs(x1 - x2) ** self.p) ** (1 / self.p)
        else:
            raise ValueError("Tipe tidak diketahui: pilih 'euclidean', 'manhattan', atau 'minkowski'")
        
        print(f"[DEBUG] Distance calculated: {distance}")
        return distance
    
    def predict(self, X_test):
        predictions = []
        print(f"[DEBUG] Predicting {len(X_test)} samples.")
        for idx, x in enumerate(X_test):
            try:
                result = self.predict_one(x)
                predictions.append(result)
            except Exception as e:
                print(f"[DEBUG] Error at index {idx}: {e}")
                raise
        return np.array(predictions)
    
    def predict_one(self, x):
        # Jarak antara x dan semua titik train
        distances = [self.calculate_distance(x, x_train) for x_train in self.X_train]
        print(f"[DEBUG] Distances for one test sample: {distances}")
        
        # Indeks k tetangga terdekat
        k_indices = np.argsort(distances)[:self.k_neighbours]
        print(f"[DEBUG] k_indices: {k_indices}")
        
        # Label dari k tetangga terdekat
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        print(f"[DEBUG] k_nearest_labels: {k_nearest_labels}")
        
        # Label yang paling umum di antara k tetangga terdekat
        most_common_label = max(set(k_nearest_labels), key=k_nearest_labels.count)
        print(f"[DEBUG] Most common label: {most_common_label}")
        
        return most_common_label