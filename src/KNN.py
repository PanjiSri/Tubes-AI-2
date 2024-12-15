import numpy as np

class KNNScratch:
    def __init__(self, k_neighbours=3, distance_type='euclidean', p=3):
        self.k_neighbours = k_neighbours
        self.distance_type = distance_type
        self.p = p
        # print(f"[DEBUG] k_neighbours={k_neighbours}, distance_type={distance_type}, p={p}")
        
    def fit(self, X_train, y_train):
        self.X_train = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else X_train
        self.y_train = y_train.to_numpy() if isinstance(y_train, pd.Series) else y_train

        if not isinstance(self.X_train, np.ndarray):
            raise ValueError("[DEBUG] X_train bukan numpy array.")
        if not isinstance(self.y_train, np.ndarray):
            raise ValueError("[DEBUG] y_train bukan numpy array.")
        
    def calculate_distance(self, x1, x2):
        try:
            if self.distance_type == 'euclidean':
                distance = np.sqrt(np.sum((x1 - x2) ** 2))
            elif self.distance_type == 'manhattan':
                distance = np.sum(np.abs(x1 - x2))
            elif self.distance_type == 'minkowski':
                distance = np.sum(np.abs(x1 - x2) ** self.p) ** (1 / self.p)
            else:
                raise ValueError("Tipe tidak diketahui: pilih 'euclidean', 'manhattan', atau 'minkowski'")
            return distance
        except Exception as e:
            print(f"[DEBUG] Error={e}")
            raise
        
    def predict(self, X_test):
        X_test = X_test.to_numpy() if isinstance(X_test, pd.DataFrame) else X_test

        predictions = []
        for idx, x in enumerate(X_test):
            try:
                print(f"[DEBUG] Predicting sample {idx}")
                result = self.predict_one(x)
                predictions.append(result)
            except Exception as e:
                print(f"[DEBUG] Error di index {idx}, error : {e}")
                raise
        return np.array(predictions)
    
    def predict_one(self, x):
        distances = []
        for idx, x_train in enumerate(self.X_train):
            try:
                distance = self.calculate_distance(x, x_train)
                distances.append(distance)
            except Exception as e:
                print(f"[DEBUG] Error in distance calculation at index {idx}: x={x}, x_train={x_train}, Error={e}")
                raise
                
        # print(f"[DEBUG] Distances for one test sample calculated.")
        
        # Indeks k tetangga terdekat
        k_indices = np.argsort(distances)[:self.k_neighbours]
        # print(f"[DEBUG] k_indices: {k_indices}")
        
        # Label k tetangga terdekat
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # print(f"[DEBUG] k_nearest_labels: {k_nearest_labels}")
        
        # Label paling umum dari antara k tetangga terdekat
        most_common_label = max(set(k_nearest_labels), key=k_nearest_labels.count)
        # print(f"[DEBUG] Most common label: {most_common_label}")
        
        return most_common_label