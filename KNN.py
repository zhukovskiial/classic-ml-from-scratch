import numpy as np

class KNN:
    def __init__(self, k=3):
        self.k = k
    
    # distance estimation using euclidian metric
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    # getting train data
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    # predict method
    def predict(self, X_test):
        predictions = []
        # for each X_test element
        for x in X_test:
            # calculate the array with distances of the element to each element of X_test
            distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
            # record indexes of k elements X_train with the smallest distance
            k_indices = np.argsort(distances)[:self.k]
            # the indexes are used to determine the labels of k elements from y_train.
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            # the most frequent one of the k labels is selected and recorded in the prediction for the element from X_test
            most_common = np.bincount(k_nearest_labels).argmax()
            predictions.append(most_common)
        return predictions
