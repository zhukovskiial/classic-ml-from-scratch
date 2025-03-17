import numpy as np

class KNN:
    # инициализация класса
    def __init__(self, k=3):
        self.k = k
    
    # расчет расстояния евклидовой метрикой
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    # передача данных для обучения
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    # метод для получения предсказаний
    def predict(self, X_test):
        predictions = []
        # для каждого элемента X_test
        for x in X_test:
            # рассчитывается массив с расстояниями элемента до каждого элемента X_train
            distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
            # записываются индексы k элементов X_train, расстояние до которых наименьшее
            k_indices = np.argsort(distances)[:self.k]
            # по индексам определяются метки k элементов из y_train
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            # из k меток выбирается самая частая и записывается в предсказание для элемента из X_test
            most_common = np.bincount(k_nearest_labels).argmax()
            predictions.append(most_common)
        return predictions