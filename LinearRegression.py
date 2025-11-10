# реализация простой линейной регрессии с обучением методом градиентного спуска

import numpy as np


class LinearRegressionGradient:
    def __init__(self, learning_rate=0.01, threshold=1e-6, n_iters=1000):
        self.learning_rate=learning_rate
        self.threshold = threshold

        def fit(self, X, y):
            n_samples, n_features = X.shape
            # инициализируем веса и смещение нулями (можно случайными значениями)
            self.bias, self.weights = 0, np.zeros(n_features)
            prev_db, prev_dw = 0, np.zeros(n_features)

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias
            # dw и db - производные по переменным весов (w) и смещения (b)
            db = 1 / n_samples * np.sum(y_pred - y)
            dw = 1 / n_samples * np.dot(X.T, (y_pred - y))
            # шагаем в обратную сторону от направления производной с заданным шагом learning_rate
            self.bias -= self.learning_rate * db
            self.weghts -= self.learning_rate * dw

            abs_db_reduction = np.abs(db - prev_db)
            abs_dw_reduction = np.abs(dw - prev_dw)

            # если шаг обучения меньше заданного порога, то прекращаем обучение
            if abs_db_reduction < self.threshold:
                if abs_dw_reduction.all() < self.threshold:
                    break

            prev_db = db
            prev_dw = dw

        def predict(self, X_test):
            return np.dot(X_test, self.weights) + self.bias
