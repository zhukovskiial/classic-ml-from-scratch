import numpy as np

# задаем функцию сигмоиды для превращения logit суммы в вероятность
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LogisticRegressionGradient():
    # инициализируем точку отсечения по умолчанию как 0,5
    # если бисбаланс классов, то надо подбирать другое значение
    def __init__(self, learning_rate=0.001, threshold=0.0001, cutoff=0.5, n_iters=3000):
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.n_iters = n_iters
        self.cutoff = cutoff
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        prev_dw, prev_db = np.zeros(n_features), 0

        # начинаем обучение с заданным кол-ом итераций
        for _ in range(self.n_iters):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_pred)

            # находим прозводные по переменным весов и смещения
            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions-y)

            # обновляем веса и смещение
            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db

            abs_db_reduction = np.abs(db - prev_db)
            abs_dw_reduction = np.abs(dw - prev_dw)

            # если шаг в сторону минимума функции меньше порога threshold, то прекращаем обучение
            if abs_db_reduction < self.threshold:
                if abs_dw_reduction.all() < self.threshold:
                    break

            prev_db = db
            prev_dw = dw

    # предсказываем класс 0 или 1, исходя из заданного порога cutoff
    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        class_pred = [0 if y <= self.cutoff else 1 for y in y_pred]
        return class_pred
        
    # получаем вероятность отнесения к классу 0 или 1
    def predict_proba(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred_proba = sigmoid(linear_pred)
        return y_pred_proba
