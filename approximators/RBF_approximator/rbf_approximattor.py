import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

class RBFApproximator:
    def __init__(self, system_order, control_order, num_gaussians, centers_bounds, centers_init='uniform'):
        """
        Initialization of RBF approximator.

        :param system_order: Dynamic system order.
        :param control_order: Control input order.
        :param num_gaussians: Number of gaussians.
        :param centers_bounds: Gaussian centers bounds (format: [(min, max), ...]).
        :param centers_init: Method for initializing centers ('uniform' or other).
        """
        self.system_order = system_order
        self.control_order = control_order
        self.num_gaussians = num_gaussians
        self.centers_bounds = centers_bounds
        self.centers_init = centers_init

        # Initialization of centers and widths of gaussians
        self.centers = self.initialize_centers()
        self.sigma = self.calculate_sigma()

        # Weights initialization
        self.weights = np.random.randn(self.num_gaussians, self.system_order)

    def initialize_centers(self):
        if self.centers_init == 'uniform':
            return self.initialize_uniform_centers()
        else:
            raise ValueError("Wrong centers initialization form")

    def initialize_uniform_centers(self):
        """
        Initialization of gaussian centers.
        """
        centers = np.zeros((self.num_gaussians, self.system_order + self.control_order))
        for i in range(self.system_order + self.control_order):
            min_val, max_val = self.centers_bounds[i]
            centers[:, i] = np.random.uniform(min_val, max_val, self.num_gaussians)
        return centers

    def calculate_sigma(self):
        """
        Calculation of gaussians width.
        """
        if self.centers is None:
            raise ValueError("Centers are not initialized. Call initialize_centers() first.")
        dist = euclidean_distances(self.centers, self.centers)
        return np.mean(dist) / np.sqrt(2 * self.num_gaussians)

    def rbf(self, x):
        """
        Calculation of gaussians.

        :param x: Input vector (format: [x1, x2, ..., xn, u1, ..., um]).
        """
        if len(x) != self.system_order + self.control_order:
            raise ValueError(f"Input vector must have length {self.system_order + self.control_order}")
        dist = euclidean_distances([x], self.centers)
        return np.exp(-dist**2 / (2 * self.sigma**2))

    def train(self, X, y):
        """
        Training of RBF approximator on dataset.

        :param X: Input vector of dataset (format: [[x1, x2, ..., xn, u1, ..., um], ...]).
        :param y: Next state of system (format: [[x1_next, ..., xn_next], ...]).
        :return: Mean squared error (MSE) after training.
        """
        # Проверка размерности X и y
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")
        if len(X[0]) != self.system_order + self.control_order:
            raise ValueError(f"Each input vector in X must have length {self.system_order + self.control_order}")
        if len(y[0]) != self.system_order:
            raise ValueError(f"Each output vector in y must have length {self.system_order}")

        # Преобразуем y в массив numpy, если это еще не сделано
        y = np.array(y)

        # Количество данных
        N = len(X)

        # Матрица для хранения значений RBF функций для всех данных
        Phi = np.zeros((N, self.num_gaussians))

        # Вычисляем значения RBF функций для всех входных данных
        for i in range(N):
            Phi[i, :] = self.rbf(X[i])

        # Решаем систему: Phi * weights = y
        self.weights, _, _, _ = np.linalg.lstsq(Phi, y, rcond=None)

        # Вычисление ошибки
        y_pred = np.dot(Phi, self.weights)
        mse = np.mean((y_pred - y) ** 2)
        return mse

    def predict(self, X):
        """
        Predict the next state of the system.

        :param X: Input vector (format: [[x1, x2, ..., xn, u1, ..., um], ...]).
        :return: Predicted next state (format: [[x1_next, ..., xn_next], ...]).
        """
        if len(X[0]) != self.system_order + self.control_order:
            raise ValueError(f"Each input vector in X must have length {self.system_order + self.control_order}")

        # Количество данных
        N = len(X)

        # Матрица для хранения значений RBF функций для всех данных
        Phi = np.zeros((N, self.num_gaussians))

        # Вычисляем значения RBF функций для всех входных данных
        for i in range(N):
            Phi[i, :] = self.rbf(X[i])

        # Предсказание
        y_pred = np.dot(Phi, self.weights)
        return y_pred


    def predict(self, X):
        """
        Predict the next state of the system.

        :param X: Input vector (format: [[x1, x2, ..., xn, u1, ..., um], ...]).
        :return: Predicted next state (format: [[x1_next, ..., xn_next], ...]).
        """
        if len(X[0]) != self.system_order + self.control_order:
            raise ValueError(f"Each input vector in X must have length {self.system_order + self.control_order}")

        # Количество данных
        N = len(X)

        # Матрица для хранения значений RBF функций для всех данных
        Phi = np.zeros((N, self.num_gaussians))

        # Вычисляем значения RBF функций для всех входных данных
        for i in range(N):
            Phi[i, :] = self.rbf(X[i])

        # Предсказание
        y_pred = np.dot(Phi, self.weights)
        return y_pred

# Пример использования
if __name__ == "__main__":
    # Пример использования
    system_order = 2  # Порядок системы (x1, x2)
    control_order = 1  # Порядок управления (u1)
    num_gaussians = 20  # Количество гауссовых функций
    centers_bounds = [(-5, 5), (-5, 5), (-1, 1)]  # Границы для центров (x1, x2, u1)

    # Инициализация RBF аппроксиматора
    rbf_approximator = RBFApproximator(
        system_order=system_order,
        control_order=control_order,
        num_gaussians=num_gaussians,
        centers_bounds=centers_bounds,
        centers_init='uniform'
    )

    # Пример данных
    X = [
        [1.0, 2.0, 0.5],  # [x1, x2, u1]
        [2.0, 3.0, 0.7],
        [3.0, 4.0, 0.9],
    ]
    y = [
        [1.5, 2.5],  # [x1_next, x2_next]
        [2.5, 3.5],
        [3.5, 4.5],
    ]

    # Обучение
    mse = rbf_approximator.train(X, y)
    print(f"MSE after training: {mse}")

    # Предсказание
    x_test = [1.5, 2.5, 0.6]  # Входной вектор [x1, x2, u1]
    y_pred = rbf_approximator.predict([x_test])  # Передаем как список из одного элемента
    print(f"Predicted next state: {y_pred}")
