import numpy as np
import matplotlib.pyplot as plt

m = 100 # number of samples

def generate_linear_dataset(m):
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    return X, y

def plot_dataset(X, y):
    plt.plot(X, y, 'o')
    plt.show()

if __name__ == '__main__':
    X, y = generate_linear_dataset(100)
    plot_dataset(X, y)