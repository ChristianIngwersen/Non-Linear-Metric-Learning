from sklearn.datasets import make_circles, make_moons, make_blobs
import numpy as np

class DataGen:
    
    @staticmethod
    def circle(n_samples, noise):
        x, y = make_circles(n_samples, noise)
        x = x + np.min(x, axis=0)
        x = x / np.sum(x, axis=1)

        return x, y
    
    @staticmethod
    def moons(n_samples, noise):
        x, y = make_moons(n_samples, noise)
        x = x + np.min(x, axis=0)
        x = x / np.sum(x, axis=1)

        return x, y

    @staticmethod
    def gauss(n_samples, n_classes, dim=2):
        x, y = make_blobs(n_samples, dim, n_classes)
        x = x + np.min(x, axis=0)
        x = x / np.sum(x, axis=1)

        return x, y


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dg = DataGen()
    n = 100
    X, y = dg.circle(n, 0)

    plt.plot(X[:int(n/2), 0], X[:int(n/2), 1], '.b')
    plt.plot(X[int(n/2):, 0], X[int(n/2):, 1], '.r')

    plt.show()
