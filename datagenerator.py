from sklearn.datasets import make_circles, make_moons, make_blobs
import numpy as np

class DataGen:
    
    @staticmethod
    def circle(n_samples, noise):
        x, y = make_circles(n_samples, noise=noise)
        x = x + np.abs(np.min(x, axis=0))

        # Pad third dimension so data will lie on 2d manifold after transformation
        pad = np.ones((n_samples, 1))
        x = np.hstack((x, pad))

        # Normalize
        tmp = np.tile(np.sum(x, axis=1), (3, 1)).T
        x = x / tmp





        return x, y
    
    @staticmethod
    def moons(n_samples, noise):
        x, y = make_moons(n_samples, noise=noise)
        x = x + np.abs(np.min(x, axis=0))

        # Pad third dimension so data will lie on 2d manifold after transformation
        pad = np.ones((n_samples, 1))
        x = np.hstack((x, pad))

        # Normalize
        tmp = np.tile(np.sum(x, axis=1), (3, 1)).T
        x = x / tmp

        return x, y

    @staticmethod
    def gauss(n_samples, n_classes, dim=2):
        x, y = make_blobs(n_samples, dim, n_classes)
        x = x + np.abs(np.min(x, axis=0))

        # Pad third dimension so data will lie on 2d manifold after transformation
        pad = np.ones((n_samples, 1))
        x = np.hstack((x, pad))

        # Normalize
        tmp = np.tile(np.sum(x, axis=1), (3, 1)).T
        x = x / tmp

        return x, y


if __name__ == "__main__":
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt


    dg = DataGen()
    n = 100
    X, y = dg.circle(n, 0)


    xd = X[:, 0] /  X[:,2]
    yd =  X[:,1] /  X[:,2]
    plt.plot(xd, yd, '*')
    plt.show()



