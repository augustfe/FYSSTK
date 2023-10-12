from OLSRegression import *
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from globals import *
from matplotlib import cm
from imageio import imread
from RegularizedRegression import *
from resampling import *


# from misk/OLSRegression import *


class TerrainData:

    def __init__(
        self,
        terrainData: np.array,
        numPoints: int,
        maxDim: int,
    ):
        """

        Parameters:
        -----------
        terrainData: (numpy.Ndarray)
            Original dataset

        numPoints: (int)
            Number of points to generate.

        maxDim: (int)
            Maximal polynomial dimension

        """
        self.terrainData = terrainData
        self.N = numPoints
        self.maxDim = maxDim

        self.x_, self.y_, self.t_ = self.ready_data(self.N)

        (
            self.x_train,
            self.x_test,
            self.y_train,
            self.y_test,
            self.z_train,
            self.z_test
        ) = train_test_split(self.x_, self.y_, self.t_, test_size=0.2)

    def ready_data(
        self, N: int
    ) -> tuple[np.array, np.array, np.array]:
        """
        Preprocess the terrain data

        Parameters:
        -----------

        N: (int)
            number of points

        Returns:

            x, y, t
        """

        w, l = self.terrainData.shape

        y_len = w // N
        x_len = l // N

        x_ = np.sort(np.linspace(0, 1, N+1))
        y_ = np.sort(np.linspace(0, 1, N+1))

        self.x_raw, self.y_raw = np.meshgrid(x_, y_)
        x = self.x_raw.flatten().reshape(-1, 1)
        y = self.y_raw.flatten().reshape(-1, 1)

        self.t_data = self.terrainData[::y_len, ::x_len]
        self.scaled_t_data = (
            self.t_data - np.mean(self.t_data))/np.std(self.t_data)
        t = self.scaled_t_data.flatten().reshape(-1, 1)

        return x, y, t

    @staticmethod
    def create_X(x: np.array, y: np.array, n: int) -> np.ndarray:
        "Create design matrix"
        if len(x.shape) > 1:
            x = np.ravel(x)
            y = np.ravel(y)

        N = len(x)
        lgth = (n + 1) * (n + 2) // 2
        X = np.ones((N, lgth))
        for i in range(1, n + 1):
            q = i * (i + 1) // 2
            for k in range(i + 1):
                X[:, q + k] = (x ** (i - k)) * (y**k)

        return X

    def plotContour(self):

        plt.contour(self.x_raw, self.y_raw, self.t_data)
        plt.title('Terrain plot')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()


if __name__ == "__main__":

    terrain1 = imread('../../DataFiles/SRTM_data_Norway_1.tif')
    terrain = np.asarray(terrain1)

    my_terrain = TerrainData(terrain, 100, 15)
    OLS_train_test(my_terrain)
    # plot_Bias_VS_Varaince(my_terrain)
    # heatmap_no_resampling(my_terrain)
    # heatmap_boostrap(my_terrain)
    # bootstrap_polydegrees(my_terrain, range(1, 15), 100)
    # my_terrain.plotContour()
