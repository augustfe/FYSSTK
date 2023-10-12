import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from globals import*

class FrankeData:
    """
    Pourpuse of this class is to hold all of our data.
    That way we can use the exact same data for all methods.
    Also it's simple to create a similiar class for the terrain
    data.
    """
    def __init__(
        self,
        numPoints: int,
        alphNoise: float,
        maxDim: int,
    ):
        """.

        Parameters:
        -----------
            numPoints: (int)
                Number of points to generate.

            alphNoise: (float)
                Amount of noise to add

            maxDim: (int)
                Maximal polynomial dimension

        """
        self.N = numPoints
        self.alphNoise = alphNoise
        self.maxDim = maxDim

        self.x_, self.y_, self.z_ = self.generate_data(self.N, self.alphNoise)

        (
            self.x_train,
            self.x_test,
            self.y_train,
            self.y_test,
            self.z_train,
            self.z_test,
        ) = train_test_split(self.x_, self.y_, self.z_, test_size=0.2)

    def generate_data(
        self, N: int, alph: float = 0.2
    ) -> tuple[np.array, np.array, np.array]:
        """Generate the data used for testing Franke.

        inputs:
            N (int): number of points
            alph (float): amount of random noise
        returns:
            x, y, z
        """
        x_ = np.sort(np.random.uniform(0, 1, N))
        y_ = np.sort(np.random.uniform(0, 1, N))

        self.x_raw, self.y_raw = np.meshgrid(x_, y_)
        x = self.x_raw.flatten().reshape(-1, 1)
        y = self.y_raw.flatten().reshape(-1, 1)

        self.z_raw = self.FrankeFunction(self.x_raw, self.y_raw)
        self.z_noise = self.z_raw + alph * np.random.randn(N, N)

        z = self.z_noise.flatten().reshape(-1, 1)

        return x, y, z

    @staticmethod
    def FrankeFunction(x: np.array, y: np.array) -> np.array:
        """Franke's function for evaluating methods.

        Parameters:
        -----------
            x: (np.array)
                values in x direction

            y: (np.array)
                values in y direction

        Returns:
            (np.array) values in z direction
        """

        term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
        term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
        term3 = 0.5 * np.exp(-((9 * x - 7) ** 2) / 4.0 - 0.25 * ((9 * y - 3) ** 2))
        term4 = -0.2 * np.exp(-((9 * x - 4) ** 2) - (9 * y - 7) ** 2)
        return term1 + term2 + term3 + term4

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

    def plotFranke(self):
        fig = plt.figure(figsize=plt.figaspect(0.5))

        ax = fig.add_subplot(1, 2, 1, projection="3d")
        surf = ax.plot_surface(
            self.x_raw,
            self.y_raw,
            self.z_raw,
            cmap=cm.coolwarm,
            linewidth=0,
            antialiased=False,
        )

        # Customize the z axis.
        ax.set_zlim(-0.10, 1.40)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
        ax.set_title("No noise")

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=10)

        ax = fig.add_subplot(1, 2, 2, projection="3d")
        surf = ax.plot_surface(
            self.x_raw,
            self.y_raw,
            self.z_noise,
            cmap=cm.coolwarm,
            linewidth=0,
            antialiased=False,
        )

        # Customize the z axis.
        ax.set_zlim(-0.10, 1.40)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
        ax.set_title("With noise")

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=10)

        fig.suptitle("Frankes function")

        plt.tight_layout()

        if savePlots:
            plt.savefig(figsPath / "FrankesFunction.png", dpi=300)
        if showPlots:
            plt.show()
        plt.clf()
