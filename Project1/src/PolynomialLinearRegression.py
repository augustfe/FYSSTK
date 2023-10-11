import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from pathlib import Path


np.random.seed(2114)


class PolynomialLinearRegression:
    def __init__(
        self,
        numPoints: int,
        alphNoise: float,
        maxDim: int,
        showPlots: bool = True,
        savePlots: bool = True,
        Franke: bool = True,
    ):
        """Initialise the values for Polynomial Linear Regression.

        Parameters:
        -----------
            numPoints: (int)
                Number of points to generate.

            alphNoise: (float)
                Amount of noise to add

            maxDim: (int)
                Maximal polynomial dimension

            showPlots: (bool)
                Whether to show plots

            savePlots: (bool)
                Whether to save plots

            Franke: (bool)
                Whether to use the Franke Function to generate points
        """
        self.figs = Path(__file__).parent.parent / "figures"
        self.N = numPoints
        self.alphNoise = alphNoise
        self.maxDim = maxDim
        self.showPlots = showPlots
        self.savePlots = savePlots

        if Franke:
            self.x_, self.y_, self.z_ = self.generate_data(self.N, self.alphNoise)
        else:
            raise NotImplementedError

        (
            self.x_train,
            self.x_test,
            self.y_train,
            self.y_test,
            self.z_train,
            self.z_test,
        ) = train_test_split(self.x_, self.y_, self.z_)

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
        x_ = np.random.uniform(0, 1, N)
        y_ = np.random.uniform(0, 1, N)

        x, y = np.meshgrid(x_, y_)
        x = x.flatten().reshape(-1, 1)
        y = y.flatten().reshape(-1, 1)

        z = self.FrankeFunction(x, y) + alph * np.random.randn(N * N).reshape(-1, 1)
        z = z.flatten().reshape(-1, 1)

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

    @staticmethod
    def MSE(y: np.array, y_pred: np.array) -> float:
        "Calculate mean squared error"
        n = np.size(y)
        return np.sum((y - y_pred) ** 2) / n

    @staticmethod
    def R2Score(y: np.array, y_pred: np.array) -> float:
        "Calculate R2 score"
        s1 = np.sum((y - y_pred) ** 2)
        m = np.sum(y_pred) / y_pred.shape[0]
        s2 = np.sum((y - m) ** 2)

        return 1 - s1 / s2

    def plotBeta(self, betas, methodname):
        dims = [i for i in range(self.maxDim)]
        for dim in dims:
            for beta in betas[dim]:
                plt.scatter(dim + 1, beta, c="r", alpha=0.5)  # , "r", alpha=1)

        plt.title(rf"$\beta$ for {methodname}")
        plt.xticks([dim + 1 for dim in dims])
        plt.xlabel("Polynomial degree")
        plt.ylabel(r"$\beta_i$ value")

        tmp = []
        for beta in betas:
            tmp += list(beta.ravel())

        maxBeta = max(abs(min(tmp)), abs(max(tmp))) * 1.2

        plt.ylim((-maxBeta, maxBeta))

        if self.savePlots:
            plt.savefig(self.figs / f"{methodname}_{self.maxDim}_betas.png", dpi=300)
        if self.showPlots:
            plt.show()
        plt.close()

    def create_OLS_beta(self, X: np.ndarray, z: np.array) -> np.array:
        """Create OLS beta.

        inputs:
            X (n x n matrix): Design matrix
            z (np.array): Solution to optimize for
        returns:
            Optimal solution (np.array)
        """
        return np.linalg.pinv(X.T @ X) @ X.T @ z

    def plotScores(
        self, MSEs_train: np.array, MSEs_test: np.array, R2s: np.array, methodname: str
    ) -> None:
        """
        Plots MSE_train, MSE_test, and R2 values as a function of polynomial degree

        Parameters:
        -----------
            MSE_train: (np.array)
                MSE of traning data and model prediction

            MSE_test: (np.array)
                MSE of test data and model prediction

            R2: (np.array)
                R-squared score of test data and model prediction

            methodname: (str)
                Name of method used to generate values
        """

        # fig, ax1 = plt.subplots()

        xVals = [i + 1 for i in range(self.maxDim)]

        color = "tab:red"
        plt.xlabel("Polynomial degree")
        plt.xticks(xVals)
        plt.ylabel("MSE score", color=color)
        plt.plot(xVals, MSEs_train, label="MSE train", color="r")
        plt.plot(xVals, MSEs_test, label="MSE test", color="g")
        # plt.tick_params(axis="y", labelcolor=color)

        minTestMSE = np.argmin(MSEs_test)
        plt.scatter(
            minTestMSE + 1,
            MSEs_test[minTestMSE],
            marker="x",
            label="Minimum test error",
        )
        plt.title(f"Mean squared error for {methodname}")
        plt.legend()

        if self.savePlots:
            plt.savefig(self.figs / f"{methodname}_{self.maxDim}_MSE.png", dpi=300)
        if self.showPlots:
            plt.show()
        plt.clf()

        color = "tab:blue"
        plt.ylabel("R2", color=color)
        plt.plot(xVals, R2s, label="R2 score", color=color)
        # plt.tick_params(axis="y", labelcolor=color)

        maxR2 = np.argmax(R2s)
        plt.scatter(maxR2 + 1, R2s[maxR2], marker="x", label="Maximum R2")

        plt.legend()
        plt.title(f"R2 Scores by polynomial degree for {methodname}")

        if self.savePlots:
            plt.savefig(self.figs / f"{methodname}_{self.maxDim}_R2.png", dpi=300)
        if self.showPlots:
            plt.show()
        plt.clf()

    def OLS(self):
        methodname = "OLS"
        betas = []

        MSETrain = np.zeros(self.maxDim)
        MSETest = np.zeros(self.maxDim)
        R2Scores = np.zeros(self.maxDim)

        for dim in range(self.maxDim):
            X_train = self.create_X(self.x_train, self.y_train, dim)
            X_test = self.create_X(self.x_test, self.y_test, dim)

            beta = self.create_OLS_beta(X_train, self.z_train)
            betas.append(beta)

            z_tilde = X_train @ beta
            z_pred = X_test @ beta

            MSETrain[dim] = self.MSE(self.z_train, z_tilde)
            MSETest[dim] = self.MSE(self.z_test, z_pred)
            R2Scores[dim] = self.R2Score(self.z_test, z_pred)

        self.plotBeta(betas, methodname)
        self.plotScores(MSETrain, MSETest, R2Scores, methodname)


if __name__ == "__main__":
    PLR = PolynomialLinearRegression(100, 0.2, 5)
    PLR.OLS()
