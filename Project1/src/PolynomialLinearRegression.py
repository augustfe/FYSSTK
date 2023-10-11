import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

plt.rcParams["font.family"] = "sans-serif"

np.random.seed(14)


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
        xVals = [i + 1 for i in range(self.maxDim)]

        color = "tab:red"
        plt.xlabel("Polynomial degree")
        plt.xticks(xVals)
        plt.ylabel("MSE score")
        plt.plot(xVals, MSEs_train, label="MSE train", color="r")
        plt.plot(xVals, MSEs_test, label="MSE test", color="g")

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
        plt.ylabel("$R^2$")
        plt.plot(xVals, R2s, label="$R^2$ score", color=color)

        maxR2 = np.argmax(R2s)
        plt.scatter(maxR2 + 1, R2s[maxR2], marker="x", label="Maximum $R^2$")

        plt.legend()
        plt.title(f"$R^2$ Scores by polynomial degree for {methodname}")

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

    def create_ridge_beta(
        self, X: np.ndarray, z: np.array, lmbd: float = 0
    ) -> np.array:
        """Create ridge beta.

        inputs:
            X (n x n matrix): Design matrix
            z (np.array): Solution to optimize for
            lmbd (float): lambda value for ridge
        returns:
            Optimal solution (np.array)
        """
        Identity = np.identity(X.shape[1])
        return np.linalg.pinv(X.T @ X + lmbd * Identity) @ X.T @ z

    def Ridge(self, nLambdas: int):
        methodname = "RIDGE"
        betas = []
        minLmbda = -3
        maxLmbda = 5
        lmbdas = np.logspace(minLmbda, maxLmbda, nLambdas)
        MSETrain = np.zeros((self.maxDim, nLambdas))
        MSETest = np.zeros((self.maxDim, nLambdas))
        R2Scores = np.zeros((self.maxDim, nLambdas))

        scaler = StandardScaler()

        for dim in range(self.maxDim):
            for i, lmbda in enumerate(lmbdas):
                X_train = self.create_X(self.x_train, self.y_train, dim)
                X_test = self.create_X(self.x_test, self.y_test, dim)

                scaler.fit(X_train)
                scaler.transform(X_train)
                scaler.transform(X_test)

                beta = self.create_ridge_beta(X_train, self.z_train, lmbda)
                # betas.append(beta)

                z_tilde = X_train @ beta
                z_pred = X_test @ beta

                MSETrain[dim, i] = self.MSE(self.z_train, z_tilde)
                MSETest[dim, i] = self.MSE(self.z_test, z_pred)
                R2Scores[dim, i] = self.R2Score(self.z_test, z_pred)

        self.lambda_heat_map(MSETest, lmbdas, methodname)

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

        if self.savePlots:
            plt.savefig(self.figs / "FrankesFunction.png", dpi=300)
        if self.showPlots:
            plt.show()
        plt.clf()

    def lambda_heat_map(self, MSE_test, lmbdas, method):
        """
        Function for making a heatmap given lambdas and MSE_test
        Have to make sure that number of degrees is equal to number of
        lambdas - 1 or else it will look kind of wierd.
        """

        # Define polynomial degrees and lambda values
        degrees = np.arange(1, len(MSE_test) + 1)

        fig, ax = plt.subplots()  # figsize=(10, 8))

        ax = sns.heatmap(
            MSE_test,
            cmap="coolwarm",
            # linecolor="black",
            # linewidths=0.8,
            annot=True,
            fmt=".4f",
            cbar=True,
            annot_kws={"fontsize": 8},
            xticklabels=[f"{lmbda:.1f}" for lmbda in np.log10(lmbdas)],
            yticklabels=degrees,
        )

        # Set x and y-axis labels
        # ax.set_xticks(
        #     np.arange(0.5, len(lambdas) + 0.5), [f"{lmbda:1.0e}" for lmbda in lambdas]
        # )
        # ax.set_yticks(np.arange(0.5, len(degrees) + 0.5), degrees)
        ax.set_xlabel(r"$log_{10} \lambda$")
        ax.set_ylabel("Polynomial Degree")  # fontsize=?

        # Set title
        ax.set_title(
            "MSE heatmap", fontweight="bold", fontsize=20, pad=25
        )  # fontsize=? fontweihgt='bold'

        fig.tight_layout()

        if self.savePlots:
            plt.savefig(self.figs / f"Heatmap_{method}.png", dpi=300)
        if self.showPlots:
            plt.show()
        plt.clf()


if __name__ == "__main__":
    PLR = PolynomialLinearRegression(30, 0.2, 15)
    # PLR.plotFranke()
    # PLR.OLS()
    PLR.Ridge(15)
