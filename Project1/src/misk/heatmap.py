import numpy as np
import globals
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

lambds = globals.lambds

def create_heatmap(self, MSE_test, lambds, title):
    """
    Function for making a heatmap given lambdas and MSE_test
    Have to make sure that number of degrees is equal to number of
    lambdas - 1 or else it will look kind of wierd.
    """

    # Define polynomial degrees and lambda values
    degrees = np.arange(1, len(MSE_test) + 1)

    fig, ax = plt.subplots()

    ax = sns.heatmap(
        MSE_test,
        cmap="coolwarm",
        # linecolor="black",
        # linewidths=0.8,
        annot=True,
        fmt=".4f",
        cbar=True,
        annot_kws={"fontsize": 8},
        xticklabels=[f"{lmbda:.1f}" for lmbda in np.log10(lambds)],
        yticklabels=degrees,
    )

    ax.set_xlabel(r"$log_{10} \lambda$")
    ax.set_ylabel("Polynomial Degree")  # fontsize=?


    # Set title
    ax.set_title(
        title, fontweight="bold", fontsize=20, pad=25
    )  # fontsize=? fontweihgt='bold'

    fig.tight_layout()

    plt.show()
