# make all plots
import numpy as np

from pathlib import Path
from OLSRegression import (
    OLS_train_test,
    plot_Bias_VS_Variance,
    bootstrap_vs_cross_val_OLS,
)
from Models import Ridge, Lasso
from FrankeData import FrankeData
from RegularizedRegression import (
    heatmap_no_resampling,
    heatmap_bootstrap,
    heatmap_sklearn_cross_val,
    heatmap_HomeMade_cross_val,
)

maxDim = 13
lambds = np.logspace(-3, 7, 13)
figsPath = Path(__file__).parent.parent.parent / "figures"
showPlots = True
savePlots = False

data = FrankeData(
    40, 0.2, maxDim, savePlots=savePlots, showPlots=showPlots, figsPath=figsPath
)


# make franke plot
def Franke():
    data.plotFranke()


# THE CLASSIC


def OLSAnalysis():
    OLS_train_test(data, showPlots=False, savePlots=False, figsPath=figsPath)
    plot_Bias_VS_Variance(
        data, maxDim=maxDim, showPlots=False, savePlots=False, figsPath=figsPath
    )
    bootstrap_vs_cross_val_OLS(
        data, maxDim=maxDim, showPlots=False, savePlots=False, figsPath=figsPath
    )


# Same analysis for lasso and Ridge as THE Classic
def RidgeAnalysis():
    heatmap_no_resampling(data, model=Ridge())
    heatmap_sklearn_cross_val(data)
    heatmap_HomeMade_cross_val(data)


def LassoAnalysis():
    heatmap_no_resampling(data, model=Lasso())
    heatmap_bootstrap(data, model=Lasso())


# bias variance for OLS using boostrap

# need to do f. That is compare boostrap and cross val

# bootstrap_vs_cross_val

if __name__ == "__main__":
    # Franke()
    # OLSAnalysis()
    RidgeAnalysis()
