# make all plots
import numpy as np

from pathlib import Path
from OLSRegression import (
    OLS_train_test,
    plot_Bias_VS_Variance,
    bootstrap_vs_cross_val_OLS,
)
from Models import Ridge, Lasso
from Data import TerrainData
from RegularizedRegression import (
    heatmap_no_resampling,
    heatmap_bootstrap,
    heatmap_sklearn_cross_val,
    heatmap_HomeMade_cross_val,
)
import sklearn.linear_model as sklm

np.random.seed(32019)
maxDim = 13
lmbds = np.logspace(-3, 5, 13)
figsPath = Path(__file__).parent.parent / "figures" / "Terrain"

# data = FrankeData(40, 0.2, maxDim, savePlots=False, showPlots=True, figsPath=figsPath)
data = TerrainData(
    figsPath.parent.parent / "DataFiles" / "SRTM_data_Norway_1.tif",
    40,
    maxDim,
    savePlots=False,
    showPlots=True,
    figsPath=figsPath,
)


# make franke plot
def Surface():
    data.plotSurface()
    # data.plotFranke()


# THE CLASSIC


def OLSAnalysis() -> None:
    "Run all the plots for Ordinary Least Squares"
    OLS_train_test(data, savePlots=False, showPlots=True, figsPath=figsPath, maxDim=15)
    # BVData = FrankeData(
    #     20, 0.2, maxDim=25, savePlots=False, showPlots=True, figsPath=figsPath
    # )
    # plot_Bias_VS_Variance(
    #     BVData,
    #     maxDim=13,
    #     showPlots=True,
    #     savePlots=False,
    #     figsPath=figsPath,
    #     title="Few points Bias Variance Tradeoff",
    # )
    plot_Bias_VS_Variance(
        data, maxDim=15, savePlots=False, showPlots=True, figsPath=figsPath
    )
    bootstrap_vs_cross_val_OLS(
        data, maxDim=15, savePlots=False, showPlots=True, figsPath=figsPath
    )


# Same analysis for lasso and Ridge as THE Classic
def RidgeAnalysis() -> None:
    "Run all the plots for Ridge"
    heatmap_no_resampling(
        data,
        maxDim=maxDim,
        lmbds=lmbds,
        model=Ridge(),
        savePlots=False,
        showPlots=True,
        title="MSE Ridge no resampling",
        figsPath=figsPath,
    )
    heatmap_sklearn_cross_val(
        data,
        maxDim=maxDim,
        lmbds=lmbds,
        model=sklm.Ridge(),
        title="MSE Ridge CV from Scikit-learn",
        savePlots=False,
        showPlots=True,
        figsPath=figsPath,
    )
    heatmap_HomeMade_cross_val(
        data,
        maxDim=maxDim,
        lmbds=lmbds,
        model=Ridge(),
        title="MSE Ridge CV",
        savePlots=False,
        showPlots=True,
        figsPath=figsPath,
    )


def LassoAnalysis() -> None:
    "Run all the plots for Lasso"
    heatmap_no_resampling(
        data,
        model=Lasso(),
        maxDim=maxDim,
        lmbds=lmbds,
        title="MSE Lasso no resampling",
        savePlots=False,
        showPlots=True,
        figsPath=figsPath,
    )
    # BootData = FrankeData(
    #     20, 0.2, maxDim=25, savePlots=False, showPlots=True, figsPath=figsPath
    # )
    # heatmap_bootstrap(
    #     BootData,
    #     model=Lasso(),
    #     maxDim=maxDim,
    #     lmbds=lmbds,
    #     title="MSE Lasso bootstrap",
    #     savePlots=False,
    #     showPlots=True,
    #     figsPath=figsPath,
    # )


# bias variance for OLS using boostrap

# need to do f. That is compare boostrap and cross val

# bootstrap_vs_cross_val

if __name__ == "__main__":
    Surface()
    OLSAnalysis()
    RidgeAnalysis()
    # LassoAnalysis()
