# make all plots
import numpy as np

from pathlib import Path
from OLSRegression import (
    OLS_train_test,
    plot_Bias_VS_Variance,
    bootstrap_vs_cross_val_OLS,
)
from sklearn.linear_model import Lasso as SKLasso
from sklearn.linear_model import Ridge as SKRidge
from Models import Ridge, Lasso
from FrankeData import FrankeData
from RegularizedRegression import (
    heatmap_no_resampling,
    heatmap_bootstrap,
    heatmap_sklearn_cross_val,
    heatmap_HomeMade_cross_val,
)

maxDim = 15
lambdas = np.logspace(-3, 4, 13)
figsPath = Path(__file__).parent.parent.parent / "figures"
showPlots = True
savePlots = False
kfolds = 5
N=20
alphaNoise=0.2
n_bootstraps=100

polyDegrees=list(range(4,maxDim+1))





data = FrankeData(
    N, alphaNoise, maxDim, savePlots=savePlots, showPlots=showPlots, figsPath=figsPath
)

kwargs={'polyDegrees': polyDegrees, 'showPlots': True, 'savePlots': False, 'figsPath': figsPath}

# make franke plot
def Franke():
    data.plotFranke()


# THE CLASSIC


def OLSAnalysis():
    OLS_train_test(data, **kwargs)
    plot_Bias_VS_Variance(data, **kwargs)
    bootstrap_vs_cross_val_OLS(data, **kwargs)


# Same analysis for lasso and Ridge as THE Classic
def RidgeAnalysis():

    heatmap_no_resampling(
        data,
        model=Ridge(),
        lambdas=lambdas,
        title="Ridge no resampling",
        **kwargs
    )
    heatmap_bootstrap(
        data,
        model=Ridge(),
        title="Ridge bootstrap",
        lambdas=lambdas,
        n_bootstraps=n_bootstraps,
        **kwargs
    )


    heatmap_sklearn_cross_val(
        data,
        model=SKRidge(),
        title=f"Ridge sklearn CV (kfolds={kfolds})",
        lambdas=lambdas,
        kfolds=kfolds,
        **kwargs
    )

    heatmap_HomeMade_cross_val(
        data,
        model=Ridge(),
        title=f"Ridge CV (kfolds={kfolds})",
        lambdas=lambdas,
        kfolds=kfolds,
        **kwargs
    )


def LassoAnalysis():
    heatmap_no_resampling(
        data,
        model=Lasso(),
        lambdas=lambdas,
        title="Lasso no resampling",
        **kwargs
    )

    heatmap_bootstrap(
        data,
        model=Lasso(),
        title="Lasso bootstrap",
        lambdas=lambdas,
        n_bootstraps=n_bootstraps,
        **kwargs
    )


    heatmap_sklearn_cross_val(
        data,
        model=SKLasso(),
        title=f"Lasso sklearn CV (kfolds={kfolds})",
        lambdas=lambdas,
        kfolds=kfolds,
        **kwargs
    )

    heatmap_HomeMade_cross_val(
        data,
        model=Lasso(),
        title=f"Lasso CV (kfolds={kfolds})",
        lambdas=lambdas,
        kfolds=kfolds,
        **kwargs
    )


# bias variance for OLS using bootstrap

# need to do f. That is compare bootstrap and cross val

# bootstrap_vs_cross_val

if __name__ == "__main__":
    # Franke()
    #OLSAnalysis()
    #RidgeAnalysis()
    LassoAnalysis()
