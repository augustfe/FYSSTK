# make all plots
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from OLSRegression import OLS_train_test, plot_Bias_VS_Varaince
from resampling import *
from metrics import *
from Models import OLS, Ridge, Lasso
from FrankeData import FrankeData
from RegularizedRegression import heatmap_no_resampling, heatmap_bootstrap, heatmap_sklearn_cross_val, heatmap_HomeMade_cross_val
from globals import *


data = FrankeData(60, 0.2, maxDim)

# make franke plot
#data.plotFranke()

# THE CLASSIC
#OLS_train_test(data, showPlots = True, savePlots = False)

# Same analysis for lasso and Ridge as THE Classic
#heatmap_no_resampling(data, model = Ridge())
#heatmap_no_resampling(data, model = Lasso())


# bias variance for OLS using boostrap
#plot_Bias_VS_Varaince(data)

# need to do f. That is compare boostrap and cross val
#heatmap_bootstrap(data)
heatmap_sklearn_cross_val(data)
#heatmap_HomeMade_cross_val(data)
