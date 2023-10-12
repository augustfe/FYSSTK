#make all plots
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from OLSRegression import OLS_train_test, plot_Bias_VS_Varaince
from resampling import*
from metrics import*
from HomeCookedModels import OLS, Ridge
from FrankeData import FrankeData
from RegularizedRegression import heatmap_no_resampling
from globals import*


data = FrankeData(33, 0.2, maxDim)

#make franke plot
data.plotFranke()

#THE CLASSIC
#OLS_train_test(data, showPlots = True, savePlots = False)

#Same analysis for lasso and Ridge as THE Classic
#heatmap_no_resampling(data, modelType = Ridge, title="Ridge")
#heatmap_no_resampling(data, modelType = Lasso, title="Lasso")

#bias variance for OLS using boostrap
plot_Bias_VS_Varaince(data)

#need to do f. That is compare boostrap and cross val
