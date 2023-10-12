#make all plots
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from OLSRegression import OLS_train_test, plot_Bias_VS_Varaince
from resampling import*
from metrics import*
from HomeCookedModels import OLS, Ridge
from RidgeRegression import Ridge_no_resampling
from FrankeData import FrankeData

data = FrankeData(21, 0.2, 13)
#data.plotFranke(showPlots = True, savePlots = False)

#OLS_train_test(data, showPlots = True, savePlots = False)
#plot_Bias_VS_Varaince(data)
Ridge_no_resampling(data)
