import numpy as np
from pathlib import Path

maxDim = 13
lambds = np.logspace(-3, 7, 13)
figsPath = Path(__file__).parent.parent.parent / "figures"
showPlots = True
savePlots = False
