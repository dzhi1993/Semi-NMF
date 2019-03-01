import scipy.io as spio
import numpy as np


mat = spio.loadmat('jet_colormap.mat')
colorData = mat['colormap']

for i in colorData:
    np.savetxt(("jet_colormap.csv"), colorData[i], delimiter=',')
