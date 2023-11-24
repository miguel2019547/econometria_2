import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf
import matplotlib.pyplot as plt
import numpy as np


def acf_plot(data, inx):
    acf_result = acf(data, qstat=True)[inx]
    t = len(acf_result)
    plt.stem(np.arange(t), acf_result)
    plt.title('Autocorrelation Function (MA)')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
