# statistical plotting library
import numpy as np
import pandas as pd
from scipy.stats import gamma, weibull_min, norm
import matplotlib.pyplot as plt
from fitter import Fitter
import seaborn as sns

df = pd.read_csv('dataset/dataset.csv')


# sns.jointplot(x='tlpm', y='failure', data=df, kind='kde')

pal = {1: 'royalblue', 2: 'forestgreen', 3: 'orange', 4: 'tomato'}
sns.boxplot(x='number of PM', y='tlcm', data=df, palette=pal)

print(df[:2000].describe(include='all').T)
plt.legend({1: 'royalblue', 2: 'forestgreen', 3: 'orange', 4: 'tomato'})
plt.show()
