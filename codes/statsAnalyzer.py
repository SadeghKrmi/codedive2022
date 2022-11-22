import numpy as np
import pandas as pd
from scipy.stats import gamma, weibull_min, norm
import matplotlib.pyplot as plt
from fitter import Fitter

df = pd.read_csv('dataset/dataset.csv')

print(df[:2000].describe(include='all').T)

plt.figure(1)
plt.hist(df.loc[df['number of PM'] == 1,'tlcm'], bins = 100, range=(0,360),alpha=0.5, label='#of PM 1', density=False, facecolor='royalblue')
plt.hist(df.loc[df['number of PM'] == 2,'tlcm'], bins = 100, range=(0,360),alpha=0.5, label='#of PM 2', density=False, facecolor='forestgreen')
plt.hist(df.loc[df['number of PM'] == 3,'tlcm'], bins = 100, range=(0,360),alpha=0.5, label='#of PM 3', density=False, facecolor='orange')
plt.hist(df.loc[df['number of PM'] == 4,'tlcm'], bins = 100, range=(0,360),alpha=0.5, label='#of PM 4', density=False, facecolor='tomato')


plt.xlabel('days')
plt.ylabel('probability')
plt.grid()
plt.legend(loc='best')



# analysis based on different #number of PMs
# ----------------------------------- PM = 1 -----------------------------------
plt.figure(2)
plt.subplot(2, 2, 1)
dataIn = df.loc[df['number of PM'] == 1,'tlcm']
f = Fitter(dataIn, distributions=['gamma', "weibull_min", "norm"])
f.fit()
print('parameter for #PM=1 are: {}'.format(f.get_best(method = 'sumsquare_error')))
bFitPara = f.get_best(method = 'sumsquare_error')
a = bFitPara['gamma']['a']
l = bFitPara['gamma']['loc']
s = bFitPara['gamma']['scale']

x = np.linspace(gamma.ppf(0.01, a, loc = l, scale = s), gamma.ppf(0.99, a, loc = l, scale = s), 100)
plt.plot(x, gamma.pdf(x, a,loc = l, scale = s), '--', color='royalblue', lw=3, alpha=0.8, label='gamma')
plt.hist(dataIn, bins = 100, range=(0,360), alpha=0.5, label='#PM 1', density=True, facecolor='royalblue')
plt.grid()
plt.legend(loc='best')

# ----------------------------------- PM = 2 -----------------------------------
plt.subplot(2, 2, 2)
dataIn = df.loc[df['number of PM'] == 2,'tlcm']
f = Fitter(dataIn, distributions=['gamma', "weibull_min", "norm"])
f.fit()
print('parameter for #PM=2 are: {}'.format(f.get_best(method = 'sumsquare_error')))
bFitPara = f.get_best(method = 'sumsquare_error')
a = bFitPara['gamma']['a']
l = bFitPara['gamma']['loc']
s = bFitPara['gamma']['scale']

x = np.linspace(gamma.ppf(0.01, a, loc = l, scale = s), gamma.ppf(0.99, a, loc = l, scale = s), 100)

plt.plot(x, gamma.pdf(x, a,loc = l, scale = s), '--', color='forestgreen', lw=3, alpha=0.8, label='gamma')

plt.hist(dataIn, bins = 100, range=(0,360), alpha=0.5, label='#PM 2', density=True, facecolor='forestgreen')
plt.grid()
plt.legend(loc='best')


# ----------------------------------- PM = 3 -----------------------------------
plt.subplot(2, 2, 3)
dataIn = df.loc[df['number of PM'] == 3,'tlcm']
f = Fitter(dataIn, distributions=['gamma', "weibull_min", "norm"])
f.fit()
print('parameter for #PM=3 are: {}'.format(f.get_best(method = 'sumsquare_error')))
bFitPara = f.get_best(method = 'sumsquare_error')
c = bFitPara['weibull_min']['c']
l = bFitPara['weibull_min']['loc']
s = bFitPara['weibull_min']['scale']

x = np.linspace(weibull_min.ppf(0.01, c, loc = l, scale = s), weibull_min.ppf(0.99, c, loc = l, scale = s), 100)

plt.plot(x, weibull_min.pdf(x, c,loc = l, scale = s), '--', color='orange', lw=3, alpha=0.8, label='weibull')

plt.hist(dataIn, bins = 100, range=(0,360), alpha=0.5, label='#PM 3', density=True, facecolor='orange')
plt.grid()
plt.legend(loc='best')



# ----------------------------------- PM = 4 -----------------------------------
plt.subplot(2, 2, 4)
dataIn = df.loc[df['number of PM'] == 4,'tlcm']
f = Fitter(dataIn, distributions=['gamma', "weibull_min", "norm"])
f.fit()
print('parameter for #PM=4 are: {}'.format(f.get_best(method = 'sumsquare_error')))
bFitPara = f.get_best(method = 'sumsquare_error')
l = bFitPara['norm']['loc']
s = bFitPara['norm']['scale']

x = np.linspace(norm.ppf(0.01, loc = l, scale = s), norm.ppf(0.99, loc = l, scale = s), 100)

plt.plot(x, norm.pdf(x, loc = l, scale = s), '--', color='tomato', lw=3, alpha=0.8, label='norm')

plt.hist(dataIn, bins = 100, range=(0,360), alpha=0.5, label='#PM 4', density=True, facecolor='tomato')
plt.grid()
plt.legend(loc='best')


# plt.suptitle("charts")
plt.show()