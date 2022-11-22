# https://towardsdatascience.com/deep-learning-with-python-neural-networks-complete-tutorial-6b53c0b06af0
import os; 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # to suppress CUDA GPU Tensoflow warning
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras import models, layers, metrics
import shap
import tensorflow as tf    
from sklearn.preprocessing import MinMaxScaler

tf.random.set_seed(5)

df = pd.read_csv('dataset/dataset.csv', index_col=None)

# create neural network model
model = models.Sequential()
model.add(layers.Dense(name="Hidden 1", input_dim=13, units=8, activation='linear'))
model.add(layers.Dropout(name="drop1", rate=0.15))
model.add(layers.Dense(name="Hidden 2", units=4, activation='linear'))
model.add(layers.Dropout(name="drop2", rate=0.15))
model.add(layers.Dense(name="Outpu", units=1, activation='linear'))


print(model.summary())


model.compile(optimizer='Adadelta', loss='mse', metrics=[tf.keras.metrics.MeanSquaredError()])

dataOut = df['failure']
dataIn  = df.drop(['failure'], axis=1)

x_training =  dataIn[0:1000].values
y_training = dataOut[0:1000].values


training = model.fit(x=x_training, y=y_training, batch_size=8, epochs=30, shuffle=True, verbose=1, validation_split=0.3)

# metric extranction
metrics = [k for k in training.history.keys() if ("loss" not in k) and ("val" not in k)]
metric = metrics[0]

# plots for training and validation on dataset
fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)

plt.figure(1) 
# training plot
ax[0].set(title="Training data")    
ax11 = ax[0].twinx()
lns01 = ax[0].plot(training.history['loss'], color='tomato', label='loss')
ax[0].set_xlabel('epochs')
ax[0].set_ylabel('loss', color='tomato')
lns02 = ax11.plot(training.history[metric], label=metric, color='royalblue')
ax11.set_ylabel("score", color='royalblue')

lns = lns01+lns02 
labs = [l.get_label() for l in lns]
ax[0].legend(lns, labs, loc='upper right')
ax[0].grid(alpha=0.4)

# validation plot
ax[1].set(title="Validation data")    
ax22 = ax[1].twinx()    
lns11 = ax[1].plot(training.history['val_loss'], color='tomato', label='loss')
ax[1].set_xlabel('epochs')    
ax[1].set_ylabel('loss', color='tomato')     
lns12 = ax22.plot(training.history['val_'+metric], label=metric, color='royalblue')
ax22.set_ylabel("score", color="royalblue")

lns = lns11+lns12
labs = [l.get_label() for l in lns]
ax[1].legend(lns, labs, loc='upper right')
ax[1].grid(alpha=0.4)

# plt.show()

# impact of feature to the value of target
dataInExplainer = dataIn[-1:].values
dataOutExpected = dataOut[-1:]

explainer = shap.Explainer(model, x_training)
shap_values = explainer(dataInExplainer)

values = shap_values.values[0]
base_values = shap_values.base_values[0][0]
datas = df[-1:].values[0]
columns = list(df.columns)[:-1] # exclude failure column


plt.figure(2, figsize=(10, 8), dpi=80)
exp = shap.Explanation(values, base_values, data=dataInExplainer[0], feature_names=columns)
shap.plots.waterfall(exp, max_display=13, show=False)
plt.rcParams["font.size"] = "12"
plt.title('expected value is: {}'.format(dataOutExpected.values[0]))
plt.grid()
plt.show()