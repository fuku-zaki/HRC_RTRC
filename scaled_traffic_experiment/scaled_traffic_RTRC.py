# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
import re
import seaborn as sns
import itertools

from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

# %%
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams["font.size"] = 15
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams["legend.fancybox"] = False
plt.rcParams["legend.framealpha"] = 1
plt.rcParams["legend.edgecolor"] = 'grey'
plt.rcParams["legend.markerscale"] = 5

# %%
STEP = 5
INTERVAL = STEP
STEP_AHEAD = 1
SAVE_FIGURE = False

output_path = "imgs_scaled_traffic"
os.makedirs(output_path, exist_ok=True)

LEN_OF_TRAIN = 200  # length of training dataset
LEN_OF_TEST = 100   # length of test dataset

START = 0
STOP = LEN_OF_TEST

car_num_list = [4, 6, 8, 10]
max_vel_list = [300, 450, 600]

TARGET = "power" # power or ave_speed
TARGET = "ave_speed"

# %%
X_list, y_list = [], []
data_name_list = []
for n in car_num_list:
  for max_vel in max_vel_list:
    data_dict = {"num":n, "vel":max_vel}
    data_name_list.append(data_dict)

    data_yX_train = pd.read_csv(f"./train_target_{TARGET}/data_{TARGET}_{n}_{max_vel}.csv")
    data_yX_test = pd.read_csv(f"./test_target_{TARGET}/data_{TARGET}_{n}_{max_vel}.csv")
    X_list.append([data_yX_train.iloc[:, 1:], data_yX_test.iloc[:, 1:]])
    y_list.append([data_yX_train.iloc[:, 0], data_yX_test.iloc[:, 0]])

# %%
def plot_measurement():
    n = 10
    max_vel = 600
    df_yX = pd.read_csv(f"./train_target_{TARGET}/data_{TARGET}_{n}_{max_vel}.csv")
    
    speed_ratio = (1000*1000/3600)/27
    fig, ax = plt.subplots(3,1,figsize=(10,5), dpi=300, tight_layout=True)
    ax[0].plot(speed_ratio*df_yX[[f"vel{_}_lag0" for _ in range(1,11)]][:100] )
    ax[1].plot(1000 * 1/27*df_yX[[f"acc{_}_lag0" for _ in range(1,11)]][:100] )
    ax[2].plot(df_yX[[f"power{_}_lag0" for _ in range(1,11)]][:100] )
    
    ss = "$^2$"
    ax[0].set_ylabel("Speed\n[mm/s]", fontsize=20)
    ax[1].set_ylabel(f"Acceleration\n[mm/s{ss}]", fontsize=20)
    ax[2].set_ylabel("Power\n[mW]", fontsize=20)
    ax[2].set_xlabel("Time", fontsize=20)

    ax[0].tick_params(labelsize=20)
    ax[1].tick_params(labelsize=20)
    ax[2].tick_params(labelsize=20)

    fig.align_ylabels()
    # plt.savefig("miniature_data_example.pdf", dpi=300)
    plt.show()

plot_measurement()

# %%
class Reservoir():
    def __init__(self, lam=1):
        self.lam = lam
    
    def train(self, X, y):
        W = (np.linalg.inv(X.T@X + self.lam * np.identity(X.shape[1]) ))@X.T@y
        self.Wout = W
        return self.Wout
        
    def predict(self, X):
        return X@self.Wout
    
    def evaluation(self, true, pred):
        return np.sqrt(np.mean((true-pred)**2))/np.std(true)

# %%
def predict(model, X_train, X_test, y_train, y_test):
    model.train(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    pred_train[pred_train<0] = 0
    pred_test[pred_test<0] = 0

    train_acc = round(model.evaluation(y_train[START:STOP], pred_train[START:STOP]),4)
    test_acc = round(model.evaluation(y_test[START:STOP], pred_test[START:STOP]),4)
    print("test len:", y_test[START:STOP].shape)
    print('Train NRMSE : ', train_acc); print('Test NRMSE : ', test_acc)
    return pred_train, pred_test, train_acc, test_acc

def plot_predict_(y, pred, name, stop):
    print("-"*10 + "RESULT" +"-"*10)
    fig = plt.figure(figsize =(8, 4), dpi=300, tight_layout=True)
    plt.plot(y[START:stop], label = r"$y$ : Target", color = 'k')
    plt.plot(pred[START:stop], label = r"$\hat y$ : Predicted", color = 'red')
    plt.tick_params(labelsize=25)
    plt.xticks(fontsize=22); plt.yticks(fontsize=25)
    plt.xlabel(r'Time step $k$', fontsize=25); plt.ylabel(r"$y$, $\hat y$", fontsize=25)
    plt.ylim([-0.05,1.1])
    plt.legend(fontsize=25, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2)
    # plt.savefig(output_path + f"/{TARGET}_n{name[0]}v{name[1]}.pdf", dpi=300)
    plt.show()

# %%
test_acc_list = []
for XX, yy, data_name in zip(X_list, y_list, data_name_list):
  nn, max_vel = data_name["num"], data_name["vel"]

  lam = 10
  RModel = Reservoir(lam = lam)
  X_train = np.array(XX[0]); y_train = np.array(yy[0])
  X_test = np.array(XX[1]); y_test = np.array(yy[1])

  scaler = MinMaxScaler()
  y_train = scaler.fit_transform(y_train[:, None]).flatten()
  y_test = scaler.transform(y_test[:, None]).flatten()
  pred_train, pred_test, train_acc, test_acc = predict(RModel, X_train, X_test, y_train, y_test)  
  test_acc_list.append(test_acc)

  print(f"Number of vehicles = {nn}, Max speed = {max_vel}")
  plot_predict_(y_test, pred_test, [nn,max_vel], LEN_OF_TEST)

# %%
plt.figure(figsize=(5,4), dpi=300, tight_layout=True)
plt.plot(car_num_list, np.array(test_acc_list).reshape(4,3), marker=".")
plt.xticks(car_num_list)
plt.legend([r"$V_{\max}=$" + rf"${i}$"  for i in [300,450,600]], fontsize=15, loc="upper center")
plt.xlabel(r"$N_c$", fontsize=20)
plt.ylabel("NRMSE", fontsize=20)
plt.tick_params(labelsize=20)
plt.ylim([0.4,1.1])
# plt.savefig(output_path + f"/{TARGET}_number_accuracy.pdf", dpi=300)
plt.show()

# %%



