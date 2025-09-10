# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import preprocessing

from scipy.integrate import solve_ivp
from tqdm import tqdm

import warnings
warnings.simplefilter('ignore')

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
def get_random_step():
  ts = 50000
  np.random.seed(42)
  signal = np.random.choice(range(0,10), 10000)
  signal = (signal<5).astype(int)

  np.random.seed(0)
  timerange = np.random.choice(range(5,11), 10000)*10
  y = np.empty(0)
  i = 0
  y_tau = np.empty(0)
  l_t, l_s = [], []
  current_signal = 100
  while len(y)<=ts:
    if current_signal!=signal[i]:
      l_t.append(len(y))
      l_s.append(signal[i])
      current_signal = signal[i]
    y = np.append(y, np.ones(timerange[i])*signal[i])
    # y_tau = np.append(y_tau, np.exp( 1- np.arange(timerange[i])/timerange[i]*np.pi/2 )*signal[i])
    y_tau = np.append(y_tau, np.ones(timerange[i])*signal[i])
    i+=1

  l = pd.DataFrame(columns=["t", "signal"])
  l["t"] = l_t
  l["signal"] = l_s
  return y[:ts]*np.pi*2, y_tau[:ts], l

random_step, random_step_tau, t_fire_df = get_random_step()

# %%
def to_spike(data, df_spike):
  ts = 50000
  df = pd.DataFrame(index=(np.arange(ts)),columns=["binary","output", "sin"])
  df["binary"] = data
  amp_now=0

  for i in range(df_spike.shape[0]):
    step = df_spike["t"].iloc[i]
    tautau = int((step-amp_now)*1)
    tautau = 50
    Amp = 1
    
    decay = (np.exp( 1- np.arange(50)/ (int((step-amp_now)*1)) *np.pi*2))
    if df_spike["signal"].iloc[i] == 1:
      df["output"].iloc[step:step+tautau] = decay*1*Amp*np.sin( np.arange(tautau)/np.pi/((tautau)/10) )#*np.exp(1-0.2*np.arange(50))
      df["sin"].iloc[step:step+tautau] = 1*Amp*np.sin( np.arange(tautau)/np.pi/((tautau)/10) )#*np.exp(1-0.2*np.arange(50))
    elif df_spike["signal"].iloc[i] == 0:
      df["output"].iloc[step:step+tautau] = decay*(-1)*Amp*np.sin( np.arange(tautau)/np.pi/((tautau)/10) )#*np.exp(1-0.2*np.arange(50))
      df["sin"].iloc[step:step+tautau] = -1*Amp*np.sin( np.arange(tautau)/np.pi/((tautau)/10) )#*np.exp(1-0.2*np.arange(50))
    amp_now=step
  
  return df.fillna(0)

# %%
def generate_spike_signal(ts=50000, seed_signal=42, seed_timerange=0):
    np.random.seed(seed_signal)
    signal = np.random.choice(range(0, 10), 10000)
    signal = (signal < 5).astype(int)

    np.random.seed(seed_timerange)
    timerange = np.random.choice(range(5, 11), 10000) * 10

    y = np.empty(0)
    l_t, l_s = [], []
    current_signal = 100
    i = 0

    while len(y) <= ts:
        if current_signal != signal[i]:
            l_t.append(len(y))
            l_s.append(signal[i])
            current_signal = signal[i]
        y = np.append(y, np.ones(timerange[i]) * signal[i])
        i += 1

    df = pd.DataFrame(index=np.arange(ts), columns=["binary", "output", "sin"])
    df["binary"] = y[:ts]
    amp_now = 0

    for i in range(len(l_t)):
        step = l_t[i]
        tautau = 50
        Amp = 1
        decay = np.exp(1 - np.arange(tautau) / (step - amp_now + 1) * np.pi * 2)

        if l_s[i] == 1:
            df["output"].iloc[step:step + tautau] = decay * Amp * np.sin(np.arange(tautau) / np.pi / (tautau / 10))
            df["sin"].iloc[step:step + tautau] = Amp * np.sin(np.arange(tautau) / np.pi / (tautau / 10))
        elif l_s[i] == 0:
            df["output"].iloc[step:step + tautau] = decay * (-Amp) * np.sin(np.arange(tautau) / np.pi / (tautau / 10))
            df["sin"].iloc[step:step + tautau] = -Amp * np.sin(np.arange(tautau) / np.pi / (tautau / 10))

        amp_now = step

    return df.fillna(0)

df_spike_code = generate_spike_signal()

# %%
fig, ax = plt.subplots(1,1, figsize=(10,3), dpi=300, tight_layout=True)
ax2 = ax.twinx()
ax.plot(to_spike(random_step, t_fire_df)["output"][:1000].values, "hotpink")
ax2.plot(random_step_tau[:1000], color="green", label="Signal")
ax.set_xlabel(r"Time step $k$", fontsize=25)
ax.set_ylabel(r"$f(s(k))$", color="hotpink", fontsize=25)
ax.tick_params(axis='x', labelsize=25)
ax.tick_params(axis='y', colors='hotpink', labelsize=25)
ax2.tick_params(axis='y', colors='green', labelsize=25)
ax2.set_ylabel(r"$s(k)$", color="green", fontsize=25)
ax2.tick_params(labelsize=25)
# plt.savefig("traffic_light_input.pdf", dpi=300)
plt.show()

# %%
def plot_signal_cycle_(input_=None):
  if input_ == "random_step":
    title = "Random step"
  elif input_ == None:
    title = "No input"

  theta_list = [0]
  tau = 50
  ts = 50000

  external_input_series=[0]
  for t in range(0, ts-1):
    theta = theta_list[-1]
    if input_ == "random_step":
      external_input = random_step[t]
    elif input_ == None:
      external_input = 0
    external_input_series.append(external_input)

    if input_ == "random_step": 
      theta_update = random_step[t]
    else:
      theta_update = theta + 2*np.pi/tau + external_input
    theta_list.append(theta_update)
  
  if input_ == "random_step":
    signal_state = random_step[:ts]
  else:
    signal_state = np.array(np.mod(theta_list, 2*np.pi))

  df = pd.DataFrame()
  df["time"] = np.arange(ts); df["signal"] = signal_state

  return df, np.array(external_input_series)

df_signal_state, external_input_series = plot_signal_cycle_("random_step")

# %%
def train(X, y, lam=1):
    W = (np.linalg.inv(X.T@X + lam*np.identity(X.shape[1])))@X.T@y
    pred_train = X@W
    return pred_train, W

def plot_result(target, pred, title):
    plt.figure(figsize=(12,2)); plt.title(title)
    plt.plot(target, label="Target")
    plt.plot(pred, label="Predicted")
    plt.legend(); plt.grid(); plt.show()

def RMSE(y, y_pred):
    return np.sqrt(np.mean((y-y_pred)**2))

def plot_Wout(W, df_X, df_road_num):
    plt.figure(figsize=(24, 2))
    plt.title("Wout", fontsize=20)
    plt.bar(list(df_X.columns), np.array(W).reshape(-1))
    plt.xticks(list(df_road_num.iloc[:,1:].columns)+["bias"], rotation=90, fontsize=15)
    plt.show(); plt.close()
    
def predict(df, df_y, n_step_ahead=0, lam=1):
    """
    args:
    n_step_ahead(int): n step ahead prediction
    lam(float): reg. coef.
    """
    df_X = df.copy(); df_X['bias'] = 1
    if n_step_ahead==0:
        X, y = np.array(df_X), np.array(df_y)
    else:
        X = np.array(df_X.iloc[:-n_step_ahead,:])
        y = np.array(df_y.iloc[n_step_ahead:])

    X_train = X[:tr_len]
    y_train = y[:tr_len]
    X_test = X[tr_len+space_len:tr_len+space_len+te_len]
    y_test = y[tr_len+space_len:tr_len+space_len+te_len]

    W = (np.linalg.inv(X_train.T@X_train + lam*np.identity(X_train.shape[1])))@X_train.T@y_train
    pred_train, pred_test = X_train@W, X_test@W

    return y_train, pred_train, y_test, pred_test, W

# %%
def get_folder_path(density):
  root_path = 'sim/data'

  ## =======  parameter  ======
  L = 50
  num_grid = (2,3)
  num_road = num_grid[0]*(num_grid[1]-1)*2 + (num_grid[0]-1)*num_grid[1]*2
  num_car = num_road*density
  n_simulation = 50000
  v_ctrl = 1.
  base_tau=(50,0)
  ## =======  parameter  ======

  folder_name = str(num_grid[0])+'x'+str(num_grid[1])+"_L"+str(L)+'_N'+str(num_car)+'_s'+str(n_simulation)+'_tau'+str(base_tau[0])+'_'+str(base_tau[1])+'_Vctrl'+str(v_ctrl)
  folder_path = os.path.join(root_path, folder_name)
  return folder_path

def get_data(density, seed_folder):
  folder_path = get_folder_path(density)
  path = os.path.join(folder_path, seed_folder)

  df = pd.read_csv(path+'/signal_inflow.csv')
  df2 = pd.read_csv(path+'/signal_input.csv')
  df_road_num = pd.read_csv(path+'/road_car_num.csv')       
  df_road_vel = pd.read_csv(path+'/road_car_ave_v.csv')
  df_theta = pd.read_csv(path+'/signal_theta.csv')
  return df, df2, df_road_num, df_road_vel, df_theta

def make_df(df, df2, df_road_num, df_theta, delay, n_roll, targets=None, fs=0):
  '''Store the features to be used in a DataFrame'''
  df_X = df_road_num.drop('step', axis=1)
  ddd = pd.DataFrame() #ddd: no bias term

  for col in df_X.columns:
    vol_df = pd.DataFrame()
    vol_df[col] = df_X[col]
    ddd = pd.concat((ddd, vol_df), axis=1)
  '''multiplex'''
  if delay !=0:
    for col in df_X.columns:
      vol_df = pd.DataFrame()
      vol_df[col] = df_X[col]
      delay_arr = np.array([df_X[col].shift(i).values for i in range(1, delay+1)]).T
      delay_df = pd.DataFrame(delay_arr, columns=[f"{col}_lag_{i}" for i in range(1, delay+1)])
      ddd = pd.concat((ddd, delay_df), axis=1)
  for i in df_theta.columns[1:]:
      ddd[str(i)+'numUD'] = df[str(i)+'numUD']
      ddd[str(i)+'numRL'] = df[str(i)+'numRL']
  
  ddd = ddd.rolling(n_roll).mean().dropna()
  ddd = ddd.dropna() # drop NaN values
  
  """normalize each column"""
  col_ddd = ddd.columns; ind_ddd = ddd.index
  mm = preprocessing.MinMaxScaler()
  ddd_normalized = mm.fit_transform(ddd.values)
  ddd = pd.DataFrame(data=ddd_normalized, columns=col_ddd, index=ind_ddd)

  """perform a tanh transformation"""
  col_ddd_tanh = [f"{i}_tanh" for i in col_ddd]    
  ddd_normalized_tanh = np.tanh(3*(ddd_normalized-0.5))
  ddd_normalized = np.hstack([ddd_normalized, ddd_normalized_tanh])
  ddd = pd.DataFrame(data=ddd_normalized, index=ind_ddd, columns=[col_ddd.tolist() + col_ddd_tanh])

  print("Data Shape: ", ddd.shape[1], df_X.shape[1], df.iloc[:,1:].shape[1])
  return ddd

def format_Xy(df, df_y, n_step_ahead=0):
  df_X = df.copy(); df_X['bias'] = 1
  if n_step_ahead==0:
      X, y = np.array(df_X), np.array(df_y)
  else:
      X = np.array(df_X.iloc[:-n_step_ahead,:])
      y = np.array(df_y.iloc[n_step_ahead:])
  return X, y

# %%
density = 1
delay = 20
n_roll = 20
seed_num = 20
seed_folder = f'SEED{seed_num}'
df, df2, df_road_num, df_road_vel, df_theta = get_data(density, seed_folder)
ddd = make_df(df, df2, df_road_num, df_theta, delay, n_roll)
ddd

# %%
def get_danmen(density, seed_folder):
   folder_path = get_folder_path(density)
   path = os.path.join(folder_path, seed_folder)
   df_danmen_flow = pd.read_csv(path+'/danmen_flow.csv')
   return df_danmen_flow

def plot_fundamental_diagram(num, density):
    col = sorted(num.iloc[:, 1:].columns)
    mean_count_list = []
    flow_list = []

    for c in col:
        count = num[c].values[5000:].reshape(450,100)
        mean_count = np.mean(count, axis=1)
        mean_count_list.append(mean_count)

        df_danmen_flow = get_danmen(density, seed_folder)
        flow = np.sum(df_danmen_flow[c].values[5000:].reshape(450,100), axis=1)
        flow_list.append(flow)
    
    return mean_count_list, flow_list, col

num_per_road = []
flow_per_road = []
for density in range(1,16):
    df, df2, df_road_num, df_road_vel, df_theta = get_data(density, seed_folder)
    count_vehicle, count_flow, road_num_col = plot_fundamental_diagram(df_road_num, density)
    num_per_road.append(count_vehicle)
    flow_per_road.append(count_flow)

# %%
def calc_density_err():
    err_critical_density = []
    for density_i in range(0,15):
        n = np.array(num_per_road)[density_i]
        criteria = 12 * np.ones([n.shape[0],n.shape[1]])
        rmse = np.sqrt(np.mean((n - criteria) ** 2, axis=1))
        err_critical_density.append(rmse)
    return np.array(err_critical_density)
err_critical_density = calc_density_err()

def calc_flow():
    flow_sum = []
    for density_i in range(0,15):
        n = np.array(flow_per_road)[density_i]
        flow_sum.append(np.sum(n, axis=1))
    return np.array(flow_sum)
flow_sum = calc_flow()

# %%
def plot_vol_pass_example(df, df_road_num, df_theta, dens, n_roll, delay):
    col_list = df_road_num.iloc[:,1:].columns
    fig, axes = plt.subplots(2,1,figsize=(10,8), dpi=300, tight_layout=True)
    axes = axes.ravel()
    ax, ax2 = axes[0], axes[1]
    for col in col_list:
        ax.plot(df_road_num[col][:1000], label=col)
    ax.set_ylabel(r"Traffic volume $m_r$", fontsize=25)


    ddd = pd.DataFrame()
    for i in df_theta.columns[1:]:
            ddd[str(i)+'numUD'] = df[str(i)+'numUD']
            ddd[str(i)+'numRL'] = df[str(i)+'numRL']

    for col in ddd.columns:  
        ax2.plot(df[col][:1000], label=col)
    ax2.set_xlabel(r"Time step $k$", fontsize=25)
    ax2.set_ylabel(r"Passing volume $m_i$", fontsize=25)  
    ax.tick_params(labelsize=25)
    ax2.tick_params(labelsize=25)
    # plt.savefig("sim_data_example.pdf", dpi=300)
    plt.show()

plot_vol_pass_example(df, df_road_num, df_theta, density, n_roll=20, delay=20)

# %%
def plot_vol_pass(df, df_road_num, df_theta, dens, n_roll, delay):
  col_list = df_road_num.iloc[:,1:].columns

  if delay !=0:
    for col in col_list:
      vol_df = pd.DataFrame()
      vol_df[col] = df_road_num[col]
      delay_arr = np.array([df_road_num[col].shift(i).values for i in range(1, delay+1)]).T
      delay_df = pd.DataFrame(delay_arr, columns=[f"{col}_lag_{i}" for i in range(1, delay+1)])

  ddd = pd.DataFrame()
  for col in col_list:
    ddd[col] = df_road_num[col]
  col_ddd = ddd.columns; ind_ddd = ddd.index
  mm = preprocessing.MinMaxScaler()
  ddd_normalized = mm.fit_transform(ddd.values)
  ddd = pd.DataFrame(data=ddd_normalized, columns=col_ddd, index=ind_ddd)
  
  ddd_normalized_tanh = pd.DataFrame(np.tanh(3*(ddd_normalized-0.5)))
  volume_tilde = ddd_normalized_tanh[:1000].rolling(n_roll).mean().dropna()

  ddd = pd.DataFrame()
  for i in df_theta.columns[1:]:
        ddd[str(i)+'numUD'] = df[str(i)+'numUD']
        ddd[str(i)+'numRL'] = df[str(i)+'numRL']
 
  col_ddd = ddd.columns; ind_ddd = ddd.index
  mm = preprocessing.MinMaxScaler()
  ddd_normalized = mm.fit_transform(ddd.values)
  ddd = pd.DataFrame(data=ddd_normalized, columns=col_ddd, index=ind_ddd)
  
  ddd_normalized_tanh = pd.DataFrame(np.tanh(3*(ddd_normalized-0.5)))
  passing_tilde = ddd_normalized_tanh[:1000].rolling(n_roll).mean().dropna()

  col_list = df_road_num.iloc[:,1:].columns
  fig, axes = plt.subplots(2,1,figsize=(10,8), dpi=300, tight_layout=True)
  axes = axes.ravel()
  ax, ax2 = axes[0], axes[1]
  ax.plot(volume_tilde, label=col)
  ax.set_ylabel(r"$g(\tilde{m}_r)$", fontsize=25)
  ax2.plot(passing_tilde, label=col)
  ax2.set_xlabel(r"Time step $k$", fontsize=25)
  ax2.set_ylabel(r"$g(\tilde{m}_i)$", fontsize=25)
  ax.tick_params(labelsize=25)
  ax2.tick_params(labelsize=25)
  # plt.savefig("sim_data_transformed_example.pdf", dpi=300)
  plt.show()

plot_vol_pass(df, df_road_num, df_theta, density, n_roll=20, delay=20)

# %%
def calc_MC(X, y, lam=0.00001, fit_len=4000, D=2000):
    MC = 0
    MC_list = []
    X_train = X[D:fit_len+D]
    tau_range = 1
    
    for tau in tqdm(range(0, D, tau_range)):
      u_tau = y[D-tau:fit_len+D-tau]
      pred_train, Wout = train(X_train, u_tau, lam)
      output = X_train@Wout
      corr = np.corrcoef(u_tau, output)
      r_coef = corr[0, 1] ** 2
      
      MC_list.append(r_coef)
      MC += r_coef
    print(f"MC = {MC}")
    return MC_list, MC

# %%
lam = 10                # reg coef.
n_step_ahead = 0        # n-step ahead prediction
delay = 20              # multiplex
n_roll = 20             # Use the average of the measured values over n_roll steps as the representative value

# training data length, test data length, and the interval between them (0 if concatenated)
tr_len = 2000; te_len = 2000; space_len = 0
transient_len = 5000    # transient length to discard

time_shift_arr = np.arange(0, 40000, 2000)
density_list = np.arange(1,16,1)

# %%
# Calculation
df_result = pd.DataFrame()
ddd_list, df_road_list = [], []
acc_train_list, acc_test_list = [], []
MC_list_all, MC_all = [], []
Wout_list = []

for density in density_list:
  print(f">> Density = {density}")
  df, df2, df_road_num, df_road_vel, df_theta = get_data(density, seed_folder)
  ddd = make_df(df, df2, df_road_num, df_theta, delay, n_roll)
  df_Y = to_spike(random_step, t_fire_df)["output"].values[ddd.index[0]:]
  ddd_list.append(ddd)
  df_road_list.append(df_road_num)
  ddd, df_Y = ddd[transient_len:], df_Y[transient_len:]
  X, y = format_Xy(ddd, df_Y, n_step_ahead=n_step_ahead)
  print(f" X.shape: {X.shape}, y.shape: {y.shape}")

  """Memory Capacity"""
  D = 2000         # maximum tau (\tau_{max})
  MC_list, MC = calc_MC(X, y, 0.00001, 4000, D)
  MC_list_all.append(MC_list); MC_all.append(MC)
  
  acc_train_shift, acc_test_shift = [], []
  for time_shift in time_shift_arr:
    print(f"> Time Shift", time_shift)
    y_train, pred_train, y_test, pred_test, Wout = predict(ddd[time_shift:], df_Y[time_shift:], n_step_ahead=n_step_ahead, lam=lam)  
    Wout_list.append(Wout)
    train_prec = RMSE(y_train, pred_train); test_prec = RMSE(y_test, pred_test)
    print(f"Train RMSE = {train_prec}, Test RMSE = {test_prec}")
    acc_train_shift.append(train_prec); acc_test_shift.append(test_prec)

  acc_train_list.append(acc_train_shift)
  acc_test_list.append(acc_test_shift)

# %%
# save
output_path = f"pred_result_SEED{seed_num}"
os.makedirs(output_path, exist_ok=True)

# %%
def output_result():
    for d in range(density_list.shape[0]):
        pd.DataFrame(np.vstack([acc_train_list[d],acc_test_list[d]]).T, columns=["train","test"]).to_csv(output_path+f"/acc_density{d+1}.csv", index=False)
    pd.DataFrame(np.array(MC_list_all).T, columns=density_list).to_csv(output_path+"/MC_series.csv", index=False)
    pd.DataFrame(np.array(MC_all)[:, None].T, columns=density_list).to_csv(output_path+"/MC.csv", index=False)
output_result()

# %%



