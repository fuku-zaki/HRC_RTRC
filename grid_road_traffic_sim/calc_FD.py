# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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

# %%
seed_num = 1
seed_folder = f'SEED{seed_num}'
density_list = np.arange(1,16,1)

# %%
import re 
def convert_format(s):
    return re.sub(r'(\d+)to(\d+)', r'\1 → \2', s)

# %%
def get_danmen(density, seed_folder):
   folder_path = get_folder_path(density)
   path = os.path.join(folder_path, seed_folder)
   df_danmen_flow = pd.read_csv(path+'/danmen_flow.csv')
   return df_danmen_flow

def calc_fundamental_diagram(num, density):
    col = sorted(num.iloc[:, 1:].columns)
    transient_time = 5000
    interval = 100
    shape_ = int( (num.shape[0]-transient_time)/interval )

    mean_count_list = []
    flow_list = []

    for c in col:
        count = num[c].values[transient_time:].reshape(shape_,interval)
        mean_count = np.mean(count, axis=1)
        mean_count_list.append(mean_count)

        df_danmen_flow = get_danmen(density, seed_folder)
        flow = np.sum(df_danmen_flow[c].values[transient_time:].reshape(shape_,interval), axis=1)
        flow_list.append(flow)
    
    return mean_count_list, flow_list, col

num_per_road = []
flow_per_road = []
for density in density_list:
    df, df2, df_road_num, df_road_vel, df_theta = get_data(density, seed_folder)
    count_vehicle, count_flow, road_num_col = calc_fundamental_diagram(df_road_num, density)
    num_per_road.append(count_vehicle)
    flow_per_road.append(count_flow)

# %%
def plot_fundamental_diagram_all():
    transient_time = 5000
    interval = 100
    
    cmap_name = "jet"
    cmap = plt.get_cmap(cmap_name, len(density_list))
    
    fig, axes = plt.subplots(3, 5, figsize=(20, 12), tight_layout=True, dpi=300)
    fig.delaxes(axes[2, 4])

    for density, zorder in zip([3, 8, 13], [3, 2, 1]):
        df, df2, df_road_num, df_road_vel, df_theta = get_data(density, seed_folder)
        col = sorted(df_road_num.iloc[:, 1:].columns)
        shape_ = int((df_road_num.shape[0] - transient_time) / interval)
        mean_count_list = []
        flow_list = []

        for c in col:
            count = df_road_num[c].values[transient_time:].reshape(shape_, interval)
            mean_count = np.mean(count, axis=1)
            mean_count_list.append(mean_count)

            df_danmen_flow = get_danmen(density, seed_folder)
            flow = np.sum(df_danmen_flow[c].values[transient_time:].reshape(shape_, interval), axis=1)
            flow_list.append(flow)

        print(fr"$\bar N_c = {density}$")
        if density == 3:
            color = "blue"
        elif density == 8:
            color = "green"
        elif density == 13:
            color = "red"
        
        axes = axes.ravel()
        for ax, c, mean_count, flow in zip(axes, col, mean_count_list, flow_list):
            ax.set_title(convert_format(c), fontsize=25)
            ax.scatter(mean_count, flow, s=5, alpha=0.5,
                       label=fr"$\bar N_c = {density}$",
                       color=cmap(density-1), zorder=zorder)
            ax.axvline(12, color="k", lw=2, linestyle="dashed")
            ax.set_xlim([0, 50])
            ax.set_ylim([0, 20])
            ax.grid()
            ax.tick_params(labelsize=25)
    
    fig.supxlabel(r"$m_r$", fontsize=30)
    fig.supylabel(r"$Q$", fontsize=30)
    # plt.savefig(f"FD_density.pdf", dpi=300)
    plt.show()

plot_fundamental_diagram_all()

# %%
def calc_density_err(num_arr):
    err_critical_density = []
    for density_i in range(0,15):
        n = np.array(num_arr)[density_i]
        criteria = 12 * np.ones([n.shape[0],n.shape[1]])
        rmse = np.sqrt(np.mean((n - criteria) ** 2, axis=1))
        err_critical_density.append(rmse)
    return np.array(err_critical_density)

def calc_flow(flow_arr):
    flow_sum = []
    for density_i in range(0,15):
        n = np.array(flow_arr)[density_i]
        flow_sum.append(np.sum(n, axis=1))
    return np.array(flow_sum)

err_critical_density = calc_density_err(num_per_road)
flow_sum = calc_flow(flow_per_road)

# %%
df_err_cd = pd.DataFrame(err_critical_density, index=density_list, columns=road_num_col)
df_flow_sum = pd.DataFrame(flow_sum, index=density_list, columns=road_num_col)

# %%
output = True
if output:
    output_path = "FD_calc"
    os.makedirs(output_path, exist_ok=True)
    df_err_cd.to_csv(output_path+f"/err_cd_SEED{seed_num}.csv", index=False)
    df_flow_sum.to_csv(output_path+f"/flow_sum_SEED{seed_num}.csv", index=False)

# %%



