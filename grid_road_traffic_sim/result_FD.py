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
def get_FD(seed_num):
    folder_name = "FD_calc"
    df_err_cd = pd.read_csv(folder_name + f"/err_cd_SEED{seed_num}.csv") 
    df_flow_sum = pd.read_csv(folder_name + f"/flow_sum_SEED{seed_num}.csv")
    
    return df_err_cd, df_flow_sum

# %%
seed_list = np.arange(1,21)
density_list = np.arange(1,16,1)
df_err_cd_all = pd.DataFrame()
df_flow_sum_all = pd.DataFrame()

for seed_num in seed_list:
    df_err_cd, df_flow_sum = get_FD(seed_num)    
    df_err_cd_all = pd.concat([df_err_cd_all, df_err_cd], axis=1)
    df_flow_sum_all = pd.concat([df_flow_sum_all, df_flow_sum], axis=1)

# %%
output = True
if output:
    output_path = "result_all"
    os.makedirs(output_path, exist_ok=True)
    df_err_cd_all.to_csv(output_path+f"/df_err_cd_all.csv", index=False)
    df_flow_sum_all.to_csv(output_path+f"/df_flow_sum_all.csv", index=False)

# %%
medianprops={'color': 'C1', 'linewidth':3, 'linestyle': '-.',}
meanprops={'markersize': 5, 'markeredgewidth': 2,}

def calc_density_err(err_critical_density):
    plt.figure(figsize=(6.2,5), dpi=300, tight_layout=True)
    plt.boxplot(err_critical_density.T, showmeans=True, medianprops=medianprops, meanprops=meanprops)
    plt.xticks(density_list)
    plt.xlabel(r"$\bar N_c$", fontsize=20)
    plt.ylabel(r"$E(\bar N^{\rho_c}, m_r)$", fontsize=20)
    plt.tick_params(labelsize=20)
    # plt.savefig("err_cd_Nc.pdf", dpi=300)
    plt.show()
    return np.array(err_critical_density)

def calc_flow(flow_sum):
    plt.figure(figsize=(6.5,5), dpi=300, tight_layout=True)
    plt.boxplot(flow_sum.T, showmeans=True, medianprops=medianprops, meanprops=meanprops)
    plt.xticks(density_list)
    plt.xlabel(r"$\bar N_c$", fontsize=20)
    plt.ylabel(r"$\Sigma Q$", fontsize=20)
    plt.tick_params(labelsize=20)
    # plt.savefig("flow_sum_Nc.pdf", dpi=300)
    plt.show()
    return np.array(flow_sum)

err_critical_density = calc_density_err(df_err_cd_all)
flow_sum = calc_flow(df_flow_sum_all)

# %%



