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
def get_result():
    folder_name = "result_all/"
    df_err_cd_all = pd.read_csv(folder_name + "df_err_cd_all.csv")
    df_flow_sum_all = pd.read_csv(folder_name + "df_flow_sum_all.csv")
    df_acc = pd.read_csv(folder_name + "df_acc.csv")
    df_MC = pd.read_csv(folder_name + "df_MC.csv")
    
    return df_err_cd_all, df_flow_sum_all, df_acc, df_MC

# %%
seed_list = np.arange(1,21)
density_list = np.arange(1,16,1)
df_err_cd, df_flow_sum, df_acc, df_MC = get_result()    

# %%
def plot_density_flow(x_list, y_list, x_label, y_label):
    """
    Parameters:
        x_list (list[float]): List of X coordinates.
        y_list (list[float]): List of Y coordinates.
        cmap_name (str, optional): Name of the colormap. Defaults to "viridis".
        marker_size (int, optional): Marker size for the scatter plot. Defaults to 100.
        legend_marker_size (int, optional): Marker size for the legend. Defaults to 200.
    """

    cmap_name="jet"; marker_size=100; legend_marker_size=10
    cmap = plt.get_cmap(cmap_name)
    fig, ax = plt.subplots(figsize=(6,5), dpi=300, tight_layout=True)
    sc = ax.scatter(x_list, y_list, c=density_list, cmap=cmap, edgecolors='black', s=marker_size)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(r"$\bar N_c$", fontsize=20)
    unique_density = sorted(set(density_list))
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markersize=legend_marker_size/10,
                   markerfacecolor=cmap(d))
                   for d in unique_density
    ]
    ax.set_xlabel(x_label, fontsize=20); ax.set_ylabel(y_label, fontsize=20)
    ax.tick_params(labelsize=20)
    # plt.savefig(f"relation_{y_label}.pdf", dpi=300)
    plt.show()

plot_density_flow(df_err_cd.mean(axis=1), df_acc.mean(axis=1), r"$E(\bar N^{\rho_c}, m_r)$", "RMSE")
plot_density_flow(df_flow_sum.mean(axis=1), df_MC.mean(axis=1), "$\Sigma Q$", "MC")

# %%



