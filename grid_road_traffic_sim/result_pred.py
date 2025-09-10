# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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
def plot_seed(seed_num):
    folder_name = f"pred_result_SEED{seed_num}"
    acc_test_list = []
    for density in density_list:
        df = pd.read_csv(folder_name + f"/acc_density{density}.csv")
        acc_test_list.append(df["test"])
    MC_list_all = pd.read_csv(folder_name + f"/MC.csv").values.tolist()[0]
    return acc_test_list, MC_list_all

# %%
seed_list = np.arange(1,21)
density_list = np.arange(1,16,1)
df_acc = pd.DataFrame(index=density_list)
df_MC = pd.DataFrame(index=density_list)

for seed_num in seed_list:
    acc_test_list, MC_list_all = plot_seed(seed_num)
    df_acc[seed_num] = np.mean(np.array(acc_test_list).T, axis=0)
    df_MC[seed_num] = np.array(MC_list_all)

# %%
import os
output = True
if output:
    output_path = "result_all"
    os.makedirs(output_path, exist_ok=True)
    df_acc.to_csv(output_path+f"/df_acc.csv", index=False)
    df_MC.to_csv(output_path+f"/df_MC.csv", index=False)

# %%
medianprops={'color': 'C1', 'linewidth':3, 'linestyle': '-.',}
meanprops={'markersize': 5, 'markeredgewidth': 2,}

plt.figure(figsize=(6.2,5), dpi=300, tight_layout=True)
plt.boxplot(df_acc.T, labels=density_list, showmeans=True, medianprops=medianprops, meanprops=meanprops)
plt.xlabel(r"$\bar N_c$", fontsize=20)
plt.ylabel("RMSE", fontsize=20)
plt.tick_params(labelsize=20)
# plt.savefig("RMSE_num.pdf", dpi=300)
plt.show()

plt.figure(figsize=(6.5,5), dpi=300, tight_layout=True)
plt.boxplot(df_MC.T, labels=density_list, showmeans=True, medianprops=medianprops, meanprops=meanprops)
plt.xlabel(r"$\bar N_c$", fontsize=20)
plt.ylabel("MC", fontsize=20)
plt.tick_params(labelsize=20)
# plt.savefig("MC_num.pdf", dpi=300)
plt.show()

# %%



