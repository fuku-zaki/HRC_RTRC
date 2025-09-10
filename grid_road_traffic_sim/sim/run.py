from road_traffic import GridTraffics
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import os
import time
import calculate_arrival_time
import pickle

"""
Grid Road Traffic Simulation (OVM-based)

Key inputs:
- Grid size (rows, cols)
- Vehicle density (cars per road)
- Random seed
- Log/export options

Usage:
    python run.py --density 1 --seed 42
"""

start_time = time.time()

###############  parameter  ###################
# Experiment Configuration
import sys
density = int(sys.argv[1])                                                      # traffic density (:= num. of cars / num. of road)
SEED = int(sys.argv[2])
print(f"* Density = {density}, SEED: {SEED}")

# Simulation Configuration
n_simulation = 50000                                                            # num.  of time steps

# Road Configuration
num_grid = (2,3)                                                                # num. of grids (rows, cols)
num_road = num_grid[0]*(num_grid[1]-1)*2 + (num_grid[0]-1)*num_grid[1]*2        # num. of roads (including both directions)
num_signals = num_grid[0]*num_grid[1]                                           # num. of intersections
L = 50                                                                          # length of each road
intersection_w = 4                                                              # width of intersections

# Traffic State Configuration
num_car = int(num_road*density)                                                 # num. of cars

# Traffic signal Configuration
base_tau=(50,0)                                                              
not_getout = True

# Others
dt = 0.5
ratio = 1                                                                       
obs_ratio = 1 - ratio                                                           
from_to = (0,3)                                                                 

# Setting for Optimal Velocity Model (OVM)
"""V(h)=V0[tanh(h-bf)-tanh(bc-bf)]"""
a_range = [0.8,1.2]
bf_max = 0.5
bf_range = [0.2, bf_max]
bc_range = [0.2, 0.5]
v_ctrl = 1.0

# Logging / Output Options
interval = 50
export_csv = True
animation_plot = False
movie_saving = False
signal_anime = False
animation_create = (animation_plot or movie_saving)
arrival_hist = False
###############  parameter  ###################


###############  simulation  ###################
GT = GridTraffics(num_grid=num_grid, 
                  num_car=num_car, 
                  car_speed_params=(a_range,bf_range,bc_range), 
                  dt=dt,
                  base_tau=base_tau,
                  v_ctrl=v_ctrl,
                  not_getout=not_getout,
                  L=L,
                  intersection_w=intersection_w,
                  obs_ratio=obs_ratio, 
                  SEED=SEED, 
                  export_csv=export_csv)

GT.simulate(n_simulation)
###############  simulation  ###################

###############  get information  ###################
x_list = GT._get_car_info('x')
y_list = GT._get_car_info('y')
cars_dict = GT.cars_dict
active = GT._get_car_info('active')
sig_xy = GT._get_signal_info('xy')
sig_judge = GT._get_signal_info('judge')
sig_theta = GT._get_signal_info('theta')
signals_info = GT.signals_dict

csv_start_time = time.time()
# Generate a CSV file
if export_csv:
    car_info = GT.car_info_in_road
    danmen_flow = GT.danmen_flow
    print('*'*50)
    print('Start export csv')
    df_road_car_num = pd.DataFrame()
    df_road_car_ave_v = pd.DataFrame()
    df_road_car_ave_v_ratio = pd.DataFrame()
    df_road_car_num['step'] = range(n_simulation)
    df_road_car_ave_v['step'] = range(n_simulation)
    df_road_car_ave_v_ratio['step'] = range(n_simulation)

    for road in car_info.keys():
        df_road_car_num[road] = car_info[road]['car_num']
        df_road_car_ave_v[road] = car_info[road]['car_ave_v']
        df_road_car_ave_v_ratio[road] = car_info[road]['car_ave_v_ratio']

    df_signal_theta = pd.DataFrame()        # internal states
    df_signal_inflow = pd.DataFrame()       # information on inflow roads
    df_signal_input = pd.DataFrame()        # traffic inflow at the intersection
    df_signal_input_w = pd.DataFrame()      # traffic inflow at the intersection (west)
    df_signal_input_e = pd.DataFrame()      # traffic inflow at the intersection (east)
    df_signal_input_s = pd.DataFrame()      # traffic inflow at the intersection (south)
    df_signal_input_n = pd.DataFrame()      # traffic inflow at the intersection (north)
    df_danmen_flow = pd.DataFrame()         # sectional traffic volume
    
    df_signal_theta['step'] = range(n_simulation)
    df_signal_inflow['step'] = range(n_simulation)
    df_signal_input['step'] = range(n_simulation)
    df_signal_input_w['step'] = range(n_simulation)
    df_signal_input_e['step'] = range(n_simulation)
    df_signal_input_s['step'] = range(n_simulation)
    df_signal_input_n['step'] = range(n_simulation)
    df_danmen_flow['step'] = range(n_simulation)
    for signal_idx in signals_info.keys():
        df_signal_theta[str(signal_idx)] = signals_info[signal_idx]['theta']
        df_signal_inflow[str(signal_idx)+'numUD'] = signals_info[signal_idx]['UD_inflow_num']
        df_signal_inflow[str(signal_idx)+'ave_vUD']  = signals_info[signal_idx]['UD_inflow_ave_v']
        df_signal_inflow[str(signal_idx)+'numRL'] = signals_info[signal_idx]['RL_inflow_num']
        df_signal_inflow[str(signal_idx)+'ave_vRL']  = signals_info[signal_idx]['RL_inflow_ave_v']

        df_signal_input[str(signal_idx)] = signals_info[signal_idx]['all_in']
        df_signal_input_w[str(signal_idx)+'w'] = signals_info[signal_idx]['left_in']
        df_signal_input_e[str(signal_idx)+'e'] = signals_info[signal_idx]['right_in']
        df_signal_input_s[str(signal_idx)+'s'] = signals_info[signal_idx]['up_in']
        df_signal_input_n[str(signal_idx)+'n'] = signals_info[signal_idx]['down_in']
        df_signal_input = pd.merge(df_signal_input_e,df_signal_input_w)
        df_signal_input = pd.merge(df_signal_input,df_signal_input_n)
        df_signal_input = pd.merge(df_signal_input,df_signal_input_s)

    for road in danmen_flow.keys():
        df_danmen_flow[road] = danmen_flow[road]
    # save
    root_path = f'./data'
    folder_name = str(num_grid[0])+'x'+str(num_grid[1])+"_L"+str(L)+'_N'+str(num_car)+'_s'+str(n_simulation)+'_tau'+str(base_tau[0])+'_'+str(base_tau[1])+'_Vctrl'+str(v_ctrl)
    seed_folder = f'SEED{SEED}'
    folder_path = os.path.join(root_path, folder_name, seed_folder)
    if not os.path.exists(folder_path): 
        os.makedirs(folder_path)
    df_road_car_num.to_csv(folder_path+'/road_car_num.csv',index=False)
    df_road_car_ave_v.to_csv(folder_path+'/road_car_ave_v.csv',index=False)
    df_road_car_ave_v_ratio.to_csv(folder_path+'/road_car_ave_v_ratio'+str(obs_ratio)+'.csv',index=False)
    df_signal_theta.to_csv(folder_path+'/signal_theta.csv',index=False)
    df_signal_inflow.to_csv(folder_path+'/signal_inflow.csv',index=False)
    df_signal_input.to_csv(folder_path+'/signal_input.csv',index=False)
    df_danmen_flow.to_csv(folder_path+'/danmen_flow.csv',index=False)

    print('Finish export csv')
    print('Time of Exporting csv : ', time.time() - csv_start_time)

if arrival_hist:
    def pickle_load(path):
        with open(path, mode='rb') as f:
            data = pickle.load(f)
            return data
    def flatten(l):
        ret = []
        for a in l:
            if hasattr(a,'__iter__'):
                ret += flatten(a)
            else:
                ret.append(a)
        return ret

    file_name = str(num_grid[0])+'x'+str(num_grid[1])
    ArrivalTime_dict = pickle_load(folder_path+ '/'+ file_name +'_ArrivalTime.pickle')
    df_arrival = pd.DataFrame.from_dict(ArrivalTime_dict)
    data_arrival = flatten(df_arrival['start'+str(from_to[0])+'_goal'+str(from_to[1])].tolist())
    print('travel time of median is '+str(np.median(data_arrival)))
    print('travel time of min is '+str(np.amin(data_arrival)))
    print('travel time of max is '+str(np.amax(data_arrival)))

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1,1,1)
    ax2.hist(data_arrival)
    ax2.set_title('start'+str(from_to[0])+'_goal'+str(from_to[1]))
    ax2.set_xlabel('travel time')
    ax2.set_ylabel('freq')
    # plt.show()
    folder_path = '../fig'
    plt.savefig(folder_path+'/'+'start'+str(from_to[0])+'_goal'+str(from_to[1])+'.png')
    # plt.savefig("figure.png")
###############  get information  ###################

###############  generate animation  ###################
if animation_create:
    ani_start_time = time.time()
    print('*'*50)
    print('Start create animation')
    xmin = 0; xmax = GT.W; ymin = 0; ymax = GT.H
    fig = plt.figure(figsize=(num_grid[1]*3,num_grid[0]*3))
    ax1 = plt.subplot2grid((1,1), (0,0))
    ax1.set_xlim(0, GT.W)
    ax1.set_ylim(0, GT.H)
    ax1.tick_params(labelbottom=False,bottom=False,
                    labelleft=False,left=False,
                    labelright=False,right=False,
                    labeltop=False,top=False)

    # road plotting
    ax1.axvline(0, color='black',zorder=3); ax1.axvline(GT.W, color='black',zorder=3)
    ax1.axhline(0, color='black',zorder=3); ax1.axhline(GT.H, color='black',zorder=3)
    for h in range(num_grid[0]):
        ax1.axhline([h*GT.intersection_w + h*GT.L],lw=1,c='black',zorder=1)
        ax1.axhline([(h+1)*GT.intersection_w + h*GT.L],lw=1,c='black',zorder=1)
        ax1.axhline([(h+1)*GT.intersection_w + h*GT.L - GT.intersection_w/2],alpha=0.5, lw=0.8,linestyle='dashed', color='grey',zorder=1) # 中央線
    for w in range(num_grid[1]):
        ax1.axvline([w*GT.intersection_w + w*GT.L],lw=1,c='black',zorder=1)
        ax1.axvline([(w+1)*GT.intersection_w + w*GT.L],lw=1,c='black',zorder=1)
        ax1.axvline([(w+1)*GT.intersection_w + w*GT.L - GT.intersection_w/2],alpha=0.5, lw=0.8,linestyle='dashed',color='grey',zorder=1) # 中央線
    # traffic signal plotting
    ax1.scatter(sig_xy[:,0], sig_xy[:,1], color='tomato',zorder=2, s=15, marker='D')
    for signal_idx in range(num_signals):
        x = sig_xy[signal_idx,0]
        y = sig_xy[signal_idx,1]
        ax1.text(x-0.3, y-0.3, str(signal_idx),color='black', zorder=2.5)
    
    all_ims = []
    for s in range(n_simulation):
        X = x_list[:,s]
        Y = y_list[:,s]
        ims = ()
        im = ax1.scatter(X, Y, c='blue', s=5)
        im_1 = ax1.text(GT.W/2-5, GT.H+0.5 , f'Step={s}')
        ims = ims + (im, im_1)
        if signal_anime:
            c = ['red','lime']
            for sig_id in range(num_signals):
                sig_v1 = ax1.vlines(x=sig_xy[sig_id,0]+GT.intersection_w/2,ymin=sig_xy[sig_id,1]-GT.intersection_w/2,ymax=sig_xy[sig_id,1]+GT.intersection_w/2,colors=c[int(sig_judge[sig_id,s])],lw=1.5,zorder=2) # 左右
                sig_v2 = ax1.vlines(x=sig_xy[sig_id,0]-GT.intersection_w/2,ymin=sig_xy[sig_id,1]-GT.intersection_w/2,ymax=sig_xy[sig_id,1]+GT.intersection_w/2,colors=c[int(sig_judge[sig_id,s])],lw=1.5,zorder=2) # 左右
                sig_h1 = ax1.hlines(y=sig_xy[sig_id,1]+GT.intersection_w/2,xmin=sig_xy[sig_id,0]-GT.intersection_w/2,xmax=sig_xy[sig_id,0]+GT.intersection_w/2,colors=c[int(not(sig_judge[sig_id,s]))],lw=1.5,zorder=2) # 上下
                sig_h2 = ax1.hlines(y=sig_xy[sig_id,1]-GT.intersection_w/2,xmin=sig_xy[sig_id,0]-GT.intersection_w/2,xmax=sig_xy[sig_id,0]+GT.intersection_w/2,colors=c[int(not(sig_judge[sig_id,s]))],lw=1.5,zorder=2) # 上下
                ims = ims + (sig_v1, sig_v2, sig_h1, sig_h2)
        all_ims.append((ims))

    ani = animation.ArtistAnimation(fig, all_ims, interval=interval, repeat=True)
    ani_finish_time = time.time()
    print('Finish create animation')
    print('Time of creating animation : ', ani_finish_time - ani_start_time)

    if animation_plot:
        plt.show()

    if movie_saving:
        movie_save_start_time = time.time()
        print('*'*50)
        print('Start save movie')
        folder_path = f'./movie'
        if not os.path.exists(folder_path): 
            os.makedirs(folder_path)
        ani.save(folder_path+'/'+str(num_grid[0])+'x'+str(num_grid[1])+"_L"+str(L)+'_N'+str(num_car)+'_s'+str(n_simulation)+'_tau'+str(base_tau[0])+'_'+str(base_tau[1])+'_Vctrl'+str(v_ctrl)+'.mp4',
                 writer="ffmpeg", fps=(1/dt)*10)
        print('Animation saving!')
        print('Time of saving movie : ', time.time()- movie_save_start_time)
###############  generate animation  ###################

print('*'*50)
print('Full Time : ', time.time() - start_time)
