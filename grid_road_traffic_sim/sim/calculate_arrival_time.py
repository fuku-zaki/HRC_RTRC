import pandas as pd
import numpy as np
import os
import pickle


def In_intersection(car_posi, sig_id, signals_dict):
    '''
    Whether car is at the signal
    '''
    car_posi = car_posi.replace('(','').replace(')','')
    car_x = float(car_posi.split(',')[0])
    car_y = float(car_posi.split(',')[1])
    sig_x = signals_dict[str(sig_id)][0]
    sig_y = signals_dict[str(sig_id)][1]
    if abs(sig_x - car_x) <=2 and abs(sig_y - car_y)<=2:
        return 1
    else:
        return 0

def get_start_goal(df, start, goal, signals_dict,n_simulation):
    '''
    return DataFrame of Start time and Goal time
    '''
    df_start = pd.DataFrame()
    df_goal = pd.DataFrame()
    for col in df.columns[1:]:
        df_start[col] = df[col].apply(In_intersection,sig_id=start, signals_dict=signals_dict)
        df_goal[col] = df[col].apply(In_intersection,sig_id=goal, signals_dict=signals_dict)
    
    df_s = df_start.diff(-1)
    df_g = df_goal.diff(1)
    df_s.iloc[n_simulation-1,:] = df_start.iloc[n_simulation-1,:]
    df_g.iloc[0,:] = df_goal.iloc[0,:]
    
    return df_s, df_g

def get_arrival_time_dict(df_s, df_g):
    '''
    get arrival_time between any two points
    '''
    arrival_time_dict = {}
    for col in df_s.columns:
        start = list(df_s[df_s[col]>0].loc[:,col].index)
        goal = list(df_g[df_g[col]>0].loc[:,col].index)
        arrival_time = []
        if len(start)==0 or len(goal)==0:
            arrival_time_dict[col] = arrival_time
            continue
          
        for i in range(len(start)):
            for j in range(i,len(goal)):
                time = goal[j] - start[i]
                if  time <= 0:
                    continue
                else:
                    arrival_time.append(time)
                    break
        
        arrival_time_dict[col] = arrival_time 

    return arrival_time_dict

def pickle_dump(obj, path):
    with open(path, mode='wb') as f:
        pickle.dump(obj,f)

    

def main(L, intersection_w, num_grid, folder_path, file_name,n_simulation):

    h = num_grid[0]
    w = num_grid[1]
    num_signals = h*w

    df = pd.read_csv(folder_path+'/car_position.csv')

    signals_dict = {}
    for i in range(num_signals):
        signals_dict[str(i)] = (intersection_w/2+L*(i%w)+intersection_w*(i%w), intersection_w/2+L*int(i/w)+intersection_w*int(i/w))


    all_arrival_time = {}
    for i in range(num_signals):
        for j in range(num_signals):
            df_s, df_g = get_start_goal(df, start=i,goal=j, signals_dict=signals_dict,n_simulation=n_simulation)
            arrival_time_dict = get_arrival_time_dict(df_s, df_g)
            all_arrival_time['start'+str(i)+'_goal'+str(j)] = arrival_time_dict

    pickle_dump(all_arrival_time, folder_path+'/'+file_name+'_ArrivalTime.pickle')
