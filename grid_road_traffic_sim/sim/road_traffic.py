import numpy as np
import math
import random
from tqdm import tqdm
from scipy.integrate import solve_ivp

class GridTraffics():
    """
    Grid Traffic Class

    Notes:
        This class is organized into the following sections:
        - Initial setting
        - Equations
        - Discrimination
        - Update
        - Information
    """
    def __init__(self, 
                 num_grid, 
                 num_car, 
                 car_speed_params=([0.8,1.2],[0.2,0.7],[0.2,0.5]), 
                 dt=0.5, 
                 base_tau=(5, 1), 
                 base_xi=(0, 0.01), 
                 v_ctrl=1,
                 not_getout=True, 
                 L=10, 
                 intersection_w=4, 
                 obs_ratio=0, 
                 SEED=1234,
                 export_csv=False):
        """
        Initialize the Grid Traffic simulation.

        Args:
            num_grid (tuple[int, int]):         Grid dimensions as (height, width).
            num_car (int):                      Initial number of vehicles.
            car_speed_params (tuple[list]):     Parameters for the optimal velocity model.
            dt (float):                         Sampling time (time step).
            base_tau (tuple, optional):         Signal cycle (mean, standard deviation).
            base_xi (tuple, optional):          Initial signal frequency (mean, standard deviation).
            v_ctrl (float, optional):           Velocity scaling factor.
            not_getout (bool, optional):        If True, vehicles do not exit the grid.
            L (float, optional):                Road length.
            intersection_w (float, optional):   Intersection width.
            obs_ratio (float, optional):        Observation ratio.
            SEED (int, optional):               Random seed for reproducibility.
            export_csv (bool, optional):        Whether to export results to CSV.
        """
        # road information
        self.num_grid = num_grid
        self.num_car = num_car
        self.L = L                                                  # length of each road
        self.intersection_w = intersection_w                        # width of intersection
        self.h = self.num_grid[0]
        self.w = self.num_grid[1]
        self.num_signals = self.h * self.w                          # num of signals
        self.W = (self.w-1)*(self.L) + self.w*self.intersection_w   # width (Total horizontal road length)
        self.H = (self.h-1)*(self.L) + self.h*self.intersection_w   # height (Total vertical road length)
        # car information
        self.a_range = car_speed_params[0]
        self.bf_range = car_speed_params[1]
        self.bc_range = car_speed_params[2]
        self.dt = dt
        self.v_ctrl = v_ctrl
        # signals
        self.base_tau = base_tau
        self.base_xi = base_xi
        # others
        self.not_get_out = not_getout 
        self.max_dist = L                                           # max distance of cars
        self.min_dist = L/20                                        # min distance of cars
        self.min_dist = 1                                           # min distance of cars
        self.min_dist = 1                                           # min distance of cars
        self.now_step = 0
        self.obs_ratio = obs_ratio
        self.SEED = SEED
        self.export_csv = export_csv
        
        # external input
        self.random_step = self.get_random_step()

        # set the initial value of signals and cars
        self._generate_initial_setting()
        
    def simulate(self, n_simulation):
        '''
        simulate
        '''
        import time
        start_time = time.time()
        for s in tqdm(range(1,n_simulation)):
            self._get_direction_car_id_dict()
            self.now_step = s
            self._update_signals()
            self._update()
            self._judge_inout()
            self._update_road_data()
            if s%100==0:print("STEP: ", s);print()
        print("FINSHED")
        print("Time of Simulate : ", time.time() - start_time)

    #################################  initial setting  ######################################
    def _generate_initial_setting(self):
        '''
        Set the initial value of agents
        '''
        self._generate_signals()
        self._generate_cars()
        self._get_direction_car_id_dict()
        self._generate_road_data()
        self._generate_signal_inflow()
        self._generate_danmen_flow()

    def _generate_signals(self):
        '''
        Set the initial value of the signal
        '''
        self.signals_dict = {}

        # signal params
        np.random.seed(self.SEED)
        # base_tau[0]:mean, base_tau[1]:std.
        tau_list = np.random.normal(self.base_tau[0], self.base_tau[1], self.num_signals)
        # initial_signal_frequency
        np.random.seed(self.SEED)
        xi_list = np.random.normal(self.base_xi[0], self.base_xi[1], self.num_signals)
        xi_list = np.zeros(self.num_signals)
        
        if self.not_get_out:
            not_up_list = list(range(self.num_signals-self.w, self.num_signals+1))
            not_down_list = list(range(self.w))
            not_left_list = list(range(0, self.num_signals-self.w+1, self.w))
            not_right_list = list(range(self.w-1, self.num_signals, self.w))

        signal_dict = {}
        road_danmen_dict = {}
        # Store signal information in dictionary
        for i in range(self.num_signals):
            signal_dict[i] = {}
            # Signal placed at the center of the intersection
            signal_dict[i]['xy'] = (self.intersection_w/2+self.L*(i%self.w)+self.intersection_w*(i%self.w), self.intersection_w/2+self.L*int(i/self.w)+self.intersection_w*int(i/self.w))
            signal_dict[i]['tau'] = tau_list[i]
            signal_dict[i]['xi'] = xi_list[i]
            signal_dict[i]['theta'] = [xi_list[i]]
            signal_dict[i]['InCar'] = 0
        
            if np.mod(signal_dict[i]['theta'], 2*np.pi) < np.pi:
                signal_dict[i]['judge'] = [False]
            else:
                signal_dict[i]['judge'] = [True]
            signal_dict[i]['directions'] = set(['up', 'down', 'left', 'right'])
            if self.not_get_out:
                if i in not_up_list:
                    signal_dict[i]['directions'] = signal_dict[i]['directions'] - set(['up'])
                if i in not_down_list:
                    signal_dict[i]['directions'] = signal_dict[i]['directions'] - set(['down'])
                if i in not_left_list:
                    signal_dict[i]['directions'] = signal_dict[i]['directions'] - set(['left'])
                if i in not_right_list:
                    signal_dict[i]['directions'] = signal_dict[i]['directions'] - set(['right'])
            
            for direction in ['up', 'down', 'left', 'right']:
                signal_dict[i][direction+'_in'] = [0]
            signal_dict[i]['all_in'] = [0]

            ## Probability of turning
            # random
            signal_dict[i]['trans_p'] = np.random.rand(len(signal_dict[i]['directions']))
            signal_dict[i]['trans_p'] = signal_dict[i]['trans_p'] / np.sum(signal_dict[i]['trans_p'])
            
            # Around signal index
            if i < self.w*(self.h-1):
                signal_dict[i]['up_sig_idx'] = i + self.w
                road_danmen_dict[str(i)+'to'+str(i+self.w)] = [signal_dict[i]['xy'][0]-self.intersection_w/4,signal_dict[i]['xy'][1]+self.intersection_w/2+self.L/2]
            else:
                signal_dict[i]['up_sig_idx'] = None
            if i >= self.w:
                signal_dict[i]['down_sig_idx'] = i - self.w
                road_danmen_dict[str(i)+'to'+str(i-self.w)] = [signal_dict[i]['xy'][0]+self.intersection_w/4,signal_dict[i]['xy'][1]-self.intersection_w/2-self.L/2]
            else:
                signal_dict[i]['down_sig_idx'] = None
            if i%self.w != 0:
                signal_dict[i]['left_sig_idx'] = i - 1
                road_danmen_dict[str(i)+'to'+str(i-1)] = [signal_dict[i]['xy'][0]-self.intersection_w/2-self.L/2,signal_dict[i]['xy'][1]-self.intersection_w/4]
            else:
                signal_dict[i]['left_sig_idx'] = None
            if (i+1)%self.w != 0:
                signal_dict[i]['right_sig_idx'] = i + 1
                road_danmen_dict[str(i)+'to'+str(i+1)] = [signal_dict[i]['xy'][0]+self.intersection_w/2+self.L/2,signal_dict[i]['xy'][1]+self.intersection_w/4]
            else:
                signal_dict[i]['right_sig_idx'] = None

        self.signals_dict = signal_dict
        self.road_danmen_dict = road_danmen_dict

    def _generate_cars(self):
        '''
        Set the initial value of the car
        '''
        self.cars_dict = {}

        N = int(self.num_car/4)
        add = int(self.num_car%4)

        ## Initial Cordinate, direction
        ## Place randomly
        # up
        # left
        # down
        # right
        np.random.seed(self.SEED)
        initX = np.hstack([np.random.choice([self.L*i+self.intersection_w*i + self.intersection_w/4 for i in range(self.w)],N),
                           np.linspace(self.intersection_w, self.W-self.intersection_w, N+add), 
                           np.random.choice([self.L*i+self.intersection_w*i + 3*self.intersection_w/4 for i in range(self.w)],N), 
                           np.linspace(self.intersection_w, self.W-self.intersection_w, N)])
        initY = np.hstack([np.linspace(self.intersection_w, self.H-self.intersection_w, N), 
                          np.random.choice([self.L*i+self.intersection_w*i + self.intersection_w/4 for i in range(self.h)],N+add), 
                          np.linspace(self.intersection_w, self.H-self.intersection_w, N), 
                          np.random.choice([self.L*i+self.intersection_w*i + 3*self.intersection_w/4 for i in range(self.h)],N)])

        initD = np.hstack([['up']*N,['left']*(N+add), ['down']*N, ['right']*N])

        car_dict = {}
        a, bf, bc = self._generate_car_speed_params()
        for i in range(self.num_car):
            car_dict[i] = {}
            car_dict[i]['x'] = [initX[i]]
            car_dict[i]['y'] = [initY[i]]
            car_dict[i]['xy'] = [(float(initX[i]),float(initY[i]))]
            car_dict[i]['d'] = [initD[i]]
            car_dict[i]['next_d'] = initD[i]
            car_dict[i]['v'] = [0]
            car_dict[i]['a'] = float(a)
            car_dict[i]['bf'] = float(bf)
            car_dict[i]['bc'] = float(bc)
            car_dict[i]['active'] = True
            car_dict[i]['dot'] = np.zeros(3)
            self.cars_dict = car_dict
            self._get_nearest_signal(i)

    def _generate_road_data(self):
        '''
        Create a dictionary to record the number of vehicles and speed betwee signals
        '''
        if self.export_csv:
            self.car_info_in_road = {}
            for signal_id in range(self.num_signals):
                sig_xy = self.signals_dict[signal_id]['xy']
                up_sig_idx = self.signals_dict[signal_id]['up_sig_idx']
                down_sig_idx = self.signals_dict[signal_id]['down_sig_idx']
                left_sig_idx = self.signals_dict[signal_id]['left_sig_idx']
                right_sig_idx = self.signals_dict[signal_id]['right_sig_idx']
                inflow_sum, UD_inflow_sum, RL_inflow_sum = 0, 0, 0
                inflow_v_sum, UD_inflow_v_sum, RL_inflow_v_sum = 0, 0, 0
                for direction in ['up','down','right','left']:
                    car_id_list = self.direction_car_id_dict[direction]
                    if direction == 'down':
                        if up_sig_idx != None:
                            x_ = sig_xy[0] + self.intersection_w/4
                            y_under = sig_xy[1] + self.intersection_w/2
                            y_upper = self.signals_dict[up_sig_idx]['xy'][1] - self.intersection_w/2
                            y_upper_ratio = self.signals_dict[up_sig_idx]['xy'][1] - self.intersection_w/2 - self.L*self.obs_ratio
                            in_car_list = [carid for carid in car_id_list if (x_==self.cars_dict[carid]['x'][-1]) and (self.cars_dict[carid]['y'][-1]>=y_under) and (self.cars_dict[carid]['y'][-1]<=y_upper)]
                            in_car_list_ratio = [carid for carid in car_id_list if (x_==self.cars_dict[carid]['x'][-1]) and (self.cars_dict[carid]['y'][-1]>=y_under) and (self.cars_dict[carid]['y'][-1]<=y_upper_ratio)]
                            self.car_info_in_road[str(up_sig_idx)+'to'+str(signal_id)] = {}
                            self.car_info_in_road[str(up_sig_idx)+'to'+str(signal_id)]['car_num'] = [len(in_car_list)]
                            self.car_info_in_road[str(up_sig_idx)+'to'+str(signal_id)]['car_ave_v'] = [np.mean([self.cars_dict[carid]['v'][-1] for carid in in_car_list])]
                            self.car_info_in_road[str(up_sig_idx)+'to'+str(signal_id)]['car_num_ratio'] = [len(in_car_list_ratio)]
                            self.car_info_in_road[str(up_sig_idx)+'to'+str(signal_id)]['car_ave_v_ratio'] = [np.mean([self.cars_dict[carid]['v'][-1] for carid in in_car_list_ratio])]
                            inflow_sum += len(in_car_list); UD_inflow_sum += len(in_car_list); UD_inflow_v_sum += len(in_car_list)
                            inflow_v_sum += np.sum([self.cars_dict[carid]['v'][-1] for carid in in_car_list])
                    elif direction == 'up':
                        if down_sig_idx != None:
                            x_ = sig_xy[0] - self.intersection_w/4
                            y_under = self.signals_dict[down_sig_idx]['xy'][1] + self.intersection_w/2
                            y_under_ratio = self.signals_dict[down_sig_idx]['xy'][1] + self.intersection_w/2 + self.L*self.obs_ratio
                            y_upper = sig_xy[1] - self.intersection_w/2
                            in_car_list = [carid for carid in car_id_list if (x_==self.cars_dict[carid]['x'][-1]) and (self.cars_dict[carid]['y'][-1]>=y_under) and (self.cars_dict[carid]['y'][-1]<=y_upper)]
                            in_car_list_ratio = [carid for carid in car_id_list if (x_==self.cars_dict[carid]['x'][-1]) and (self.cars_dict[carid]['y'][-1]>=y_under_ratio) and (self.cars_dict[carid]['y'][-1]<=y_upper)]
                            self.car_info_in_road[str(down_sig_idx)+'to'+str(signal_id)] = {}
                            self.car_info_in_road[str(down_sig_idx)+'to'+str(signal_id)]['car_num'] = [len(in_car_list)]
                            self.car_info_in_road[str(down_sig_idx)+'to'+str(signal_id)]['car_ave_v'] = [np.mean([self.cars_dict[carid]['v'][-1] for carid in in_car_list])]
                            self.car_info_in_road[str(down_sig_idx)+'to'+str(signal_id)]['car_num_ratio'] = [len(in_car_list_ratio)]
                            self.car_info_in_road[str(down_sig_idx)+'to'+str(signal_id)]['car_ave_v_ratio'] = [np.mean([self.cars_dict[carid]['v'][-1] for carid in in_car_list_ratio])]
                            inflow_sum += len(in_car_list); UD_inflow_sum += len(in_car_list); UD_inflow_v_sum += len(in_car_list)
                            inflow_v_sum += np.sum([self.cars_dict[carid]['v'][-1] for carid in in_car_list])
                    elif direction == 'right':
                        if left_sig_idx != None:
                            y_ = sig_xy[1] + self.intersection_w/4
                            x_under = self.signals_dict[left_sig_idx]['xy'][0] + self.intersection_w/2
                            x_under_ratio = self.signals_dict[left_sig_idx]['xy'][0] + self.intersection_w/2 + self.L*self.obs_ratio
                            x_upper = sig_xy[0] - self.intersection_w/2
                            in_car_list = [carid for carid in car_id_list if (y_==self.cars_dict[carid]['y'][-1]) and (self.cars_dict[carid]['x'][-1]>=x_under) and (self.cars_dict[carid]['x'][-1]<=x_upper)]
                            in_car_list_ratio = [carid for carid in car_id_list if (y_==self.cars_dict[carid]['y'][-1]) and (self.cars_dict[carid]['x'][-1]>=x_under_ratio) and (self.cars_dict[carid]['x'][-1]<=x_upper)]
                            self.car_info_in_road[str(left_sig_idx)+'to'+str(signal_id)] = {}
                            self.car_info_in_road[str(left_sig_idx)+'to'+str(signal_id)]['car_num'] = [len(in_car_list)]
                            self.car_info_in_road[str(left_sig_idx)+'to'+str(signal_id)]['car_ave_v'] = [np.mean([self.cars_dict[carid]['v'][-1] for carid in in_car_list])]
                            self.car_info_in_road[str(left_sig_idx)+'to'+str(signal_id)]['car_num_ratio'] = [len(in_car_list_ratio)]
                            self.car_info_in_road[str(left_sig_idx)+'to'+str(signal_id)]['car_ave_v_ratio'] = [np.mean([self.cars_dict[carid]['v'][-1] for carid in in_car_list_ratio])]
                            inflow_sum += len(in_car_list); RL_inflow_sum += len(in_car_list); RL_inflow_v_sum += len(in_car_list)
                            inflow_v_sum += np.sum([self.cars_dict[carid]['v'][-1] for carid in in_car_list])
                    elif direction == 'left':
                        if right_sig_idx != None:
                            y_ = sig_xy[1] - self.intersection_w/4
                            x_under = sig_xy[0] + self.intersection_w/2
                            x_upper = self.signals_dict[right_sig_idx]['xy'][0] - self.intersection_w/2
                            x_upper_ratio = self.signals_dict[right_sig_idx]['xy'][0] - self.intersection_w/2 - self.L*self.obs_ratio
                            in_car_list = [carid for carid in car_id_list if (y_==self.cars_dict[carid]['y'][-1]) and (self.cars_dict[carid]['x'][-1]>=x_under) and (self.cars_dict[carid]['x'][-1]<=x_upper)]
                            in_car_list_ratio = [carid for carid in car_id_list if (y_==self.cars_dict[carid]['y'][-1]) and (self.cars_dict[carid]['x'][-1]>=x_under) and (self.cars_dict[carid]['x'][-1]<=x_upper_ratio)]
                            self.car_info_in_road[str(right_sig_idx)+'to'+str(signal_id)] = {}
                            self.car_info_in_road[str(right_sig_idx)+'to'+str(signal_id)]['car_num'] = [len(in_car_list)]
                            self.car_info_in_road[str(right_sig_idx)+'to'+str(signal_id)]['car_ave_v'] = [np.mean([self.cars_dict[carid]['v'][-1] for carid in in_car_list])]
                            self.car_info_in_road[str(right_sig_idx)+'to'+str(signal_id)]['car_num_ratio'] = [len(in_car_list_ratio)]
                            self.car_info_in_road[str(right_sig_idx)+'to'+str(signal_id)]['car_ave_v_ratio'] = [np.mean([self.cars_dict[carid]['v'][-1] for carid in in_car_list_ratio])]
                            inflow_sum += len(in_car_list); RL_inflow_sum += len(in_car_list); RL_inflow_v_sum += len(in_car_list)
                            inflow_v_sum += np.sum([self.cars_dict[carid]['v'][-1] for carid in in_car_list])

                self.signals_dict[signal_id]['inflow_ave_v'] = [0]
                self.signals_dict[signal_id]['UD_inflow_ave_v'] = [0]
                self.signals_dict[signal_id]['RL_inflow_ave_v'] = [0]
                self.signals_dict[signal_id]['inflow_num'] = [inflow_sum]
                self.signals_dict[signal_id]['UD_inflow_num'] = [UD_inflow_sum]
                self.signals_dict[signal_id]['RL_inflow_num'] = [RL_inflow_sum]
    
    def _generate_signal_inflow(self):
        '''
        Create dictionary to measure inflow of intersection
        '''
        self.signals_inflow = {}
        for signal_id in range(self.num_signals):
            self.signals_inflow[signal_id] = {}
            self.signals_inflow[signal_id]['up'] = [0]
            self.signals_inflow[signal_id]['left'] = [0]
            self.signals_inflow[signal_id]['right'] = [0]
            self.signals_inflow[signal_id]['down'] = [0]

    def _generate_danmen_flow(self):
        '''
        Create dictionary to measure cross-sectional traffic volume for each road
        '''
        self.danmen_flow = {}
        for road in self.road_danmen_dict.keys():
            self.danmen_flow[road] = [0]
    
    #################################  Initial setting  ######################################

    #################################  External input  ######################################
    def get_random_step(self):
        ts = 50000
        np.random.seed(42)
        signal = np.random.choice(range(0,10), 10000)
        signal = (signal<5).astype(int)

        np.random.seed(0)
        timerange = np.random.choice(range(5,11), 10000)*10
        y = np.empty(0)
        i = 0
        while len(y)<=ts:
            y = np.append(y, np.ones(timerange[i])*signal[i])
            i+=1

        return y[:ts]*np.pi*2
    #################################  External input  ######################################

    #################################  Equations  ######################################
    def _generate_car_speed_params(self):
        np.random.seed(self.SEED)
        # bf = np.random.uniform(self.min_dist, self.bf_range[1], size=1)    
        bf = 3
        a = 5
        bc = self.min_dist
        return a, bf, bc
    
    def get_V(self, x, bf, bc):
        '''
        Speed equation
        '''
        x = max(self.min_dist, x)
        return (np.tanh(x-bf) - np.tanh(bc-bf))*self.v_ctrl
    
    def In_intersection(self, car_id, sig_id):
        '''
        Whether car is at the signal
        '''
        car_x = self.cars_dict[car_id]['x'][self.now_step-1]
        car_y = self.cars_dict[car_id]['y'][self.now_step-1]
        sig_x = self.signals_dict[sig_id]['xy'][0]
        sig_y = self.signals_dict[sig_id]['xy'][1]
        if abs(sig_x - car_x) <2 and abs(sig_y - car_y)<2:
            return 1
        else:
            return 0
    
    def _update_signals(self):
        '''
        update signal state
        '''
        for signal_idx in range(self.num_signals):
            # update signal phase
            tau = self.signals_dict[signal_idx]['tau']
            xi = self.signals_dict[signal_idx]['xi']

            self.signals_dict[signal_idx]['theta'].append(self.random_step[self.now_step])
            signal_state = self.random_step[self.now_step]

            # update red-blue of traffic light
            if signal_state < np.pi:
                self.signals_dict[signal_idx]['judge'].append(False)
            else:
                self.signals_dict[signal_idx]['judge'].append(True)
          
            self.signals_dict[signal_idx]['InCar'] = 0
            for car_id in range(self.num_car):
                self.signals_dict[signal_idx]['InCar'] = self.signals_dict[signal_idx]['InCar'] + self.In_intersection(car_id, signal_idx)
               
    #################################  Equations  ######################################

    #################################  Discrimination  ######################################
    def _get_nearest_signal(self, car_id):
        '''
        Stere index of closest signal
        '''
        current_d = self.cars_dict[car_id]['d'][-1]
        next_d = self.cars_dict[car_id]['next_d']
        x = self.cars_dict[car_id]['x'][-1]
        y = self.cars_dict[car_id]['y'][-1]
        if current_d == 'up':
            selected_idx = np.asarray([k for k in self.signals_dict.keys() if "down" in self.signals_dict[k]["directions"]])
            selected_idx = np.asarray([k for k in selected_idx if x == self.signals_dict[k]["xy"][0]-self.intersection_w/4])
            signal_y = np.asarray([self.signals_dict[k]['xy'][1] for k in selected_idx])
            if next_d == 'right':
                is_up = signal_y - y >= -self.intersection_w/4
                diff = signal_y - y
                min_diff = np.min(diff[is_up])
                self.cars_dict[car_id]['nearest_signal_idx'] = int(selected_idx[diff==min_diff])
            else:
                is_up = signal_y - y >= 0
                diff = signal_y - y
                min_diff = np.min(diff[is_up])
                self.cars_dict[car_id]['nearest_signal_idx'] = int(selected_idx[diff==min_diff])

        elif current_d == 'down':
            selected_idx = np.asarray([k for k in self.signals_dict.keys() if "up" in self.signals_dict[k]["directions"]])
            selected_idx = np.asarray([k for k in selected_idx if x == self.signals_dict[k]["xy"][0]+self.intersection_w/4])
            signal_y = np.asarray([self.signals_dict[k]['xy'][1] for k in selected_idx])
            if next_d == 'left':
                is_down = signal_y - y <= self.intersection_w/4
                diff = signal_y - y
                min_diff = np.max(diff[is_down])
                self.cars_dict[car_id]['nearest_signal_idx'] = int(selected_idx[diff==min_diff])
            else:
                is_down = signal_y - y <= 0
                diff = signal_y - y
                min_diff = np.max(diff[is_down])
                self.cars_dict[car_id]['nearest_signal_idx'] = int(selected_idx[diff==min_diff])

        elif current_d =='left':
            selected_idx = np.asarray([k for k in self.signals_dict.keys() if "right" in self.signals_dict[k]["directions"]])
            selected_idx = np.asarray([k for k in selected_idx if y == self.signals_dict[k]["xy"][1]-self.intersection_w/4])
            signal_x = np.asarray([self.signals_dict[k]['xy'][0] for k in selected_idx])
            if next_d == 'up':
                is_left = signal_x - x <= self.intersection_w/4
                diff = signal_x - x
                min_diff = np.max(diff[is_left])
                self.cars_dict[car_id]['nearest_signal_idx'] = int(selected_idx[diff==min_diff])
            else:
                is_left = signal_x - x <= 0
                diff = signal_x - x
                min_diff = np.max(diff[is_left])
                self.cars_dict[car_id]['nearest_signal_idx'] = int(selected_idx[diff==min_diff])

        elif current_d == 'right':
            selected_idx = np.asarray([k for k in self.signals_dict.keys() if "left" in self.signals_dict[k]["directions"]])
            selected_idx = np.asarray([k for k in selected_idx if y == self.signals_dict[k]["xy"][1]+self.intersection_w/4])
            signal_x = np.asarray([self.signals_dict[k]['xy'][0] for k in selected_idx])
            if next_d == 'down':
                is_right = signal_x - x >= -self.intersection_w/4
                diff = signal_x - x
                min_diff = np.min(diff[is_right])
                self.cars_dict[car_id]['nearest_signal_idx'] = int(selected_idx[diff==min_diff])
            else:
                is_right = signal_x - x >= 0
                diff = signal_x - x
                min_diff = np.min(diff[is_right])
                self.cars_dict[car_id]['nearest_signal_idx'] = int(selected_idx[diff==min_diff])

    def _get_direction_car_id_dict(self):
        '''
        A collection of car index for each direction
        {'up':[0,1,2,3,4],'left':[5,6,7,8], ... }
        '''
        active_agent_idxs = [k for k in self.cars_dict.keys() if self.cars_dict[k]['active']]
        direction_id_dict = {}
        direction_id_dict['up'], direction_id_dict['down'], direction_id_dict['right'], direction_id_dict['left'] = [], [], [], []
        for k in active_agent_idxs:
            direction = self.cars_dict[k]['d'][-1]
            if direction not in direction_id_dict.keys():
                direction_id_dict[direction] = []
            direction_id_dict[direction].append(k)
        self.direction_car_id_dict = direction_id_dict

    def _get_ahead_diff(self, car_id):
        '''
        Returns the distance and the speed of the car ahead
        '''
        direction = self.cars_dict[car_id]['d'][self.now_step-1]
        x = self.cars_dict[car_id]['x'][self.now_step-1]
        y = self.cars_dict[car_id]['y'][self.now_step-1]
        car_id_list = self.direction_car_id_dict[direction]
        
        if direction == 'up':
            ys = np.asarray([self.cars_dict[carid]['y'][self.now_step-1] for carid in car_id_list if (carid != car_id) and (x==self.cars_dict[carid]['x'][self.now_step-1])])
            vs = np.asarray([self.cars_dict[carid]['v'][self.now_step-1] for carid in car_id_list if (carid != car_id) and (x==self.cars_dict[carid]['x'][self.now_step-1])])
            if len(ys)==0:
                return self.max_dist, None
            diff = ys - y
            is_up = diff > 0
            if len(diff[is_up]): # if there is a car in front
                min_idx = np.argmin(diff[is_up])
                if diff[is_up][min_idx]< self.max_dist:
                    return diff[is_up][min_idx], vs[is_up][min_idx]
                else:
                    return self.max_dist, None
            else:
                return self.max_dist, None

        elif direction == 'down':
            ys = np.asarray([self.cars_dict[carid]['y'][self.now_step-1] for carid in car_id_list if (carid != car_id) and (x==self.cars_dict[carid]['x'][self.now_step-1])])
            vs = np.asarray([self.cars_dict[carid]['v'][self.now_step-1] for carid in car_id_list if (carid != car_id) and (x==self.cars_dict[carid]['x'][self.now_step-1])])
            if len(ys)==0:
                return self.max_dist, None
            diff = y - ys
            is_down = diff > 0
            if len(diff[is_down]):
                min_idx = np.argmin(diff[is_down])
                if diff[is_down][min_idx] < self.max_dist:
                    return diff[is_down][min_idx], vs[is_down][min_idx]
                else:
                    return self.max_dist, None
            else:
                return self.max_dist, None

        elif direction == 'left':
            xs = np.asarray([self.cars_dict[carid]['x'][self.now_step-1] for carid in car_id_list if (carid != car_id) and (y==self.cars_dict[carid]['y'][self.now_step-1])])
            vs = np.asarray([self.cars_dict[carid]['v'][self.now_step-1] for carid in car_id_list if (carid != car_id) and (y==self.cars_dict[carid]['y'][self.now_step-1])])
            if len(xs)==0:
                return self.max_dist, None
            diff = x - xs
            is_left = diff > 0
            if len(diff[is_left]):
                min_idx = np.argmin(diff[is_left])
                if diff[is_left][min_idx] < self.max_dist:
                    return diff[is_left][min_idx], vs[is_left][min_idx]
                else:
                    return self.max_dist, None
            else:
                return self.max_dist, None
        
        elif direction == 'right':
            xs = np.asarray([self.cars_dict[carid]['x'][self.now_step-1] for carid in car_id_list if (carid != car_id) and (y==self.cars_dict[carid]['y'][self.now_step-1])])
            vs = np.asarray([self.cars_dict[carid]['v'][self.now_step-1] for carid in car_id_list if (carid != car_id) and (y==self.cars_dict[carid]['y'][self.now_step-1])])
            if len(xs)==0:
                return self.max_dist, None
            diff = xs - x
            is_right = diff >0
            if len(diff[is_right]):
                min_idx = np.argmin(diff[is_right])
                if diff[is_right][min_idx] < self.max_dist:
                    return diff[is_right][min_idx], vs[is_right][min_idx]
                else:
                    return self.max_dist, None
            else:
                return self.max_dist, None

    def _judge_inout(self):
        '''
        In the system or not
        '''
        active_agent_idxs = [k for k in self.cars_dict.keys() if self.cars_dict[k]['active']]
        for k in active_agent_idxs:
            x = self.cars_dict[k]['x'][self.now_step]
            y = self.cars_dict[k]['y'][self.now_step]
            if x < 0 or x > self.W:
                self.cars_dict[k]['active'] = False
            if y < 0 or y > self.H:
                self.cars_dict[k]['active'] = False

    #################################  Discrimination  ######################################

    #################################  Update  ######################################
    def _update(self):
        if self.export_csv:
            danmen_flow = {}
            signal_in = {}
            for signal_idx in range(self.num_signals):
                up_signal_idx = self.signals_dict[signal_idx]['up_sig_idx']
                down_signal_idx = self.signals_dict[signal_idx]['down_sig_idx']
                right_signal_idx = self.signals_dict[signal_idx]['right_sig_idx']
                left_signal_idx = self.signals_dict[signal_idx]['left_sig_idx']
                signal_in[signal_idx] = {}
                signal_in[signal_idx]['up_in'] = 0
                signal_in[signal_idx]['down_in'] = 0
                signal_in[signal_idx]['left_in'] = 0
                signal_in[signal_idx]['right_in'] = 0
                signal_in[signal_idx]['all_in'] = 0
                if up_signal_idx != None:
                    danmen_flow[str(signal_idx)+'to'+str(up_signal_idx)] = 0
                if down_signal_idx != None:
                    danmen_flow[str(signal_idx)+'to'+str(down_signal_idx)] = 0
                if right_signal_idx != None:
                    danmen_flow[str(signal_idx)+'to'+str(right_signal_idx)] = 0
                if left_signal_idx != None:
                    danmen_flow[str(signal_idx)+'to'+str(left_signal_idx)] = 0

        active_agent_idxs = [k for k in self.cars_dict.keys() if self.cars_dict[k]['active']]
        non_active_agent_idxs = [k for k in self.cars_dict.keys() if not(self.cars_dict[k]['active'])]

        for car_id in non_active_agent_idxs: # Out the system
            print('there is not active')
            x = self.cars_dict[car_id]['x'][self.now_step-1]
            y = self.cars_dict[car_id]['y'][self.now_step-1]
            v = self.cars_dict[car_id]['v'][self.now_step-1]
            d = self.cars_dict[car_id]['d'][self.now_step-1]
            self.cars_dict[car_id]['x'].append(x)
            self.cars_dict[car_id]['y'].append(y)
            self.cars_dict[car_id]['v'].append(v)
            self.cars_dict[car_id]['d'].append(d)


        for car_id in active_agent_idxs: # In the system
            x = self.cars_dict[car_id]['x'][self.now_step-1]
            y = self.cars_dict[car_id]['y'][self.now_step-1]
            v = self.cars_dict[car_id]['v'][self.now_step-1]
            # Runge-Kutta method
            k1 = self.dt * self._caluculate_dot(x, y, v, car_id)
            k2 = self.dt * self._caluculate_dot(x+k1[0]/2., y+k1[1]/2., v+k1[2]/2, car_id)
            k3 = self.dt * self._caluculate_dot(x+k2[0]/2., y+k2[1]/2., v+k2[2]/2, car_id)
            k4 = self.dt * self._caluculate_dot(x+k3[0], y+k3[1], v+k3[2], car_id)
            dx = (k1[0]+2.0*k2[0]+2.0*k3[0]+k4[0])/6
            dy = (k1[1]+2.0*k2[1]+2.0*k3[1]+k4[1])/6
            dv = (k1[2]+2.0*k2[2]+2.0*k3[2]+k4[2])/6
            self.cars_dict[car_id]['dot'] = [dx, dy, dv]
            

        #for car_id in active_agent_idxs:
            self._update_xyv(car_id)
            self.cars_dict[car_id]['xy'].append((float(self.cars_dict[car_id]['x'][self.now_step]),float(self.cars_dict[car_id]['y'][self.now_step])))
            # Closest signal
            try:
                self._get_nearest_signal(car_id)
            except:
                pass

            if self.export_csv:
                past_x = self.cars_dict[car_id]['x'][self.now_step-1]
                now_x = self.cars_dict[car_id]['x'][self.now_step]
                past_y = self.cars_dict[car_id]['y'][self.now_step-1]
                now_y = self.cars_dict[car_id]['y'][self.now_step]
                past_d = self.cars_dict[car_id]['d'][self.now_step-1]
                nearest_signal_idx = self.cars_dict[car_id]['nearest_signal_idx']
                sig_x = self.signals_dict[nearest_signal_idx]['xy'][0]
                sig_y = self.signals_dict[nearest_signal_idx]['xy'][1]
                    
                ## Measures the number of inflows at intersections
                if past_d == 'up':
                    border = sig_y - self.intersection_w/2
                    if past_y <= border and now_y > border:
                        signal_in[nearest_signal_idx]['down_in'] += 1
                if past_d == 'down':
                    border = sig_y + self.intersection_w/2
                    if past_y >= border and now_y<border:
                        signal_in[nearest_signal_idx]['up_in'] += 1
                if past_d == 'right':
                    border = sig_x - self.intersection_w/2
                    if past_x <= border and now_x>border:
                        signal_in[nearest_signal_idx]['left_in'] += 1
                if past_d == 'left':
                    border = sig_x + self.intersection_w/2
                    if past_x >= border and now_x<border:
                        signal_in[nearest_signal_idx]['right_in'] += 1

                ## Measure cross section traffic
                up_signal_idx = self.signals_dict[nearest_signal_idx]['up_sig_idx']
                down_signal_idx = self.signals_dict[nearest_signal_idx]['down_sig_idx']
                right_signal_idx = self.signals_dict[nearest_signal_idx]['right_sig_idx']
                left_signal_idx = self.signals_dict[nearest_signal_idx]['left_sig_idx']
                
                if past_d == 'up':
                    border = self.road_danmen_dict[str(down_signal_idx)+'to'+str(nearest_signal_idx)][1]
                    if past_y <= border and now_y > border:
                        danmen_flow[str(down_signal_idx)+'to'+str(nearest_signal_idx)] += 1
                if past_d == 'down':
                    border = self.road_danmen_dict[str(up_signal_idx)+'to'+str(nearest_signal_idx)][1]
                    if past_y >= border and now_y<border:
                        danmen_flow[str(up_signal_idx)+'to'+str(nearest_signal_idx)] += 1
                if past_d == 'right':
                    border = self.road_danmen_dict[str(left_signal_idx)+'to'+str(nearest_signal_idx)][0]
                    if past_x <= border and now_x>border:
                        danmen_flow[str(left_signal_idx)+'to'+str(nearest_signal_idx)] += 1
                if past_d == 'left':
                    border = self.road_danmen_dict[str(right_signal_idx)+'to'+str(nearest_signal_idx)][0]
                    if past_x >= border and now_x<border:
                        danmen_flow[str(right_signal_idx)+'to'+str(nearest_signal_idx)] += 1

        if self.export_csv:
            for signal_idx in range(self.num_signals):
                up_signal_idx = self.signals_dict[signal_idx]['up_sig_idx']
                down_signal_idx = self.signals_dict[signal_idx]['down_sig_idx']
                right_signal_idx = self.signals_dict[signal_idx]['right_sig_idx']
                left_signal_idx = self.signals_dict[signal_idx]['left_sig_idx']
                self.signals_dict[signal_idx]['up_in'].append(signal_in[signal_idx]['up_in'])
                self.signals_dict[signal_idx]['down_in'].append(signal_in[signal_idx]['down_in'])
                self.signals_dict[signal_idx]['left_in'].append(signal_in[signal_idx]['left_in'])
                self.signals_dict[signal_idx]['right_in'].append(signal_in[signal_idx]['right_in'])
                self.signals_dict[signal_idx]['all_in'].append(signal_in[signal_idx]['up_in']+signal_in[signal_idx]['down_in']+signal_in[signal_idx]['left_in']+signal_in[signal_idx]['right_in'])
                if up_signal_idx != None:
                    self.danmen_flow[str(signal_idx)+'to'+str(up_signal_idx)].append(danmen_flow[str(signal_idx)+'to'+str(up_signal_idx)])
                if down_signal_idx != None:
                    self.danmen_flow[str(signal_idx)+'to'+str(down_signal_idx)].append(danmen_flow[str(signal_idx)+'to'+str(down_signal_idx)])
                if right_signal_idx != None:
                    self.danmen_flow[str(signal_idx)+'to'+str(right_signal_idx)].append(danmen_flow[str(signal_idx)+'to'+str(right_signal_idx)])
                if left_signal_idx != None:
                    self.danmen_flow[str(signal_idx)+'to'+str(left_signal_idx)].append(danmen_flow[str(signal_idx)+'to'+str(left_signal_idx)])

    def _update_road_data(self):
        if self.export_csv:
            for signal_id in range(self.num_signals):
                sig_xy = self.signals_dict[signal_id]['xy']
                up_sig_idx = self.signals_dict[signal_id]['up_sig_idx']
                down_sig_idx = self.signals_dict[signal_id]['down_sig_idx']
                left_sig_idx = self.signals_dict[signal_id]['left_sig_idx']
                right_sig_idx = self.signals_dict[signal_id]['right_sig_idx']
                is_stop_UD = self.signals_dict[signal_id]['judge'][self.now_step]
                inflow_sum = 0
                UD_inflow_sum = 0
                RL_inflow_sum = 0
                inflow_v_sum = 0
                UD_inflow_v_sum = 0
                RL_inflow_v_sum = 0
                for direction in ['up','down','right','left']:
                    car_id_list = self.direction_car_id_dict[direction]
                    if direction == 'down':
                        if up_sig_idx != None:
                            x_ = sig_xy[0] + self.intersection_w/4
                            y_under = sig_xy[1] + self.intersection_w/2
                            y_upper = self.signals_dict[up_sig_idx]['xy'][1] - self.intersection_w/2 
                            y_upper_ratio = self.signals_dict[up_sig_idx]['xy'][1] - self.intersection_w/2 - self.L*self.obs_ratio
                            in_car_list = [carid for carid in car_id_list if (x_==self.cars_dict[carid]['x'][-1]) and (self.cars_dict[carid]['y'][-1]>=y_under) and (self.cars_dict[carid]['y'][-1]<=y_upper)]
                            in_car_list_ratio = [carid for carid in car_id_list if (x_==self.cars_dict[carid]['x'][-1]) and (self.cars_dict[carid]['y'][-1]>=y_under) and (self.cars_dict[carid]['y'][-1]<=y_upper_ratio)]
                            self.car_info_in_road[str(up_sig_idx)+'to'+str(signal_id)]['car_num'].append(len(in_car_list))
                            self.car_info_in_road[str(up_sig_idx)+'to'+str(signal_id)]['car_ave_v'].append(np.mean([self.cars_dict[carid]['v'][-1] for carid in in_car_list]))
                            self.car_info_in_road[str(up_sig_idx)+'to'+str(signal_id)]['car_num_ratio'].append(len(in_car_list_ratio))
                            self.car_info_in_road[str(up_sig_idx)+'to'+str(signal_id)]['car_ave_v_ratio'].append(np.mean([self.cars_dict[carid]['v'][-1] for carid in in_car_list_ratio]))
                            if not is_stop_UD:
                                inflow_sum += len(in_car_list)
                                UD_inflow_sum += len(in_car_list)
                                UD_inflow_v_sum += len(in_car_list)
                                inflow_v_sum += np.sum([self.cars_dict[carid]['v'][-1] for carid in in_car_list])
                    elif direction == 'up':
                        if down_sig_idx != None:
                            x_ = sig_xy[0] - self.intersection_w/4
                            y_under = self.signals_dict[down_sig_idx]['xy'][1] + self.intersection_w/2 
                            y_under_ratio = self.signals_dict[down_sig_idx]['xy'][1] + self.intersection_w/2 + self.L*self.obs_ratio
                            y_upper = sig_xy[1] - self.intersection_w/2
                            in_car_list = [carid for carid in car_id_list if (x_==self.cars_dict[carid]['x'][-1]) and (self.cars_dict[carid]['y'][-1]>=y_under) and (self.cars_dict[carid]['y'][-1]<=y_upper)]
                            in_car_list_ratio = [carid for carid in car_id_list if (x_==self.cars_dict[carid]['x'][-1]) and (self.cars_dict[carid]['y'][-1]>=y_under_ratio) and (self.cars_dict[carid]['y'][-1]<=y_upper)]
                            self.car_info_in_road[str(down_sig_idx)+'to'+str(signal_id)]['car_num'].append(len(in_car_list))
                            self.car_info_in_road[str(down_sig_idx)+'to'+str(signal_id)]['car_ave_v'].append(np.mean([self.cars_dict[carid]['v'][-1] for carid in in_car_list]))
                            self.car_info_in_road[str(down_sig_idx)+'to'+str(signal_id)]['car_num_ratio'].append(len(in_car_list_ratio))
                            self.car_info_in_road[str(down_sig_idx)+'to'+str(signal_id)]['car_ave_v_ratio'].append(np.mean([self.cars_dict[carid]['v'][-1] for carid in in_car_list_ratio]))
                            if not is_stop_UD:
                                inflow_sum += len(in_car_list)
                                UD_inflow_sum += len(in_car_list)
                                UD_inflow_v_sum += len(in_car_list)
                                inflow_v_sum += np.sum([self.cars_dict[carid]['v'][-1] for carid in in_car_list])
                    elif direction == 'right':
                        if left_sig_idx != None:
                            y_ = sig_xy[1] + self.intersection_w/4
                            x_under = self.signals_dict[left_sig_idx]['xy'][0] + self.intersection_w/2
                            x_under_ratio = self.signals_dict[left_sig_idx]['xy'][0] + self.intersection_w/2 + self.L*self.obs_ratio
                            x_upper = sig_xy[0] - self.intersection_w/2
                            in_car_list = [carid for carid in car_id_list if (y_==self.cars_dict[carid]['y'][-1]) and (self.cars_dict[carid]['x'][-1]>=x_under) and (self.cars_dict[carid]['x'][-1]<=x_upper)]
                            in_car_list_ratio = [carid for carid in car_id_list if (y_==self.cars_dict[carid]['y'][-1]) and (self.cars_dict[carid]['x'][-1]>=x_under_ratio) and (self.cars_dict[carid]['x'][-1]<=x_upper)]
                            self.car_info_in_road[str(left_sig_idx)+'to'+str(signal_id)]['car_num'].append(len(in_car_list))
                            self.car_info_in_road[str(left_sig_idx)+'to'+str(signal_id)]['car_ave_v'].append(np.mean([self.cars_dict[carid]['v'][-1] for carid in in_car_list]))
                            self.car_info_in_road[str(left_sig_idx)+'to'+str(signal_id)]['car_num_ratio'].append(len(in_car_list_ratio))
                            self.car_info_in_road[str(left_sig_idx)+'to'+str(signal_id)]['car_ave_v_ratio'].append(np.mean([self.cars_dict[carid]['v'][-1] for carid in in_car_list_ratio]))
                            
                            if is_stop_UD:
                                inflow_sum += len(in_car_list)
                                RL_inflow_sum += len(in_car_list)
                                RL_inflow_v_sum += len(in_car_list)
                                inflow_v_sum += np.sum([self.cars_dict[carid]['v'][-1] for carid in in_car_list])
                    elif direction == 'left':
                        if right_sig_idx != None:
                            y_ = sig_xy[1] - self.intersection_w/4
                            x_under = sig_xy[0] + self.intersection_w/2
                            x_upper = self.signals_dict[right_sig_idx]['xy'][0] - self.intersection_w/2
                            x_upper_ratio = self.signals_dict[right_sig_idx]['xy'][0] - self.intersection_w/2 - self.L*self.obs_ratio
                            in_car_list = [carid for carid in car_id_list if (y_==self.cars_dict[carid]['y'][-1]) and (self.cars_dict[carid]['x'][-1]>=x_under) and (self.cars_dict[carid]['x'][-1]<=x_upper)]
                            in_car_list_ratio = [carid for carid in car_id_list if (y_==self.cars_dict[carid]['y'][-1]) and (self.cars_dict[carid]['x'][-1]>=x_under) and (self.cars_dict[carid]['x'][-1]<=x_upper_ratio)]
                            self.car_info_in_road[str(right_sig_idx)+'to'+str(signal_id)]['car_num'].append(len(in_car_list))
                            self.car_info_in_road[str(right_sig_idx)+'to'+str(signal_id)]['car_ave_v'].append(np.mean([self.cars_dict[carid]['v'][-1] for carid in in_car_list]))
                            self.car_info_in_road[str(right_sig_idx)+'to'+str(signal_id)]['car_num_ratio'].append(len(in_car_list_ratio))
                            self.car_info_in_road[str(right_sig_idx)+'to'+str(signal_id)]['car_ave_v_ratio'].append(np.mean([self.cars_dict[carid]['v'][-1] for carid in in_car_list_ratio]))
                            if is_stop_UD:
                                inflow_sum += len(in_car_list)
                                RL_inflow_sum += len(in_car_list)
                                RL_inflow_v_sum += len(in_car_list)
                                inflow_v_sum += np.sum([self.cars_dict[carid]['v'][-1] for carid in in_car_list])
                            
                if inflow_sum == 0:
                    self.signals_dict[signal_id]['inflow_ave_v'].append(0)
                    self.signals_dict[signal_id]['inflow_num'].append(0)
                else:
                    self.signals_dict[signal_id]['inflow_ave_v'].append(inflow_v_sum/inflow_sum)
                    self.signals_dict[signal_id]['inflow_num'].append(inflow_sum)

                if UD_inflow_sum == 0:
                    self.signals_dict[signal_id]['UD_inflow_ave_v'].append(0)
                    self.signals_dict[signal_id]['UD_inflow_num'].append(0)
                else:
                    self.signals_dict[signal_id]['UD_inflow_ave_v'].append(UD_inflow_v_sum/UD_inflow_sum)
                    self.signals_dict[signal_id]['UD_inflow_num'].append(UD_inflow_sum)
                if RL_inflow_sum == 0:
                    self.signals_dict[signal_id]['RL_inflow_ave_v'].append(0)
                    self.signals_dict[signal_id]['RL_inflow_num'].append(0)
                else:
                    self.signals_dict[signal_id]['RL_inflow_ave_v'].append(RL_inflow_v_sum/RL_inflow_sum)
                    self.signals_dict[signal_id]['RL_inflow_num'].append(RL_inflow_sum)

    def _caluculate_dot(self, x, y, v, car_id):
        '''
        x, y, v change calculation
        dot:(x, y, v)
        '''
        dot = np.zeros(3)

        a = self.cars_dict[car_id]['a']
        bf = self.cars_dict[car_id]['bf']
        bc = self.cars_dict[car_id]['bc']

        current_d = self.cars_dict[car_id]['d'][self.now_step-1]
        ahead_diff, _ = self._get_ahead_diff(car_id)
        ahead_diff = np.min([ahead_diff, self.max_dist])

        if current_d == 'up':
            dot[0] = 0
            dot[1] = v
            dot[2] = a*(self.get_V(ahead_diff, bf=bf,bc=bc) - v)
        elif current_d == 'down':
            dot[0] = 0
            dot[1] = -v
            dot[2] = a*(self.get_V(ahead_diff, bf=bf,bc=bc) - v)
        elif current_d == 'right':
            dot[0] = v
            dot[1] = 0
            dot[2] = a*(self.get_V(ahead_diff, bf=bf,bc=bc) - v)
        elif current_d == 'left':
            dot[0] = -v
            dot[1] = 0
            dot[2] = a*(self.get_V(ahead_diff, bf=bf,bc=bc) - v)
        return dot
                
    def _update_xyv(self, car_id):
        '''
        Change the position and speed of the car according to the conditions
        '''
        # now position
        x = self.cars_dict[car_id]['x'][self.now_step-1]
        y = self.cars_dict[car_id]['y'][self.now_step-1]
        v = self.cars_dict[car_id]['v'][self.now_step-1]
        # Change
        dot = self.cars_dict[car_id]['dot']
        dx = dot[0]
        dy = dot[1]
        dv = dot[2]

        current_d = self.cars_dict[car_id]['d'][self.now_step-1]
        nearest_signal_idx = self.cars_dict[car_id]['nearest_signal_idx']
        sig_xy = self.signals_dict[nearest_signal_idx]['xy']
        is_stop_UD = self.signals_dict[nearest_signal_idx]['judge'][self.now_step]
        next_border = list(sig_xy)
        ahead_diff, ahead_v = self._get_ahead_diff(car_id)
        ahead_diff = np.min([ahead_diff, self.max_dist])

        if not is_stop_UD: # if the upper and lower traffic lights are blue

            ## Change calculation
            next_d = self.cars_dict[car_id]['next_d'] 
            next_border = list(sig_xy)
            if current_d == 'up':
                next_border[1] = next_border[1]- self.intersection_w/2
                if y<=next_border[1] and (y+dy)>next_border[1]: # if car is going over the stop line
                    ## Determining new directions according to conditions
                    self.cars_dict[car_id]['next_d'] = np.random.choice(list(self.signals_dict[nearest_signal_idx]['directions']),p=self.signals_dict[nearest_signal_idx]['trans_p'])
                    while self.cars_dict[car_id]['next_d']=='down':
                        self.cars_dict[car_id]['next_d'] = np.random.choice(list(self.signals_dict[nearest_signal_idx]['directions']),p=self.signals_dict[nearest_signal_idx]['trans_p'])

                    next_d = self.cars_dict[car_id]['next_d']

                    if self.signals_dict[nearest_signal_idx]['InCar'] >= 3: # If there is a car inside the intersection 
                        dx = 0
                        dy = next_border[1] - y
                        dv = -self.cars_dict[car_id]['v'][self.now_step-1]
                        self.cars_dict[car_id]['x'].append(x+dx)
                        self.cars_dict[car_id]['y'].append(y+dy)
                        self.cars_dict[car_id]['v'].append(v+dv)
                        self.cars_dict[car_id]['d'].append(self.cars_dict[car_id]['d'][self.now_step-1])
                    else:
                        self.cars_dict[car_id]['x'].append(x+dx)
                        self.cars_dict[car_id]['y'].append(y+dy)
                        self.cars_dict[car_id]['v'].append(v+dv)
                        self.cars_dict[car_id]['d'].append(self.cars_dict[car_id]['d'][self.now_step-1])
                else:
                    next_d = self.cars_dict[car_id]['next_d']

                    if next_d == 'up':
                        self.cars_dict[car_id]['x'].append(x+dx)
                        self.cars_dict[car_id]['y'].append(y+dy)
                        self.cars_dict[car_id]['v'].append(v+dv)
                        self.cars_dict[car_id]['d'].append(next_d)
                    elif next_d == 'right':
                        next_y = sig_xy[1]+self.intersection_w/4 
                        if y<=next_y and y+dy > next_y:
                            dx = y+dy - next_y
                            dy = next_y - y
                            self.cars_dict[car_id]['x'].append(x+dx)
                            self.cars_dict[car_id]['y'].append(y+dy)
                            self.cars_dict[car_id]['v'].append(v+dv)
                            self.cars_dict[car_id]['d'].append(next_d)
                        else:
                            self.cars_dict[car_id]['x'].append(x+dx)
                            self.cars_dict[car_id]['y'].append(y+dy)
                            self.cars_dict[car_id]['v'].append(v+dv)
                            self.cars_dict[car_id]['d'].append(self.cars_dict[car_id]['d'][self.now_step-1])
                    elif next_d == 'left':
                        next_y = sig_xy[1] - self.intersection_w/4
                        if y <= next_y and y+dy> next_y:
                            dx = next_y - (y+dy)
                            dy = next_y - y
                            self.cars_dict[car_id]['x'].append(x+dx)
                            self.cars_dict[car_id]['y'].append(y+dy)
                            self.cars_dict[car_id]['v'].append(v+dv)
                            self.cars_dict[car_id]['d'].append(next_d)
                        else:
                            self.cars_dict[car_id]['x'].append(x+dx)
                            self.cars_dict[car_id]['y'].append(y+dy)
                            self.cars_dict[car_id]['v'].append(v+dv)
                            self.cars_dict[car_id]['d'].append(self.cars_dict[car_id]['d'][self.now_step-1])
            elif current_d == 'down':
                next_border[1] = next_border[1] + self.intersection_w/2
                if y>=next_border[1] and (y+dy)<next_border[1]:
                    self.cars_dict[car_id]['next_d'] = np.random.choice(list(self.signals_dict[nearest_signal_idx]['directions']),p=self.signals_dict[nearest_signal_idx]['trans_p'])
                    while self.cars_dict[car_id]['next_d']=='up':
                        self.cars_dict[car_id]['next_d'] = np.random.choice(list(self.signals_dict[nearest_signal_idx]['directions']),p=self.signals_dict[nearest_signal_idx]['trans_p'])

                    next_d = self.cars_dict[car_id]['next_d']

                    if self.signals_dict[nearest_signal_idx]['InCar'] >= 3: # If there is a car inside the intersection
                        dx = 0
                        dy = next_border[1] - y
                        dv = -self.cars_dict[car_id]['v'][self.now_step-1]
                        self.cars_dict[car_id]['x'].append(x+dx)
                        self.cars_dict[car_id]['y'].append(y+dy)
                        self.cars_dict[car_id]['v'].append(v+dv)
                        self.cars_dict[car_id]['d'].append(self.cars_dict[car_id]['d'][self.now_step-1])
                    else:
                        self.cars_dict[car_id]['x'].append(x+dx)
                        self.cars_dict[car_id]['y'].append(y+dy)
                        self.cars_dict[car_id]['v'].append(v+dv)
                        self.cars_dict[car_id]['d'].append(self.cars_dict[car_id]['d'][self.now_step-1])
                else:    
                    ## Determining new directions according to conditions
                    next_d = self.cars_dict[car_id]['next_d']

                    if next_d =='down':
                        self.cars_dict[car_id]['x'].append(x+dx)
                        self.cars_dict[car_id]['y'].append(y+dy)
                        self.cars_dict[car_id]['v'].append(v+dv)
                        self.cars_dict[car_id]['d'].append(next_d)
                    elif next_d == 'right':
                        next_y = sig_xy[1]+self.intersection_w/4
                        if y>=next_y and y+dy<next_y:
                            dx = next_y - (y+dy)
                            dy = next_y - y
                            self.cars_dict[car_id]['x'].append(x+dx)
                            self.cars_dict[car_id]['y'].append(y+dy)
                            self.cars_dict[car_id]['v'].append(v+dv)
                            self.cars_dict[car_id]['d'].append(next_d)
                        else:
                            self.cars_dict[car_id]['x'].append(x+dx)
                            self.cars_dict[car_id]['y'].append(y+dy)
                            self.cars_dict[car_id]['v'].append(v+dv)
                            self.cars_dict[car_id]['d'].append(self.cars_dict[car_id]['d'][self.now_step-1])
                    elif next_d =='left':
                        next_y = sig_xy[1]-self.intersection_w/4
                        if y >= next_y and y+dy < next_y:
                            dx = y+dy - next_y
                            dy = next_y - y
                            self.cars_dict[car_id]['x'].append(x+dx)
                            self.cars_dict[car_id]['y'].append(y+dy)
                            self.cars_dict[car_id]['v'].append(v+dv)
                            self.cars_dict[car_id]['d'].append(next_d)
                        else:
                            self.cars_dict[car_id]['x'].append(x+dx)
                            self.cars_dict[car_id]['y'].append(y+dy)
                            self.cars_dict[car_id]['v'].append(v+dv)
                            self.cars_dict[car_id]['d'].append(self.cars_dict[car_id]['d'][self.now_step-1])
            # red traffic light
            elif current_d == 'right':
                next_border[0] = next_border[0]- self.intersection_w/2 # stop line
                if x > next_border[0]: # Continue past the stop line
                    if next_d == 'right':
                        self.cars_dict[car_id]['x'].append(x+dx)
                        self.cars_dict[car_id]['y'].append(y+dy)
                        self.cars_dict[car_id]['v'].append(v+dv)
                        self.cars_dict[car_id]['d'].append(next_d)
                    elif next_d == 'up':
                        next_x = sig_xy[0]-self.intersection_w/4
                        if x<=next_x and x+dx > next_x:
                            dx = next_x - x
                            dy = x+dx - next_x
                            self.cars_dict[car_id]['x'].append(x+dx)
                            self.cars_dict[car_id]['y'].append(y+dy)
                            self.cars_dict[car_id]['v'].append(v+dv)
                            self.cars_dict[car_id]['d'].append(next_d)
                        else:
                            self.cars_dict[car_id]['x'].append(x+dx)
                            self.cars_dict[car_id]['y'].append(y+dy)
                            self.cars_dict[car_id]['v'].append(v+dv)
                            self.cars_dict[car_id]['d'].append(self.cars_dict[car_id]['d'][self.now_step-1])
                    elif next_d == 'down':
                        next_x = sig_xy[0]+self.intersection_w/4
                        if x<=next_x and x+v > next_x:
                            dx = next_x - x
                            dy = next_x - (x+dx)
                            self.cars_dict[car_id]['x'].append(x+dx)
                            self.cars_dict[car_id]['y'].append(y+dy)
                            self.cars_dict[car_id]['v'].append(v+dv)
                            self.cars_dict[car_id]['d'].append(next_d)
                        else:
                            self.cars_dict[car_id]['x'].append(x+dx)
                            self.cars_dict[car_id]['y'].append(y+dy)
                            self.cars_dict[car_id]['v'].append(v+dv)
                            self.cars_dict[car_id]['d'].append(self.cars_dict[car_id]['d'][self.now_step-1])
                else: # If the stop line not exceeded
                    if x+ahead_diff>next_border[0]: # if car in front exceeded the stop line
                        if x<=next_border[0] and (x+dx)>next_border[0]: # if car is going over the stop line
                            dx = next_border[0]- x
                            dy = 0
                            dv = -self.cars_dict[car_id]['v'][self.now_step-1]
                            self.cars_dict[car_id]['x'].append(x+dx)
                            self.cars_dict[car_id]['y'].append(y+dy)
                            self.cars_dict[car_id]['v'].append(v+dv)
                            self.cars_dict[car_id]['d'].append(self.cars_dict[car_id]['d'][self.now_step-1])
                        else:
                            self.cars_dict[car_id]['x'].append(x+dx)
                            self.cars_dict[car_id]['y'].append(y+dy)
                            self.cars_dict[car_id]['v'].append(v+dv)
                            self.cars_dict[car_id]['d'].append(self.cars_dict[car_id]['d'][self.now_step-1])
                    else: # if the car in front has not crossed the stop line
                        if x<=next_border[0] and (x+dx)>next_border[0]: 
                            dx = next_border[0] -self.min_dist - x
                            dy = 0
                            dv = -self.cars_dict[car_id]['v'][self.now_step-1]
                            self.cars_dict[car_id]['x'].append(x+dx)
                            self.cars_dict[car_id]['y'].append(y+dy)
                            self.cars_dict[car_id]['v'].append(v+dv)
                            self.cars_dict[car_id]['d'].append(self.cars_dict[car_id]['d'][self.now_step-1])
                        else:
                            self.cars_dict[car_id]['x'].append(x+dx)
                            self.cars_dict[car_id]['y'].append(y+dy)
                            self.cars_dict[car_id]['v'].append(v+dv)
                            self.cars_dict[car_id]['d'].append(self.cars_dict[car_id]['d'][self.now_step-1])

            elif current_d == 'left':
                next_border[0] = next_border[0] + self.intersection_w/2
                if x < next_border[0]:# Continue past the stop line
                    if next_d == 'left':
                        self.cars_dict[car_id]['x'].append(x+dx)
                        self.cars_dict[car_id]['y'].append(y+dy)
                        self.cars_dict[car_id]['v'].append(v+dv)
                        self.cars_dict[car_id]['d'].append(next_d)
                    elif next_d == 'up':
                        next_x = sig_xy[0]-self.intersection_w/4
                        if x>=next_x and x+dx < next_x:
                            dx = next_x - x
                            dy = next_x - (x+dx)
                            self.cars_dict[car_id]['x'].append(x+dx)
                            self.cars_dict[car_id]['y'].append(y+dy)
                            self.cars_dict[car_id]['v'].append(v+dv)
                            self.cars_dict[car_id]['d'].append(next_d)
                        else:
                            self.cars_dict[car_id]['x'].append(x+dx)
                            self.cars_dict[car_id]['y'].append(y+dy)
                            self.cars_dict[car_id]['v'].append(v+dv)
                            self.cars_dict[car_id]['d'].append(self.cars_dict[car_id]['d'][self.now_step-1])
                    elif next_d == 'down':
                        next_x = sig_xy[0]+self.intersection_w/4
                        if x>=next_x and x+dx < next_x:
                            dx = next_x - x
                            dy = x+dx - next_x
                            self.cars_dict[car_id]['x'].append(x+dx)
                            self.cars_dict[car_id]['y'].append(y+dy)
                            self.cars_dict[car_id]['v'].append(v+dv)
                            self.cars_dict[car_id]['d'].append(next_d)
                        else:
                            self.cars_dict[car_id]['x'].append(x+dx)
                            self.cars_dict[car_id]['y'].append(y+dy)
                            self.cars_dict[car_id]['v'].append(v+dv)
                            self.cars_dict[car_id]['d'].append(self.cars_dict[car_id]['d'][self.now_step-1])
                else:# If the stop line not exceeded
                    if x-ahead_diff<next_border[0]:# if car in front exceeded the stop line
                        if x>=next_border[0] and (x+dx)<next_border[0]:
                            dx = next_border[0] - x
                            dy = 0
                            dv = -self.cars_dict[car_id]['v'][self.now_step-1]
                            self.cars_dict[car_id]['x'].append(x+dx)
                            self.cars_dict[car_id]['y'].append(y+dy)
                            self.cars_dict[car_id]['v'].append(v+dv)
                            self.cars_dict[car_id]['d'].append(self.cars_dict[car_id]['d'][self.now_step-1])
                        else:
                            self.cars_dict[car_id]['x'].append(x+dx)
                            self.cars_dict[car_id]['y'].append(y+dy)
                            self.cars_dict[car_id]['v'].append(v+dv)
                            self.cars_dict[car_id]['d'].append(self.cars_dict[car_id]['d'][self.now_step-1])
                    else: # if the car in front has not crossed the stop line
                        if x>=next_border[0] and (x+dx)<next_border[0]:
                            dx = next_border[0] + self.min_dist - x
                            dy = 0
                            dv = -self.cars_dict[car_id]['v'][self.now_step-1]
                            self.cars_dict[car_id]['x'].append(x+dx)
                            self.cars_dict[car_id]['y'].append(y+dy)
                            self.cars_dict[car_id]['v'].append(v+dv)
                            self.cars_dict[car_id]['d'].append(self.cars_dict[car_id]['d'][self.now_step-1])
                        else:
                            self.cars_dict[car_id]['x'].append(x+dx)
                            self.cars_dict[car_id]['y'].append(y+dy)
                            self.cars_dict[car_id]['v'].append(v+dv)
                            self.cars_dict[car_id]['d'].append(self.cars_dict[car_id]['d'][self.now_step-1])

        else: # If the left and right lights are green
            ## Change Calculate
            next_border = list(sig_xy)
            next_d = self.cars_dict[car_id]['next_d']
            if current_d == 'right':
                next_border[0] = next_border[0] - self.intersection_w/2
                if x<=next_border[0] and (x+dx)>next_border[0]: # if car is going over the stop line
                    ## Determining new directions according to conditions
                    self.cars_dict[car_id]['next_d'] = np.random.choice(list(self.signals_dict[nearest_signal_idx]['directions']),p=self.signals_dict[nearest_signal_idx]['trans_p'])
                    while self.cars_dict[car_id]['next_d']=='left':
                        self.cars_dict[car_id]['next_d'] = np.random.choice(list(self.signals_dict[nearest_signal_idx]['directions']),p=self.signals_dict[nearest_signal_idx]['trans_p'])

                    if self.signals_dict[nearest_signal_idx]['InCar'] >=3: # If there is a car inside the intersection
                        dx = next_border[0] - x
                        dy = 0
                        dv = -self.cars_dict[car_id]['v'][self.now_step-1]
                        self.cars_dict[car_id]['x'].append(x+dx)
                        self.cars_dict[car_id]['y'].append(y+dy)
                        self.cars_dict[car_id]['v'].append(v+dv)
                        self.cars_dict[car_id]['d'].append(self.cars_dict[car_id]['d'][self.now_step-1])
                    else:
                        self.cars_dict[car_id]['x'].append(x+dx)
                        self.cars_dict[car_id]['y'].append(y+dy)
                        self.cars_dict[car_id]['v'].append(v+dv)
                        self.cars_dict[car_id]['d'].append(self.cars_dict[car_id]['d'][self.now_step-1])
                else:
                    next_d = self.cars_dict[car_id]['next_d']
                    if next_d == 'right':
                        self.cars_dict[car_id]['x'].append(x+dx)
                        self.cars_dict[car_id]['y'].append(y+dy)
                        self.cars_dict[car_id]['v'].append(v+dv)
                        self.cars_dict[car_id]['d'].append(next_d)
                    elif next_d == 'up':
                        next_x = sig_xy[0]-self.intersection_w/4
                        if x<=next_x and x+dx>next_x:
                            dx = next_x - x
                            dy = x+dx - next_x
                            self.cars_dict[car_id]['x'].append(x+dx)
                            self.cars_dict[car_id]['y'].append(y+dy)
                            self.cars_dict[car_id]['v'].append(v+dv)
                            self.cars_dict[car_id]['d'].append(next_d)
                        else:
                            self.cars_dict[car_id]['x'].append(x+dx)
                            self.cars_dict[car_id]['y'].append(y+dy)
                            self.cars_dict[car_id]['v'].append(v+dv)
                            self.cars_dict[car_id]['d'].append(self.cars_dict[car_id]['d'][self.now_step-1])
                    elif next_d == 'down':
                        next_x = sig_xy[0]+self.intersection_w/4
                        if x<=next_x and x+dx>next_x:
                            dx = next_x - x
                            dy = next_x - (x+dx)
                            self.cars_dict[car_id]['x'].append(x+dx)
                            self.cars_dict[car_id]['y'].append(y+dy)
                            self.cars_dict[car_id]['v'].append(v+dv)
                            self.cars_dict[car_id]['d'].append(next_d)
                        else:
                            self.cars_dict[car_id]['x'].append(x+dx)
                            self.cars_dict[car_id]['y'].append(y+dy)
                            self.cars_dict[car_id]['v'].append(v+dv)
                            self.cars_dict[car_id]['d'].append(self.cars_dict[car_id]['d'][self.now_step-1])
            elif current_d == 'left':
                next_border[0] = next_border[0] + self.intersection_w/2
                if x>=next_border[0] and (x+dx)<next_border[0]:
                    ## Determining new directions according to conditions
                    self.cars_dict[car_id]['next_d'] = np.random.choice(list(self.signals_dict[nearest_signal_idx]['directions']))
                    while self.cars_dict[car_id]['next_d']=='right':
                        self.cars_dict[car_id]['next_d'] = np.random.choice(list(self.signals_dict[nearest_signal_idx]['directions']))

                    if self.signals_dict[nearest_signal_idx]['InCar'] >=3: # If there is a car inside the intersection
                        dx = next_border[0] - x
                        dy = 0
                        dv = -self.cars_dict[car_id]['v'][self.now_step-1]
                        self.cars_dict[car_id]['x'].append(x+dx)
                        self.cars_dict[car_id]['y'].append(y+dy)
                        self.cars_dict[car_id]['v'].append(v+dv)
                        self.cars_dict[car_id]['d'].append(self.cars_dict[car_id]['d'][self.now_step-1])
                    else:
                        self.cars_dict[car_id]['x'].append(x+dx)
                        self.cars_dict[car_id]['y'].append(y+dy)
                        self.cars_dict[car_id]['v'].append(v+dv)
                        self.cars_dict[car_id]['d'].append(self.cars_dict[car_id]['d'][self.now_step-1])
                else:
                    next_d = self.cars_dict[car_id]['next_d']

                    if next_d == 'left':
                        self.cars_dict[car_id]['x'].append(x+dx)
                        self.cars_dict[car_id]['y'].append(y+dy)
                        self.cars_dict[car_id]['v'].append(v+dv)
                        self.cars_dict[car_id]['d'].append(self.cars_dict[car_id]['d'][self.now_step-1])
                    elif next_d == 'up':
                        next_x = sig_xy[0] - self.intersection_w/4
                        if x>=next_x and x+dx < next_x:
                            dx = next_x - x
                            dy = next_x - (x+dx)
                            self.cars_dict[car_id]['x'].append(x+dx)
                            self.cars_dict[car_id]['y'].append(y+dy)
                            self.cars_dict[car_id]['v'].append(v+dv)
                            self.cars_dict[car_id]['d'].append(next_d)
                        else:
                            self.cars_dict[car_id]['x'].append(x+dx)
                            self.cars_dict[car_id]['y'].append(y+dy)
                            self.cars_dict[car_id]['v'].append(v+dv)
                            self.cars_dict[car_id]['d'].append(self.cars_dict[car_id]['d'][self.now_step-1])
                    elif next_d == 'down':
                        next_x = sig_xy[0] + self.intersection_w/4
                        if x>=next_x and x+dx < next_x:
                            dx = next_x - x
                            dy = x+dx - next_x
                            self.cars_dict[car_id]['x'].append(x+dx)
                            self.cars_dict[car_id]['y'].append(y+dy)
                            self.cars_dict[car_id]['v'].append(v+dv)
                            self.cars_dict[car_id]['d'].append(next_d)
                        else:
                            self.cars_dict[car_id]['x'].append(x+dx)
                            self.cars_dict[car_id]['y'].append(y+dy)
                            self.cars_dict[car_id]['v'].append(v+dv)
                            self.cars_dict[car_id]['d'].append(self.cars_dict[car_id]['d'][self.now_step-1])
            # Red traffic signal
            elif current_d == 'up':
                next_border[1] = next_border[1]- self.intersection_w/2
                if y > next_border[1]: # Continue past the stop line
                    if next_d == 'up':
                        self.cars_dict[car_id]['x'].append(x+dx)
                        self.cars_dict[car_id]['y'].append(y+dy)
                        self.cars_dict[car_id]['v'].append(v+dv)
                        self.cars_dict[car_id]['d'].append(next_d)
                    elif next_d == 'right':
                        next_y = sig_xy[1] + self.intersection_w/4
                        if y<=next_y and y+dy > next_y:
                            dx = y+dy - next_y
                            dy = next_y - y
                            self.cars_dict[car_id]['x'].append(x+dx)
                            self.cars_dict[car_id]['y'].append(y+dy)
                            self.cars_dict[car_id]['v'].append(v+dv)
                            self.cars_dict[car_id]['d'].append(next_d)
                        else:
                            self.cars_dict[car_id]['x'].append(x+dx)
                            self.cars_dict[car_id]['y'].append(y+dy)
                            self.cars_dict[car_id]['v'].append(v+dv)
                            self.cars_dict[car_id]['d'].append(self.cars_dict[car_id]['d'][self.now_step-1])
                    elif next_d == 'left':
                        next_y = sig_xy[1] - self.intersection_w/4
                        if y<=next_y and y+dy>next_y:
                            dx = next_y - (y+dy)
                            dy = next_y - y
                            self.cars_dict[car_id]['x'].append(x+dx)
                            self.cars_dict[car_id]['y'].append(y+dy)
                            self.cars_dict[car_id]['v'].append(v+dv)
                            self.cars_dict[car_id]['d'].append(next_d)
                        else:
                            self.cars_dict[car_id]['x'].append(x+dx)
                            self.cars_dict[car_id]['y'].append(y+dy)
                            self.cars_dict[car_id]['v'].append(v+dv)
                            self.cars_dict[car_id]['d'].append(self.cars_dict[car_id]['d'][self.now_step-1])
                else: # If the stop line not exceeded
                    if y+ahead_diff>next_border[1]: # if car in front exceeded the stop line
                        if y<=next_border[1] and (y+dy)>next_border[1]: # if car is going over the stop line
                            dx = 0
                            dy = next_border[1] - y
                            dv = -self.cars_dict[car_id]['v'][self.now_step-1]
                            self.cars_dict[car_id]['x'].append(x+dx)
                            self.cars_dict[car_id]['y'].append(y+dy)
                            self.cars_dict[car_id]['v'].append(v+dv)
                            self.cars_dict[car_id]['d'].append(self.cars_dict[car_id]['d'][self.now_step-1])
                        else:
                            self.cars_dict[car_id]['x'].append(x+dx)
                            self.cars_dict[car_id]['y'].append(y+dy)
                            self.cars_dict[car_id]['v'].append(v+dv)
                            self.cars_dict[car_id]['d'].append(self.cars_dict[car_id]['d'][self.now_step-1])
                    else: # if the car in front has not crossed the stop line
                        if y<=next_border[1] and (y+dy)>next_border[1]: # if car is going over the stop line
                            dx = 0
                            dy = next_border[1] -self.min_dist- y
                            dv = -self.cars_dict[car_id]['v'][self.now_step-1]
                            self.cars_dict[car_id]['x'].append(x+dx)
                            self.cars_dict[car_id]['y'].append(y+dy)
                            self.cars_dict[car_id]['v'].append(v+dv)
                            self.cars_dict[car_id]['d'].append(self.cars_dict[car_id]['d'][self.now_step-1])
                        else:
                            self.cars_dict[car_id]['x'].append(x+dx)
                            self.cars_dict[car_id]['y'].append(y+dy)
                            self.cars_dict[car_id]['v'].append(v+dv)
                            self.cars_dict[car_id]['d'].append(self.cars_dict[car_id]['d'][self.now_step-1])

            elif current_d == 'down':
                next_border[1] = next_border[1] + self.intersection_w/2
                if y < next_border[1]:# Continue past the stop line
                    if next_d == 'down':
                        self.cars_dict[car_id]['x'].append(x+dx)
                        self.cars_dict[car_id]['y'].append(y+dy)
                        self.cars_dict[car_id]['v'].append(v+dv)
                        self.cars_dict[car_id]['d'].append(next_d)
                    elif next_d == 'right':
                        next_y = sig_xy[1] + self.intersection_w/4
                        if y>=next_y and y+dy<next_y:
                            dx = next_y - (y+dy)
                            dy = next_y - y
                            self.cars_dict[car_id]['x'].append(x+dx)
                            self.cars_dict[car_id]['y'].append(y+dy)
                            self.cars_dict[car_id]['v'].append(v+dv)
                            self.cars_dict[car_id]['d'].append(next_d)
                        else:
                            self.cars_dict[car_id]['x'].append(x+dx)
                            self.cars_dict[car_id]['y'].append(y+dy)
                            self.cars_dict[car_id]['v'].append(v+dv)
                            self.cars_dict[car_id]['d'].append(self.cars_dict[car_id]['d'][self.now_step-1])
                    elif next_d == 'left':
                        next_y = sig_xy[1] - self.intersection_w/4 
                        if y>=next_y and y+dy<next_y:
                            dx = y+dy - next_y
                            dy = next_y - y
                            self.cars_dict[car_id]['x'].append(x+dx)
                            self.cars_dict[car_id]['y'].append(y+dy)
                            self.cars_dict[car_id]['v'].append(v+dv)
                            self.cars_dict[car_id]['d'].append(next_d)
                        else:
                            self.cars_dict[car_id]['x'].append(x+dx)
                            self.cars_dict[car_id]['y'].append(y+dy)
                            self.cars_dict[car_id]['v'].append(v+dv)
                            self.cars_dict[car_id]['d'].append(self.cars_dict[car_id]['d'][self.now_step-1])
                else: # If the stop line not exceeded
                    if y-ahead_diff<next_border[1]: # if car in front exceeded the stop line
                        if y>=next_border[1] and (y+dy)<next_border[1]:
                            dx = 0
                            dy = next_border[1] - y
                            dv = -self.cars_dict[car_id]['v'][self.now_step-1]
                            self.cars_dict[car_id]['x'].append(x+dx)
                            self.cars_dict[car_id]['y'].append(y+dy)
                            self.cars_dict[car_id]['v'].append(v+dv)
                            self.cars_dict[car_id]['d'].append(self.cars_dict[car_id]['d'][self.now_step-1])
                        else:
                            self.cars_dict[car_id]['x'].append(x+dx)
                            self.cars_dict[car_id]['y'].append(y+dy)
                            self.cars_dict[car_id]['v'].append(v+dv)
                            self.cars_dict[car_id]['d'].append(self.cars_dict[car_id]['d'][self.now_step-1])
                    else:# if the car in front has not crossed the stop line
                        if y>=next_border[1] and (y+dy)<next_border[1]:
                            dx = 0
                            dy = next_border[1] + self.min_dist - y
                            dv = -self.cars_dict[car_id]['v'][self.now_step-1]
                            self.cars_dict[car_id]['x'].append(x+dx)
                            self.cars_dict[car_id]['y'].append(y+dy)
                            self.cars_dict[car_id]['v'].append(v+dv)
                            self.cars_dict[car_id]['d'].append(self.cars_dict[car_id]['d'][self.now_step-1])
                        else:
                            self.cars_dict[car_id]['x'].append(x+dx)
                            self.cars_dict[car_id]['y'].append(y+dy)
                            self.cars_dict[car_id]['v'].append(v+dv)
                            self.cars_dict[car_id]['d'].append(self.cars_dict[car_id]['d'][self.now_step-1])
    
    #################################  Update  ######################################  

    ###############################  Information  #####################################
    def _get_car_info(self, param_name):
        info_list = []
        for i in range(self.num_car):
            param = self.cars_dict[i][param_name]
            info_list.append(param)

        return np.array(info_list)

    def _get_signal_info(self, param_name):
        info_list = []
        for i in range(self.num_signals):
            param = self.signals_dict[i][param_name]
            info_list.append(param)
        
        return np.array(info_list)    
    ################################  Information  ####################################
