# flocking.py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import sys
from IPython.display import HTML #maybe get rid of
from matplotlib import colors
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import rotate
from scipy.stats import norm
from scipy.stats import circmean
import time
import os
from scipy.signal import find_peaks #for vorticies
from collections import defaultdict #unique distances
import statsmodels.api as sm #smoothing
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics import r2_score
import plotly.graph_objs as go
import plotly.figure_factory as ff

# import cProfile
# import pstats
# profiler = cProfile.Profile()
# profiler.enable()

try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    drive.mount('/content/drive')
    from IPython.display import HTML
    base_path = '/content/drive/MyDrive/Brown/21-S3-Colorado/jack-flocking'
else:
    matplotlib.rcParams['animation.ffmpeg_path'] = '/opt/homebrew/bin/ffmpeg'
    # animation.writer: ffmpeg
    base_path = '/Users/elijah_lew-smith/Documents/work/2023-2-summer/flocking'
def load_parameters_to_instance(filepath, instance):
    """
    Load parameters from a compressed .npz file and set them as attributes on the given instance.

    Parameters:
    - filepath: str, the path to the .npz file containing the parameters.
    - instance: object, the instance of the class to which the parameters will be set.
    """
    data = np.load(filepath, allow_pickle=True)
    for key in data.files:
        value = data[key]
        # Convert 0-d numpy arrays to scalars
        if isinstance(value, np.ndarray) and value.shape == ():
            value = value.item()
        setattr(instance, key, value)

class ActiveXY:
    defaults = {
        'X': 10,
        'Y': 10,
        'stop_time': 10,
        'model_type': 'XY',
        'active_component':   ['dV', 'muV'],
        'damping_noise_temp': [1, None, 0.7],
        'activity': 1,
        'dt': 0.1,
        'max_saved_points': 5e7,
        'boundary_conditions': 'periodic',
        'background': 'uniform',
        'perturbation_type': 'none',
        'background_angle': np.pi / 2,
        'perturbation_angle': np.pi / 4,
        'perturbation_size': 20,
        'real_time_interval': 10,
        'first_estimate': 2,
        'num_runs': 1,
        'figure_size': 3
    }

    def __init__(self, **kwargs):
        for key, default in self.defaults.items():
            setattr(self, key, kwargs.get(key, default))

        #This is so that it can be change after creating the class by rerunning this 
        self.set_figure_dim(self.figure_size)

        if self.boundary_conditions not in ['periodic', 'open']:
            raise ValueError(f"Boundary conditions must be periodic or open but are set to {self.boundary_conditions}")
        #in case of invalid input
        self.background_angle = self.background_angle % (2 * np.pi)
        self.perturbation_angle = self.perturbation_angle % (2 * np.pi)

        # None-keyword class varibles
        self.results = None
        self.times = None

        self.center = [self.X//2,self.Y//2]

        self._start_time = None
        self._real_time = [0, 0]
        # self.full_time = 0

        self.perturbation_size = self.perturbation_size
        # force function for finite and infinite T

        if 'dV' in self.active_component:
            self.include_dV = True  
        if 'muV' in self.active_component: 
            self.include_muV= True  

        self.initialize_active_force()


        #Parallel processing. 
        self.num_cpus = os.cpu_count()
        self.max_threads = min(self.num_runs, self.num_cpus)

        self._result_list = [] 
        self._time_list = [] 

        #graphing
        self.scale = .3
        self.fontweight =  'normal'
        self.fontsize = 8

        self.initialize_FDT()
        self.equilibrium_file_path = base_path+f'/equilibrium/X-{self.X}_Y-{self.Y}_T-{self.temp}_J-{self.damping}.npz'
        self.title_objects = ['model_type','size','temp','damping','sim_time','real_time','initial_condition']


    ###  ------------------ Time and Evolution  ------------------

    def time_evolution(self):
            self.initialize_time_evolution()
            for run_number in range(self.num_runs): 
                self.run_number = run_number
                self.initialize_single_time_evolution()
                self.constant_time_evolution(run_number)
                self.finalize_time_evolution(run_number)


    def constant_time_evolution(self,run_number):
        t = 0
        n = 1
        while t < self.stop_time:
            self.lattice = self.runge_kutta_step()
            t += self.dt
            n += 1

            #log results
            if n % self.log_increment == 0:
                self._result_list.append(np.copy(self.lattice % (np.pi * 2)))
                self._time_list.append(t)

            #log real time
            if self._start_time is None:
                self._start_time = time.time()

            self._real_time[0] = self._real_time[1]
            self._real_time[1] = time.time() - self._start_time

            if (self._real_time[0] - self.first_estimate) // self.real_time_interval < (self._real_time[1] - self.first_estimate) // self.real_time_interval:
                frac_complete = (run_number + t / (self.stop_time )) / self.num_runs
                estimated_time = self._real_time[1] / frac_complete - self._real_time[1]
                formatted_estimate = self.format_seconds(estimated_time)
                formatted_time = self.format_seconds(self._real_time[1])
                formatted_frac = f"{frac_complete:.3f}"
                print(f"Completed {formatted_frac} in {formatted_time}. Estimated remaining: {formatted_estimate}")

    def initialize_time_evolution(self):
        self.start_time = time.time()
        self.log_increment = 1

        total_number_points = self.X * self.Y * self.stop_time * self.num_runs / self.dt
        if self.max_saved_points < total_number_points:
            self.log_increment = int(total_number_points/ self.max_saved_points)
            print(f"Log increment of {self.log_increment}")



    def initialize_single_time_evolution(self):
        self.initial_condition()
        self._result_list = []
        self._time_list = []

        #log first result
        self._result_list.append(np.copy(self.lattice % (np.pi * 2)))
        self._time_list.append(0)

    def finalize_time_evolution(self, run_number):
        self.full_time = time.time() - self.start_time
        #First we intialize everything as zero 
        if run_number == 0:
            num_saved_times = len(self._time_list)
            self.results = np.zeros((num_saved_times, self.X, self.Y, self.num_runs))
            self.times =   np.zeros((num_saved_times, self.num_runs))

        #then we set the appropriate elements 
        self.results[:,:,:,run_number] = np.array(self._result_list)
        self.times[:,run_number] = np.array(self._time_list)

        #Get one with xy indexing for plotting 
        self.results_T = np.transpose(self.results, axes=(0, 2, 1, 3))

    # ## OLD strong_order_one
    # def runge_kutta_step(self):
    #     Y0 = self.lattice
    #     dW = np.random.standard_normal(Y0.shape) * np.sqrt(self.dt)
    #     k1 = self.total_force(Y0) * self.dt
    #     k2 = self.total_force(Y0 + 0.5 * k1 + 0.5 * self.noise * dW) * self.dt
    #     lattice_new = Y0 + k2 + self.noise * dW
    #     error = self.samples_per_point_per_run(np.abs(k2 - k1))
    #     return lattice_new

    #strong_order_two (11.3.2)
    def runge_kutta_step(self):
        Y0 = self.lattice

        # print('Andy:', np.average(abs(self.andy_dV1(self.lattice)+ self.andy_dV3(self.lattice))),  np.average(abs(self.andy_muV(self.lattice) / self.temp)))
        # print('\n Marvin:', np.average(abs(self.marvin_dV(self.lattice))), np.average(abs(self.marvin_muV(self.lattice) / self.temp)))
        zeta_1= np.random.standard_normal(Y0.shape)
        xi    = np.random.standard_normal(Y0.shape)
        mu    = np.random.standard_normal(Y0.shape)
        eta   = np.random.standard_normal(Y0.shape)
        phi   = np.random.standard_normal(Y0.shape)

        #since p=1 we can directly input the values of ρ and α
        a_10 = - 1/np.pi * np.sqrt(2 * self.dt) * xi - 2 * np.sqrt(self.dt * (1/12 - 1/(2 * np.pi**2))) * mu
        b_1  = np.sqrt(self.dt/2) * eta + np.sqrt(self.dt * ((np.pi**2 / 180) - (1 / (2 * np.pi**2)))) * phi

        #the final term of J vanishes for p=1
        dW = np.sqrt(self.dt) * zeta_1
        dZ = .5 * self.dt * (np.sqrt(self.dt) * zeta_1 + a_10)
        J_110 =  (1/6) * self.dt**2 * zeta_1**2 + .25 * self.dt * a_10**2 - (.5/np.pi) * self.dt**(1.5) * zeta_1 * b_1 + .25 * self.dt**(1.5) * a_10 * zeta_1
        argument = np.abs(2 * J_110 * self.dt - dZ**2)#otherwise the floating point arithmetic can't take the square root
        Ybar1 = Y0 + .5 * self.total_force(Y0) * self.dt + (self.noise / self.dt) * (dZ + np.sqrt(argument))
        Ybar2 = Y0 + .5 * self.total_force(Y0) * self.dt + (self.noise / self.dt) * (dZ - np.sqrt(argument))

        Y1 = Y0 + .5 * (self.total_force(Ybar1) + self.total_force(Ybar2)) * self.dt + self.noise * dW
        return Y1


    ###------------------  FDT ------------------
    # you need to seperate out the intial so it remains a part of the class identity
    def initialize_FDT(self):
        self.damping = self.damping_noise_temp[0]
        self.noise = self.damping_noise_temp[1]
        self.temp = self.damping_noise_temp[2]

        #and set the third
        if self.damping_noise_temp[0] is None:
            self.damping = self.damping_noise_temp[1] ** 2 / (self.damping_noise_temp[2] * 2)
        elif self.damping_noise_temp[1] is None:
            self.noise = np.sqrt(self.damping_noise_temp[0] * self.damping_noise_temp[2]* 2)
        elif self.damping_noise_temp[2] is None:
            self.temp = self.damping_noise_temp[1] ** 2 / (self.damping_noise_temp[0] * 2)
        elif self.damping_noise_temp[0] == 0 and self.damping_noise_temp[2] == 'inf':
            pass
        else:
            raise ValueError("Specify Exactly 2 FDT values")

    ###------------------ Force Initialization ------------------
    def total_force(self, lattice):
        total_force = self.dissipative_force(lattice) * self.damping
        total_force += self.active_force_function(lattice) * self.activity
        return total_force
    
    def initialize_active_force(self):
        if self.model_type == 'XY':
            #This will just give zeros
            self.active_force_function = self.create_force_function(self.zero_force, self.zero_force)
        elif self.model_type == 'kuramoto':
            self.active_force_function = self.create_force_function(self.kuramoto_dV, self.kuramoto_muV)
        elif self.model_type == '1':
            self.active_force_function = self.create_force_function(self.andy_dV, self.andy_muV)
        elif self.model_type == '2':
            self.active_force_function = self.create_force_function(self.marvin_dV, self.marvin_muV)
        elif self.model_type == '3':
            self.active_force_function = self.create_force_function(self.eli_dV, self.eli_muV)
        else:
            raise ValueError("Invalid model type")

    def create_force_function(self, dV_function, muV_function):
        def force_function(lattice):
            total_force = np.zeros_like(lattice)
            if self.include_dV:
                total_force += dV_function(lattice)
            if self.include_muV:
                total_force += muV_function(lattice) / self.temp
            return total_force
        return force_function

    def zero_force(self, lattice): 
        return np.zeros_like(lattice)
    

     ###  ------------------ Force Functions ------------------
    def shift(self, lattice, x, y):
        shifted_lattice = np.roll(np.roll(lattice, shift=x, axis=0), shift=y, axis=1)
        return shifted_lattice

    def dissipative_force(self, lattice):
        f1 = np.sin(self.shift(lattice, 1, 0) - lattice)
        f2 = np.sin(self.shift(lattice,-1, 0) - lattice)
        f3 = np.sin(self.shift(lattice, 0, 1) - lattice)
        f4 = np.sin(self.shift(lattice, 0,-1) - lattice)
        if self.boundary_conditions == 'open':
          f1[ 0,:] = 0
          f2[-1,:] = 0
          f3[:, 0] = 0
          f4[:,-1] = 0
        force = f1 + f2 + f3 + f4
        return force

    def kuramoto_dV(self,lattice):
        f1 = np.cos(self.shift(lattice, 1, 0)-lattice)
        f2 = np.cos(self.shift(lattice,-1, 0)-lattice)
        f3 = np.cos(self.shift(lattice, 0, 1)-lattice)
        f4 = np.cos(self.shift(lattice, 0,-1)-lattice)
        force =  (f1 + f2 + f3 + f4)
        return force

    def kuramoto_muV(self,lattice):
      g1 = -np.cos(self.shift(lattice, 1, 0)-lattice) * (np.sin(self.shift(lattice, 1, 0)-self.shift(lattice, 2, 0)) + np.sin(self.shift(lattice, 1, 0)-self.shift(lattice, 0, 0)))
      g2 =  np.cos(self.shift(lattice,-1, 0)-lattice) * (np.sin(self.shift(lattice,-1, 0)-self.shift(lattice, 0, 0)) + np.sin(self.shift(lattice,-1, 0)-self.shift(lattice,-2, 0)))
      g3 = -np.cos(self.shift(lattice, 1, 0)-lattice) * (np.sin(self.shift(lattice, 1, 0)-self.shift(lattice, 1, 1)) + np.sin(self.shift(lattice, 1, 0)-self.shift(lattice, 1,-1)))
      g4 =  np.cos(self.shift(lattice,-1, 0)-lattice) * (np.sin(self.shift(lattice,-1, 0)-self.shift(lattice,-1, 1)) + np.sin(self.shift(lattice,-1, 0)-self.shift(lattice,-1,-1)))
      g5 = -np.cos(self.shift(lattice, 0, 1)-lattice) * (np.sin(self.shift(lattice, 0, 1)-self.shift(lattice, 1, 1)) + np.sin(self.shift(lattice, 0, 1)-self.shift(lattice,-1, 1)))
      g6 =  np.cos(self.shift(lattice, 0,-1)-lattice) * (np.sin(self.shift(lattice, 0,-1)-self.shift(lattice, 1,-1)) + np.sin(self.shift(lattice, 0,-1)-self.shift(lattice,-1,-1)))
      g7 = -np.cos(self.shift(lattice, 0, 1)-lattice) * (np.sin(self.shift(lattice, 0, 1)-self.shift(lattice, 0, 2)) + np.sin(self.shift(lattice, 0, 1)-self.shift(lattice, 0, 0)))
      g8 =  np.cos(self.shift(lattice, 0,-1)-lattice) * (np.sin(self.shift(lattice, 0,-1)-self.shift(lattice, 0, 0)) + np.sin(self.shift(lattice, 0,-1)-self.shift(lattice, 0,-2)))
      force = (g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8)
      return force

    def andy_dV1(self,lattice):
      f1 =  np.sin(self.shift(lattice, 1, 0)-lattice) * np.cos((self.shift(lattice, 1, 0)+lattice)/2)
      f2 = -np.sin(self.shift(lattice,-1, 0)-lattice) * np.cos((self.shift(lattice,-1, 0)+lattice)/2)
      f3 =  np.sin(self.shift(lattice, 0, 1)-lattice) * np.sin((self.shift(lattice, 0, 1)+lattice)/2)
      f4 = -np.sin(self.shift(lattice, 0,-1)-lattice) * np.sin((self.shift(lattice, 0,-1)+lattice)/2)
      if self.boundary_conditions == 'open':
          f1[ 0,:] = 0
          f2[-1,:] = 0
          f3[:, 0] = 0
          f4[:,-1] = 0
      force = .5 * (f1 + f2 + f3 + f4)
      return force

    def andy_dV3(self,lattice):
      f1 =  (np.cos(self.shift(lattice, 1, 0)-lattice)-1) * np.sin((self.shift(lattice, 1, 0)+lattice)/2)
      f2 = -(np.cos(self.shift(lattice,-1, 0)-lattice)-1) * np.sin((self.shift(lattice,-1, 0)+lattice)/2)
      f3 = -(np.cos(self.shift(lattice, 0, 1)-lattice)-1) * np.cos((self.shift(lattice, 0, 1)+lattice)/2)
      f4 =  (np.cos(self.shift(lattice, 0,-1)-lattice)-1) * np.cos((self.shift(lattice, 0,-1)+lattice)/2)
      if self.boundary_conditions == 'open':
          f1[ 0,:] = 0
          f2[-1,:] = 0
          f3[:, 0] = 0
          f4[:,-1] = 0
      force = .25 *  (f1 + f2 + f3 + f4)
      return force
    
    def andy_dV(self, lattice):
        return self.andy_dV1(lattice) + self.andy_dV3(lattice)

    def andy_muV(self,lattice):
      f1 =  (np.cos(self.shift(lattice, 1, 0)-lattice)-1) * np.cos((self.shift(lattice, 1, 0)+lattice)/2) * (np.sin(self.shift(lattice, 1, 0)-self.shift(lattice, 2, 0)) + np.sin(self.shift(lattice, 1, 0)-self.shift(lattice, 0, 0)))
      f2 = -(np.cos(self.shift(lattice,-1, 0)-lattice)-1) * np.cos((self.shift(lattice,-1, 0)+lattice)/2) * (np.sin(self.shift(lattice,-1, 0)-self.shift(lattice, 0, 0)) + np.sin(self.shift(lattice,-1, 0)-self.shift(lattice,-2, 0)))
      f3 =  (np.cos(self.shift(lattice, 1, 0)-lattice)-1) * np.cos((self.shift(lattice, 1, 0)+lattice)/2) * (np.sin(self.shift(lattice, 1, 0)-self.shift(lattice, 1, 1)) + np.sin(self.shift(lattice, 1, 0)-self.shift(lattice, 1,-1)))
      f4 = -(np.cos(self.shift(lattice,-1, 0)-lattice)-1) * np.cos((self.shift(lattice,-1, 0)+lattice)/2) * (np.sin(self.shift(lattice,-1, 0)-self.shift(lattice,-1, 1)) + np.sin(self.shift(lattice,-1, 0)-self.shift(lattice,-1,-1)))
      f5 =  (np.cos(self.shift(lattice, 0, 1)-lattice)-1) * np.sin((self.shift(lattice, 0, 1)+lattice)/2) * (np.sin(self.shift(lattice, 0, 1)-self.shift(lattice, 1, 1)) + np.sin(self.shift(lattice, 0, 1)-self.shift(lattice,-1, 1)))
      f6 = -(np.cos(self.shift(lattice, 0,-1)-lattice)-1) * np.sin((self.shift(lattice, 0,-1)+lattice)/2) * (np.sin(self.shift(lattice, 0,-1)-self.shift(lattice, 1,-1)) + np.sin(self.shift(lattice, 0,-1)-self.shift(lattice,-1,-1)))
      f7 =  (np.cos(self.shift(lattice, 0, 1)-lattice)-1) * np.sin((self.shift(lattice, 0, 1)+lattice)/2) * (np.sin(self.shift(lattice, 0, 1)-self.shift(lattice, 0, 2)) + np.sin(self.shift(lattice, 0, 1)-self.shift(lattice, 0, 0)))
      f8 = -(np.cos(self.shift(lattice, 0,-1)-lattice)-1) * np.sin((self.shift(lattice, 0,-1)+lattice)/2) * (np.sin(self.shift(lattice, 0,-1)-self.shift(lattice, 0, 0)) + np.sin(self.shift(lattice, 0,-1)-self.shift(lattice, 0,-2)))
      if self.boundary_conditions == 'open':
          f1[:2, :] = 0
          f2[-2:, :] = 0
          f3[ 0, :] = 0
          f4[-1, :] = 0
          f5[:,  0] = 0
          f6[:, -1] = 0
          f3[:, [0,-1]] = 0
          f4[:, [0,-1]] = 0
          f5[[0,-1], :] = 0
          f6[[0,-1], :] = 0
          f7[:, :2] = 0
          f8[:, -2:] = 0
      force = .5 * (f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8)
      return force

    def marvin_dV(self,lattice):
      f1 =  np.sin(self.shift(lattice, 1, 0))
      f2 = -np.sin(self.shift(lattice,-1, 0))
      f3 = -np.cos(self.shift(lattice, 0, 1))
      f4 =  np.cos(self.shift(lattice, 0,-1))
      if self.boundary_conditions == 'open':
          f1[ 0,:] = 0
          f2[-1,:] = 0
          f3[:, 0] = 0
          f4[:,-1] = 0
      force = .5 * (f1 + f2 + f3 + f4)
      return force

    def marvin_muV(self,lattice):
      g1 =  (np.cos(lattice) + np.cos(self.shift(lattice, 1, 0))) * (np.sin(self.shift(lattice, 1, 0)-self.shift(lattice, 2, 0)) + np.sin(self.shift(lattice, 1, 0)-self.shift(lattice, 0, 0)))
      g2 = -(np.cos(lattice) + np.cos(self.shift(lattice,-1, 0))) * (np.sin(self.shift(lattice,-1, 0)-self.shift(lattice, 0, 0)) + np.sin(self.shift(lattice,-1, 0)-self.shift(lattice,-2, 0)))
      g3 =  (np.cos(lattice) + np.cos(self.shift(lattice, 1, 0))) * (np.sin(self.shift(lattice, 1, 0)-self.shift(lattice, 1, 1)) + np.sin(self.shift(lattice, 1, 0)-self.shift(lattice, 1,-1)))
      g4 = -(np.cos(lattice) + np.cos(self.shift(lattice,-1, 0))) * (np.sin(self.shift(lattice,-1, 0)-self.shift(lattice,-1, 1)) + np.sin(self.shift(lattice,-1, 0)-self.shift(lattice,-1,-1)))
      g5 =  (np.sin(lattice) + np.sin(self.shift(lattice, 0, 1))) * (np.sin(self.shift(lattice, 0, 1)-self.shift(lattice, 1, 1)) + np.sin(self.shift(lattice, 0, 1)-self.shift(lattice,-1, 1)))
      g6 = -(np.sin(lattice) + np.sin(self.shift(lattice, 0,-1))) * (np.sin(self.shift(lattice, 0,-1)-self.shift(lattice, 1,-1)) + np.sin(self.shift(lattice, 0,-1)-self.shift(lattice,-1,-1)))
      g7 =  (np.sin(lattice) + np.sin(self.shift(lattice, 0, 1))) * (np.sin(self.shift(lattice, 0, 1)-self.shift(lattice, 0, 2)) + np.sin(self.shift(lattice, 0, 1)-self.shift(lattice, 0, 0)))
      g8 = -(np.sin(lattice) + np.sin(self.shift(lattice, 0,-1))) * (np.sin(self.shift(lattice, 0,-1)-self.shift(lattice, 0, 0)) + np.sin(self.shift(lattice, 0,-1)-self.shift(lattice, 0,-2)))
      if self.boundary_conditions == 'open':
          g1[:1, :] = 0
          g2[-2:, :] = 0
          g3[ 0, :] = 0
          g4[-1, :] = 0
          g5[:,  0] = 0
          g6[:, -1] = 0
          g3[:, [0,-1]] = 0
          g4[:, [0,-1]] = 0
          g5[[0,-1], :] = 0
          g6[[0,-1], :] = 0
          g7[:, :1] = 0
          g8[:, -2:] = 0
      force = .5 * (g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8)
      return force

    def eli_dV(self,lattice):
      f1 =  np.sin((lattice + self.shift(lattice, 1, 0))/2)
      f2 = -np.sin((lattice + self.shift(lattice,-1, 0))/2)
      f3 = -np.cos((lattice + self.shift(lattice, 0, 1))/2)
      f4 =  np.cos((lattice + self.shift(lattice, 0,-1))/2)
      if self.boundary_conditions == 'open':
          f1[ 0,:] = 0
          f2[-1,:] = 0
          f3[:, 0] = 0
          f4[:,-1] = 0
      force =  (f1 + f2 + f3 + f4)
      return force

    def eli_muV(self,lattice):
      g1 =  (np.cos(.5*(lattice + self.shift(lattice, 1, 0)))) * (np.sin(self.shift(lattice, 1, 0)-self.shift(lattice, 2, 0)) + np.sin(self.shift(lattice, 1, 0)-self.shift(lattice, 0, 0)))
      g2 = -(np.cos(.5*(lattice + self.shift(lattice,-1, 0)))) * (np.sin(self.shift(lattice,-1, 0)-self.shift(lattice, 0, 0)) + np.sin(self.shift(lattice,-1, 0)-self.shift(lattice,-2, 0)))
      g3 =  (np.cos(.5*(lattice + self.shift(lattice, 1, 0)))) * (np.sin(self.shift(lattice, 1, 0)-self.shift(lattice, 1, 1)) + np.sin(self.shift(lattice, 1, 0)-self.shift(lattice, 1,-1)))
      g4 = -(np.cos(.5*(lattice + self.shift(lattice,-1, 0)))) * (np.sin(self.shift(lattice,-1, 0)-self.shift(lattice,-1, 1)) + np.sin(self.shift(lattice,-1, 0)-self.shift(lattice,-1,-1)))
      g5 =  (np.sin(.5*(lattice + self.shift(lattice, 0, 1)))) * (np.sin(self.shift(lattice, 0, 1)-self.shift(lattice, 1, 1)) + np.sin(self.shift(lattice, 0, 1)-self.shift(lattice,-1, 1)))
      g6 = -(np.sin(.5*(lattice + self.shift(lattice, 0,-1)))) * (np.sin(self.shift(lattice, 0,-1)-self.shift(lattice, 1,-1)) + np.sin(self.shift(lattice, 0,-1)-self.shift(lattice,-1,-1)))
      g7 =  (np.sin(.5*(lattice + self.shift(lattice, 0, 1)))) * (np.sin(self.shift(lattice, 0, 1)-self.shift(lattice, 0, 2)) + np.sin(self.shift(lattice, 0, 1)-self.shift(lattice, 0, 0)))
      g8 = -(np.sin(.5*(lattice + self.shift(lattice, 0,-1)))) * (np.sin(self.shift(lattice, 0,-1)-self.shift(lattice, 0, 0)) + np.sin(self.shift(lattice, 0,-1)-self.shift(lattice, 0,-2)))
      if self.boundary_conditions == 'open':
          g1[:1, :] = 0
          g2[-2:, :] = 0
          g3[ 0, :] = 0
          g4[-1, :] = 0
          g5[:,  0] = 0
          g6[:, -1] = 0
          g3[:, [0,-1]] = 0
          g4[:, [0,-1]] = 0
          g5[[0,-1], :] = 0
          g6[[0,-1], :] = 0
          g7[:, :1] = 0
          g8[:, -2:] = 0
      force = 2 * (g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8)
      return force


    ###  ------------------ Initialization  ------------------
    def _perpendicular_vector_angle(self,vector):
        x, y = vector
        perpendicular_x, perpendicular_y = (-y, x)
        angle_rad = np.arctan2(perpendicular_y, perpendicular_x)
        if angle_rad < 0:
            angle_rad += 2 * np.pi
        return angle_rad

    def initial_condition(self):
        self.center = [self.X//2,self.Y//2]
        center1 = (self.X // 2, self.Y // 4)
        center2 = (self.X // 2, 3* self.Y // 4)
        if self.background == 'uniform':
          self.theta_initial = np.zeros((self.X, self.Y)) + self.background_angle
        elif self.background == 'uniform_samples':
            change_per_run = np.pi *2 / self.num_runs 
            self.theta_initial = np.zeros((self.X, self.Y)) + change_per_run * self.run_number 
        elif self.background == 'random':
            self.theta_initial = np.random.rand(self.X, self.Y) * np.pi * 2
        elif self.background == 'equilibrium':
          #if temperatuer is infinite, equilibrium is random
          if self.temp =='inf':
            self.theta_initial = np.random.rand(self.X, self.Y) * np.pi * 2
          #if there is a saved file, we use that
          elif os.path.exists(self.equilibrium_file_path):
              self.theta_initial = self.get_equilibirum_file()
          #if not, we make one
          else:
            self.equilibrium_prompt()
            #tThis is a just in case check the file doesn't create (and shouldn't ever run)
            if not os.path.exists(self.equilibrium_file_path):
                print("didn't create path for some reason")
            self.theta_initial = self.get_equilibirum_file()
        else:
            raise ValueError("Invalid initial background type")
        if self.perturbation_type == 'none':
            pass
        elif self.perturbation_type == 'smooth_random':
          theta_random = np.random.rand(self.X, self.Y) * np.pi * 2
          theta_temporary = np.copy(theta_random)
          filter_radius = 10
          grid_x, grid_y = np.meshgrid(np.arange(self.Y), np.arange(self.X))
          for i in range(self.X):
              for j in range(self.Y):
                  distances = np.sqrt((grid_x - i)**2 + (grid_y - j)**2)
                  circular_indices = np.where(distances <= filter_radius)
                  subarray = theta_temporary[circular_indices]
                  theta_random[i, j] = circmean(subarray)
        elif self.perturbation_type == 'random_blocks':
            for k in range(self.X // self.perturbation_size):
              for l in range(self.Y // self.perturbation_size):
                random_number = np.random.rand()*np.pi*2
                for i in range(self.perturbation_size):
                  for j in range(self.perturbation_size):
                    self.theta_initial[i+k*self.perturbation_size, j+l*self.perturbation_size] = random_number
        elif self.perturbation_type == 'stripe':
          self.theta_initial[self.X//2 - self.perturbation_size//2:self.X//2 + self.perturbation_size//2,:]=self.perturbation_angle
          self.theta_initial = gaussian_filter(self.theta_initial, sigma=self.perturbation_size/5)
        elif self.perturbation_type == 'cross':
          self.theta_initial[self.X//2 - self.perturbation_size//2:self.X//2 + self.perturbation_size//2,:]=self.perturbation_angle
          self.theta_initial[:,self.Y//2 - self.perturbation_size//2:self.Y//2 + self.perturbation_size//2]=self.perturbation_angle
          self.theta_initial[self.X//2 - self.perturbation_size//2:self.X//2 + self.perturbation_size//2,self.Y//2 - self.perturbation_size//2:self.Y//2 + self.perturbation_size//2]=self.background_angle-self.perturbation_angle
        elif self.perturbation_type == 'bottom':
          self.theta_initial[0:20,:] = self.background_angle+np.pi
        elif self.perturbation_type == 'half_random':
          self.theta_initial[:, self.Y//4:]
          for i in range(int(self.X)):
              for j in range(int(self.Y/4), int(3*self.Y/4)):
                  self.theta_initial[i, j] = np.pi+self.background_angle
        elif self.perturbation_type == 'uniform':
            self.theta_initial = self.theta_initial
        elif self.perturbation_type == 'blob':
            self.theta_initial_block = self.theta_initial
            for i in range(self.perturbation_size):
                for j in range(self.perturbation_size):
                    x = int(self.X/2 - self.perturbation_size/2 + i)
                    y = int(self.Y/2 - self.perturbation_size/2 + j)
                    self.theta_initial_block[x, y] = self.perturbation_angle
            self.theta_initial = gaussian_filter(self.theta_initial_block, sigma=self.perturbation_size/5)
        elif self.perturbation_type == 'blob_random':
            for i in range(self.X):
                for j in range(self.Y):
                    x, y = j - self.center[0], i - self.center[1]
                    radius = np.sqrt(x**2 + y**2)
                    if radius <= self.perturbation_size / 2:
                        self.theta_initial[i, j] = np.random.rand() * np.pi * 2
        elif self.perturbation_type == 'plus_vortex':
            for i in range(self.X):
              for j in range(self.Y):
                self.theta_initial[i, j] = ( np.arctan2(self.center[1]-j, self.center[0]-i)+np.pi/2+self.perturbation_angle )% (2 * np.pi)

        elif self.perturbation_type == 'minus_vortex':
            for i in range(self.X):
              for j in range(self.Y):
                self.theta_initial[i, j] = (np.arctan2(i - self.center[0], j - self.center[1])+np.pi/2+self.perturbation_angle )% (2 * np.pi)
            # for i in range(self.X):
            #     for j in range(self.Y):
            #         x, y = j - self.center[0], i - self.center[1]
            #         radius = np.sqrt(x**2 + y**2)
            #         if radius <= self.perturbation_size / 2:
            #             self.theta_initial[i, j] = ((self._perpendicular_vector_angle((x, y))+self.perturbation_angle) % (2 * np.pi))
        elif self.perturbation_type =='vortex_demo':
            vortex_demo_path = base_path + f'/simulation-results/_XY_X-{self.X}_Y-{self.Y}_noise-0.1_J-1_activity-1.npz'
            if not os.path.exists(vortex_demo_path):
              raise FileNotFoundError(f"No vortex demo with X={self.X}, Y={self.Y}, J={self.damping} has been created")
            self.theta_initial = np.load(vortex_demo_path)['results'][-1]
        elif self.perturbation_type == 'opposite_vortex':
            for i in range(self.X):
              for j in range(self.Y):
                center1 = [self.center[0]-self.perturbation_size, self.center[1]]
                center2 = [self.center[0]+self.perturbation_size, self.center[1]]
                votex1 = np.arctan2(i - center1[0], j - center1[1])
                vortex2 = np.arctan2(i - center2[0], j - center2[1])
                self.theta_initial[i, j] = (votex1-vortex2+np.pi/2+self.perturbation_angle)% (2 * np.pi)
                # self.theta_initial[i, j] = (np.arctan2(self.center[0], self.center[1] + i + self.perturbation_size) - np.arctan2(self.center[0], self.center[1]+ i - self.perturbation_size) * np.pi )% (2 * np.pi)
        elif self.perturbation_type == 'same_vortex':
          for i in range(self.X):
              for j in range(self.Y):
                  x, y = j - center1[0], i - center1[1]
                  radius = np.sqrt(x**2 + y**2)
                  if j <= self.Y / 2:
                      self.theta_initial[i, j] = ((self._perpendicular_vector_angle((x, y))+self.perturbation_angle) % (2 * np.pi))
                  w, z = j - center2[0], i - center2[1]
                  radius = np.sqrt(w**2 + z**2)
                  if j > self.Y / 2:
                      self.theta_initial[i, j] = ((self._perpendicular_vector_angle((w,z))+self.perturbation_angle) % (2 * np.pi))
        elif self.perturbation_type == 'counter_vertical_vortex':
          for i in range(self.X):
              for j in range(self.Y):
                  x, y = j - center1[0], i - center1[1]
                  radius = np.sqrt(x**2 + y**2)
                  if j <= self.Y / 2:
                      self.theta_initial[i, j] = (self._perpendicular_vector_angle((x, y)) % (2 * np.pi))
                  w, z = j - center2[0], i - center2[1]
                  radius = np.sqrt(w**2 + z**2)
                  if j > self.Y / 2:
                      self.theta_initial[i, j] = (self._perpendicular_vector_angle((z,w)) % (2 * np.pi))
        elif self.perturbation_type == 'radial':
            for i in range(self.X):
                for j in range(self.Y):
                    x, y = j - self.center[0], i - self.center[1]
                    radius = np.sqrt(x**2 + y**2)
                    if radius <= self.perturbation_size / 2:
                        self.theta_initial[i, j] = np.arctan2(y, x)
        elif self.perturbation_type == 'perturbation':
          if self.perturbation_size == 1:
              self.theta_initial[self.center[0], self.center[1]] = self.perturbation_angle
          else:
              for i in range(self.center[0] - self.perturbation_size//2, self.center[0] + self.perturbation_size//2):
                  for j in range(self.center[1] - self.perturbation_size//2, self.center[1] + self.perturbation_size//2):
                      self.theta_initial[i, j] = self.perturbation_angle
        else:
            raise ValueError("Invalid initial perturbation type")
        self.lattice = self.theta_initial % (np.pi * 2)

    def get_equilibirum_file(self): 
        loaded_result = np.load(self.equilibrium_file_path)['results']
        #We only take after time=100 
        initial_index = np.argmin(abs(loaded_result[0]-100))
        random_index = np.random.randint(initial_index, np.shape(loaded_result)[0])
        theta_initial = loaded_result[random_index, :, :, 0]
        return theta_initial


    def equilibrium_prompt(self):
        msg = f"No equilibrium with X={self.X}, Y={self.Y}, J={self.damping}, and T={self.noise} has been created. Would you like to make one ('y'/'yes' or anything for no)?"
        response = input(msg).strip().lower()
        if response == 'y' or response == 'yes':
            # 1. Generate a new instance with a uniform initial condition.
            params = vars(self).copy()
            params['background'] = 'uniform'
            params['dt'] = 0.1
            params['perturbation'] = 'none'
            params['boundary_conditions'] = 'periodic'
            params['stop_time'] = 1000
            params['model_type'] = 'XY'
            params['num_runs'] = 1
            params['max_saved_points'] = 1e6
            equilibrium_sim = ActiveXY(**params)

            # 2. Run the system's time evolution from this uniform condition.
            equilibrium_sim.time_evolution()

            # 3. Save the final state as the equilibrium.
            equilibrium_sim.save_equilibrium()
            print('Equilibrium Created')

        else:
            raise FileNotFoundError(f"Fine. Have it your way.")
    ###  ------------------ Outputting Data ------------------
    def save_equilibrium(self):
      np.savez(self.equilibrium_file_path, **vars(self))

    def save_permanent(self, specific_path = None): 
        if specific_path: 
            mid_path = f'/simulations-permanent/{specific_path}/'
        else: 
            mid_path = f'/simulations-permanent/'
        file_path =  base_path + mid_path + f'_{self.model_type}_X-{self.X}_Y-{self.Y}_J-{self.damping}_temp-{self.temp:.3}_time-{self.stop_time}_runs-{self.num_runs}_dt-{self.dt}'
        i = 0 
        file_exists = False 
        while file_exists == False:
            instance_file_path = file_path + f'_file-{i}.npz' 
            if os.path.exists(instance_file_path):
                file_exists = False 
                i += 1
            else: 
                np.savez_compressed(instance_file_path, **vars(self))
                print(f'File number {i}')
                file_exists = True 
                

    def save_temporary(self):
        file_path =  base_path + f'/simulations-temporary/_{self.model_type}_X-{self.X}_Y-{self.Y}_time-{self.stop_time}_runs-{self.num_runs}_dt-{self.dt}_J-{self.damping}_temp-{self.temp:.3}_activity-{self.activity}'
        i = 0 
        file_exists = False 
        while file_exists == False:
            instance_file_path = file_path + f'_file-{i}.npz' 
            if os.path.exists(instance_file_path):
                file_exists = False 
                i += 1
            else: 
                np.savez(instance_file_path, **vars(self))
                print(f'File number {i}')
                file_exists = True 
                

    def get_simulation(self):
        params = vars(self).copy()
        return params 


    ###  ------------------ Displaying Time and Numbers ------------------
    def decimal_format(self, value, places):
        #first, check if it is a number
        if isinstance(value, str):
            return value
        else:
            #if its an integer, give that
            if int(value) == value:
                return str(int(value))
            #if not, round to three places
            else:
                format_string = f"{{:.{places}f}}"
                return format_string.format(value).rstrip('0').rstrip('.')

    def format_seconds(self, seconds):
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        sec = seconds % 60

        hour_str = f"{hours:02.0f}"
        min_str = f"{minutes:02.0f}"
        sec_str = f"{sec:02.0f}"
        return ":".join([hour_str, min_str, sec_str])


    ###  ------------------ Images  ------------------

    def set_figure_dim(self, figure_size):
        self.figure_X = figure_size*np.sqrt(self.Y/self.X)
        self.figure_Y = figure_size*np.sqrt(self.X/self.Y)
        self.figure_dim = (self.figure_X, self.figure_Y )
    
    def graph_label(self, title_objects=None):
        if title_objects is not None:
          self.title_objects = title_objects
        model_display_mappings = {
            'XY': 'XY',
            'kuramoto': 'Kuramoto',
            '1': '1st',
            '2': '2nd',
            '3': '3rd'
        }

        model_displayed = model_display_mappings.get(self.model_type, '')

        perturbation_displayed = ''
        if self.perturbation_type and self.perturbation_type != 'none':
            perturbation_displayed = f'with {self.perturbation_type} size {self.perturbation_size}'

        _title_handlers = {
            'model_type': model_displayed,
            'size': f'{self.X}x{self.Y}',
            'temp': f'T={self.decimal_format(self.temp,3)}',
            'damping': f'γ={self.decimal_format(self.damping,3)}',
            'sim_time': f'Sim Time={self.stop_time}',
            # 'real_time': f'Real Time={self.format_seconds(self.full_time)}',
            'initial_condition': f'IC={self.background} {perturbation_displayed}'
        }

        title_parts = []
        for item in self.title_objects:
            handler = _title_handlers.get(item)
            if handler:
                title_parts.append(handler)

        return ', '.join(title_parts)



    def _plot_configuration(self, ax, configuration, if_title, arrow_number, x_coarseness=None, y_coarseness=None):
        #this will make it so the size of the arrows is determined automatically so they fill the image
        desired_total_arrows = arrow_number
        scale_factor = np.sqrt(desired_total_arrows / (self.X * self.Y))
        x_arrow_number = round(self.X * scale_factor)
        y_arrow_number = round(self.Y * scale_factor)
        if x_coarseness is None:
            x_coarseness = int(self.X / x_arrow_number)
        if y_coarseness is None:
            y_coarseness = int(self.Y / y_arrow_number)
        max_arrow_length = min(x_coarseness, y_coarseness) * 0.65
        if max_arrow_length <= 0:
            raise ValueError("Too many arrows for size of grid")

        scale = 1 / max_arrow_length
        arrow_results = np.copy(configuration)
        x_grid, y_grid = np.meshgrid(np.arange(0, self.X, x_coarseness), np.arange(0, self.Y, y_coarseness))
        color = ax.pcolormesh(configuration, vmin=0, vmax=2*np.pi, cmap='hsv', alpha=0.5)
        arrow = ax.quiver(x_grid, y_grid,
                          np.cos(arrow_results[::x_coarseness, ::y_coarseness]),
                          np.sin(arrow_results[::x_coarseness, ::y_coarseness]),
                          angles='xy', scale_units='xy',
                          scale=scale)
        ax.set_xticks([])
        ax.set_yticks([])
        if if_title:
          ax.set_title(self.graph_label(), fontsize=self.fontsize)
        return color, arrow


    def plotly_configuration(self, configuration=None):
        if configuration is None: 
            configuration = self.results[-1,:,:,0]

        fig = go.Figure()

        fig.add_trace(go.Heatmap(
            z=configuration,
            zmin=0, zmax=np.pi*2, 
            colorscale='hsv',
            showscale=False,  
        ))
        if self.arrow_number > 0:
            X, Y = configuration.shape
            arrow_length = X/self.arrow_number * 0.5
            # Arrows
            x, y = np.meshgrid(np.linspace(0, X-1, self.arrow_number, dtype=int), np.linspace(0, Y-1, self.arrow_number, dtype=int))
            u = np.cos(configuration)[y, x] 
            v = np.sin(configuration)[y, x] 

            # Create quiver plot
            quiver_fig = ff.create_quiver(x, y, u, v, scale=arrow_length)

            arrow_color = 'black'  # Define your desired color here
            for trace in quiver_fig.data:
                trace['line']['color'] = arrow_color
                if 'marker' in trace:
                    trace['marker']['color'] = arrow_color
            # Merge quiver plot with heatmap
            for trace in quiver_fig.data:
                fig.add_trace(trace)
     
        # Show the figure
        return fig

    def show_initial_condition(self, arrow_number=100):
        initial_configuration = self.initial_condition()
        fig, ax = plt.subplots(1, 1, figsize=self.figure_dim )
        _,_ = self._plot_configuration(ax, initial_configuration, if_title=True, arrow_number=arrow_number)
        return ax

    def show_images(self, sim_number=1,  arrow_number=100, title_objects=None):
        #Make sure that it doesn't try to show more plots than there are
        sim_number = min(sim_number, self.num_runs)
        fig, ax = plt.subplots(sim_number, 2, figsize=(2 * self.figure_X, sim_number * self.figure_Y))
        for i in range(sim_number): 
            if sim_number > 1:
                # Use ax[i, 0] and ax[i, 1] when there are multiple rows
                _,_ = self._plot_configuration(ax[i, 0], self.results_T[0, :, :, i], if_title=False, arrow_number=arrow_number)
                _,_ = self._plot_configuration(ax[i, 1], self.results_T[-1, :, :, i], if_title=False, arrow_number=arrow_number)
            else:
                # Use ax[0] and ax[1] when there's only one row
                _,_ = self._plot_configuration(ax[0], self.results_T[0, :, :, 0], if_title=False, arrow_number=arrow_number)
                _,_ = self._plot_configuration(ax[1], self.results_T[-1, :, :, 0], if_title=False, arrow_number=arrow_number)
        fig.suptitle(self.graph_label(), fontsize=self.fontsize)
        plt.show()


     ###  ------------------ Video  ------------------
    def _update_frame(self, frame, color, arrow, time_per_frame):
        current_sim_time = frame * time_per_frame
        actual_frame_idx = int(current_sim_time  / (self.dt*self.log_increment) )
        if actual_frame_idx < self.times.shape[0] and frame != 0:
            color.set_array(self.results_T[actual_frame_idx,:,:,self.run_number_shown].ravel())
            arrow.set_UVC(np.cos(self.results_T[actual_frame_idx, ::self.x_coarseness, ::self.y_coarseness, self.run_number_shown].ravel()),
                          np.sin(self.results_T[actual_frame_idx, ::self.x_coarseness, ::self.y_coarseness, self.run_number_shown].ravel()))
        return color, arrow

    def make_video(self, fps, length, arrow_number):
        #Interval is in ms and fps is in seconds. 
        interval         = 1000 / fps
        shown_frames     = fps * length
        time_per_frame   = self.stop_time / shown_frames

        scale_factor     = np.sqrt(arrow_number / (self.X * self.Y))
        x_arrow_number   = round(self.X * scale_factor)
        y_arrow_number   = round(self.Y * scale_factor)
        self.x_coarseness = int(self.X / x_arrow_number)
        self.y_coarseness = int(self.Y / y_arrow_number)

        fig, ax = plt.subplots(1, 1, figsize=self.figure_dim )
        color, arrow = self._plot_configuration(ax=ax, 
                                                configuration=self.results_T[0,:,:,self.run_number_shown], 
                                                if_title=True,
                                                arrow_number=arrow_number, 
                                                x_coarseness=self.x_coarseness, 
                                                y_coarseness=self.y_coarseness)
        anim = animation.FuncAnimation(fig, 
                                       lambda frame: self._update_frame(frame, color, arrow, time_per_frame),
                                        frames=shown_frames, interval=interval, blit=False)


        plt.close()
        HTML(anim.to_html5_video())
        return anim

    def video(self, permanent=False, fps=10, length=10, arrow_number=150, title_objects=None, run_number_shown=0):
        self.run_number_shown = run_number_shown
        if title_objects is not None:
            self.title_objects = title_objects
        #If it's permanent usually nice to have it longer
        if permanent:
            fps = 20 
            length = 30
        anim = self.make_video(fps, length, arrow_number)
        if IN_COLAB:
            video = HTML(anim.to_html5_video())
            return video
        else:
            if not permanent:
                video_path = "video.mp4"
                anim.save(video_path, writer='ffmpeg', fps=fps)
            else: 
                video_path = base_path + f'/videos-permanent/'+f'length-{length}_fps-{fps}_{self.model_type}_X-{self.X}_Y-{self.Y}_temp-{self.temp}_J-{self.damping}_activity-{self.activity}.mp4'
                anim.save(video_path, writer='ffmpeg', fps=fps)
            # os.system(f"open {video_file}")




##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##################################################### NEW CLASS ##############################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################

##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##################################################### NEW CLASS ##############################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################

# class MultiActiveXY:
#     def __init__(self,
#                  num_simulations=2,
#                  sim_kwargs={},
#                  corr_params={}):
#         self.num_simulations = num_simulations
#         self.sim_kwargs = sim_kwargs
#         self.corr_params = corr_params
#         self.all_correlations = []
#         self.correlation_times = None

#     def run_simulation(self, _):
#         sim_instance = ActiveXY(**self.sim_kwargs)
#         sim_instance.time_evolution()

#         # Set self.correlation_times only if it hasn't been set before
#         if self.correlation_times is None:
#             self.correlation_times = sim_instance.times

#         correlation_instance = CorrelationFunctions(results=sim_instance.results,
#                                                      times=sim_instance.times,
#                                                      **self.corr_params)
#         correlation_instance.time_correlation()
#         return correlation_instance.correlations

#     def run_simulations(self):
#         num_cpus = os.cpu_count()
#         max_threads = min(self.num_simulations, num_cpus)
#         print(f"Running simulations on {max_threads} CPU cores...")

#         #the range(self.num_simulations) is so that each run_simulation has a unique argument
#         with ThreadPoolExecutor(max_threads) as executor:
#             for i, result in enumerate(executor.map(self.run_simulation, range(self.num_simulations))):
#                 self.all_correlations.append(result)
#                 if (i + 1) % max_threads == 0 or i + 1 == self.num_simulations:
#                     print(f"Completed {i + 1} out of {self.num_simulations} simulations")
#         self.correlation_averages = np.mean(self.all_correlations, axis=0)



