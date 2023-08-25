from shapely.geometry import Point, Polygon
from typing import Tuple, Optional, List
import math
import random
import numpy as np
import pickle

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import atcenv.src.functions as fn
from atcenv.src.environment_objects.flight import Flight

class SubModel():
    
    def __init__(self, neurons: int = 10, layers: int = 10):
        self.neurons = neurons
        self.layers = layers

        print(f"I am a subclass with {self.neurons} neurons, and {self.layers} layers.")

class Model(ABC):

    def __init__(self, action_dim: int):
        self.action_dim = action_dim
        self.transform_action = False
        
    @abstractmethod
    def get_action(self, observation: dict) -> np.ndarray:
        pass

    @abstractmethod
    def store_transition(self, *args) -> None:
        pass

    @abstractmethod
    def new_episode(self, test: bool) -> None:
        pass

    @abstractmethod
    def setup_model(self, experiment_folder: str) -> None:
        """ initializes the model for the current experiment
        
        should either load the correct weights and make a copy of them to the experiment folder
        or create a directory in the experiment folder for saving the weights during training 
        
        Parameters
        ___________
        experiment_folder: str
            location where the output weights will be saves

        Returns
        __________
        None

        """
        self.experiment_folder = experiment_folder
        pass

class Straight(Model):

    def get_action(self, observation: dict) -> np.ndarray:
        observation = observation["observation"]
        return np.zeros((len(observation),self.action_dim))
    
    def store_transition(self, *args) -> None:
        pass
    
    def new_episode(self, test: bool) -> None:
        pass   

    def setup_model(self, experiment_folder: str) -> None:
        super().setup_model(experiment_folder)

class Random(Model):

    def __init__(self, action_dim: int):
        super().__init__(action_dim)
        self.transform_action = True

    def get_action(self, observation: dict) -> np.ndarray:
        observation = observation["observation"]
        
        return np.random.standard_normal((len(observation),self.action_dim)) * 0.33
    
    def store_transition(self, *args) -> None:
        pass
    
    def new_episode(self, test: bool) -> None:
        pass   

    def setup_model(self, experiment_folder: str) -> None:
        super().setup_model(experiment_folder)

class MVP(Model):

    def __init__(self, action_dim: int, t_lookahead: float = 300., margin: float = 1.05 ):
        super().__init__(action_dim)
        self.t_lookahead = t_lookahead
        self.margin = margin
        self.past_conflicts = []
    
    def get_action(self, observation: dict) -> np.ndarray:

        flights = observation["flights"]
        observation = observation["observation"]

        self.check_past_conflicts(len(flights))

        I = np.eye(len(flights))
        
        drift = observation[:,0]
        v_dif = observation[:,1]
        
        x = observation[:,2]
        y = observation[:,3]
        vx = observation[:,4]
        vy = observation[:,5]

        dx, dy, dvx, dvy, dist, dv2 = self.get_rel_states(x, y, vx, vy)

        tcpa = -(np.multiply(dvx, dx) + np.multiply(dvy, dy)) / dv2 + 1e9 * I
        dcpa2 = np.abs(np.multiply(dist, dist) - np.multiply(np.multiply(tcpa, tcpa),  dv2))
        dcpa = np.sqrt(dcpa2)

        action = np.zeros((len(flights),2))

        for i in range(len(flights)):
            delta_vx = 0
            delta_vy = 0

            conf_list = []

            min_distance = flights[i].aircraft.min_distance
            for j in range(len(flights)):
                if (i != j and
                    (tcpa[i,j] < self.t_lookahead and tcpa[i,j] > 0 and 
                     dcpa[i,j] < min_distance * self.margin or
                     dist[i,j] < min_distance * self.margin)):
                    
                    dcpa_x = dx[i,j] + dvx[i,j]*tcpa[i,j]
                    dcpa_y = dy[i,j] + dvy[i,j]*tcpa[i,j]
                    
                    # Add aircraft 'j' to conflict list 
                    if j not in self.past_conflicts[i]:
                        self.past_conflicts[i].append(j)

                    # Compute horizontal intrusion
                    iH = min_distance - dcpa[i,j]

                    # Exception handlers for head-on conflicts
                    # This is done to prevent division by zero in the next step
                    if dcpa[i,j] <= 0.1:
                        dcpa[i,j] = 0.1
                        dcpa_x = dy[i,j] / dist[i,j] * dcpa[i,j]
                        dcpa_y = -dx[i,j] / dist[i,j] * dcpa[i,j]
                    
                    # If intruder is outside the ownship PZ, then apply extra factor
                    # to make sure that resolution does not graze IPZ
                    if min_distance < dist[i,j] and dcpa[i,j] < dist[i,j]:
                        # Compute the resolution velocity vector in horizontal direction.
                        # abs(tcpa) because it bcomes negative during intrusion.
                        erratum = np.cos(np.arcsin(min_distance / dist[i,j])-np.arcsin(dcpa[i,j] / dist[i,j]))
                        dv1 = ((min_distance / erratum - dcpa[i,j]) * dcpa_x) / (abs(tcpa[i,j]) * dcpa[i,j])
                        dv2 = ((min_distance / erratum - dcpa[i,j]) * dcpa_y) / (abs(tcpa[i,j]) * dcpa[i,j])
                    else:
                        dv1 = (iH * dcpa_x) / (abs(tcpa[i,j]) * dcpa[i,j])
                        dv2 = (iH * dcpa_y) / (abs(tcpa[i,j]) * dcpa[i,j])
                    
                    delta_vx -= dv1
                    delta_vy -= dv2

            new_vx = vx[i] + delta_vx
            new_vy = vy[i] + delta_vy

            oldtrack = (np.arctan2(vx[i],vy[i])*180/np.pi) % 360
            newtrack = (np.arctan2(new_vx,new_vy)*180/np.pi) % 360

            action[i,0] = np.deg2rad(oldtrack-newtrack)

            old_airspeed = np.sqrt(vx[i] * vx[i] + vy[i] * vy[i])
            new_airspeed = np.sqrt(new_vx * new_vx + new_vy * new_vy)

            action[i,1] = old_airspeed - new_airspeed

            for j in list(self.past_conflicts[i]):
                if tcpa[i,j] < 0:
                    self.past_conflicts[i].remove(j)
            
            if not self.past_conflicts[i]:
                action[i,0] = drift[i]
                action[i,1] = v_dif[i]

        return action

    def store_transition(self, *args) -> None:
        pass
    
    def new_episode(self, test: bool) -> None:
        self.past_conflicts = []

    def setup_model(self, experiment_folder: str) -> None:
        super().setup_model(experiment_folder)

    def check_past_conflicts(self, num_flights: int) -> None:
        if len(self.past_conflicts) == num_flights:
            pass
        else:
            self.past_conflicts = [[] for i in range(num_flights)]

    def get_rel_states(self, x: float, y: float, vx: float, vy: float) -> Tuple[float, float, float, float, float, float]:
        x_matrix = np.asmatrix(x)
        y_matrix = np.asmatrix(y)
        vx_matrix = np.asmatrix(vx)
        vy_matrix = np.asmatrix(vy)

        dx = x_matrix.T - x_matrix
        dy = y_matrix.T - y_matrix

        dist = np.sqrt(np.multiply(dx,dx) + np.multiply(dy,dy)) 

        dvx = vx_matrix.T - vx_matrix
        dvy = vy_matrix.T - vy_matrix

        dv2 = np.multiply(dvx, dvx) + np.multiply(dvy, dvy)
        dv2 = np.where(np.abs(dv2) < 1e-6, 1e-6, dv2)  # limit lower absolute value

        return dx, dy, dvx, dvy, dist, dv2

class StraightSubModel(Model):

    def __init__(self, action_dim: int, sub_model: SubModel):
        self.action_dim = action_dim
        self.transform_action = False
        self.sub_model = sub_model
        

    def get_action(self, observation: dict) -> np.ndarray:
        observation = observation["observation"]
        return np.zeros((len(observation),self.action_dim))
    
    def store_transition(self, *args) -> None:
        pass
    
    def new_episode(self, test: bool) -> None:
        pass   

    def setup_model(self, experiment_folder: str) -> None:
        super().setup_model(experiment_folder)





    


