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

class Reward(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def get_reward(self, flights: List[Flight]) -> np.ndarray:
        pass

class NoReward(Reward):

    def get_reward(self, flights: List[Flight]) -> np.ndarray:
        return np.ones((len(flights),1))
    
class DefaultReward(Reward):

    def __init__(self, 
                 intrusion_weight: float = -1.,
                 drift_weight: float = -0.002):
        super().__init__()
        self.intrusion_weight = intrusion_weight
        self.drift_weight = drift_weight

    def get_reward(self, flights: List[Flight]) -> np.ndarray:

        reward = np.zeros(len(flights))
        reward += self.get_intrusion_reward(flights)
        reward += self.get_drift_reward(flights)

        return reward

    def get_intrusion_reward(self, flights: List[Flight]) -> np.ndarray:
        
        intrusions = np.zeros(len(flights))
        
        x = np.array([f.position.x for f in flights])
        y = np.array([f.position.y for f in flights])

        min_distance = np.array([f.aircraft.min_distance for f in flights])
        _, _, distances = fn.get_distance_matrices(x, y)
        distances = fn.remove_diagonal(distances)

        # count for each aircraft the number of distances < separation minima
        intrusions = (distances < min_distance[:, np.newaxis]).sum(axis = 1)

        if sum(intrusions) > 0:
            print(intrusions)
            input("press any key to continue")
                           
        return intrusions * self.intrusion_weight
    
    def get_drift_reward(self, flights: List[Flight]) -> np.ndarray:

        drift = np.array([abs(f.drift) for f in flights])
        return drift * self.drift_weight

