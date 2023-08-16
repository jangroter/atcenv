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
        return np.zeros((len(flights),1))