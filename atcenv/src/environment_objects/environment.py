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
from atcenv.src.environment_objects.airspace import Airspace


class Environment(ABC):

    def __init__(self,
                 dt: Optional[float] = 5,
                 max_episode_len: Optional[int] = 150):
        self.dt = dt
        self.max_episode_len = max_episode_len

        self.airspace = None
        self.flights = None
        self.done = False

        self.counter = 0

    def create_environment(self, airspace: Airspace, flights: Flight):
        self.done = False
        self.airspace = airspace
        self.flights = flights
        self.counter = 0

    @abstractmethod
    def step(self, action: np.ndarray) -> bool:
        pass

