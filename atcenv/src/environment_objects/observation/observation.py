from shapely.geometry import Point, Polygon
from typing import Tuple, Optional, List
import math
import random
import numpy as np

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import atcenv.src.functions as fn
from atcenv.src.environment_objects.airspace import Airspace
from atcenv.src.environment_objects.flight import Flight

class Observation(ABC):

    def __init__(self,
                something: float):
        self.something = something
    
    def get_observation(self, airspace: Airspace, flights: List[Flight], normalize: Optional[bool] = False) -> List[np.ndarray]:
        pass

    def normalize_observation(self, observation: List[np.ndarray]) -> List[np.ndarray]:
        pass

    