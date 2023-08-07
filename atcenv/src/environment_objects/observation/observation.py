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

class Observation(ABC):

    def __init__(self,
                 observation_size: int,
                 normalize_data: bool,
                 create_normalization_data: Optional[bool] = False,
                 normalization_data_dir: Optional[str] = None,
                 normalization_samples: Optional[int] = 250000):
        self.observation_size = observation_size
        self.normalize_data = normalize_data
        self.create_normalization_data = create_normalization_data
        self.normalization_data_dir = normalization_data_dir
        self.normalization_samples = normalization_samples

        self.means = np.zeros(self.observation_size)
        self.stds = np.ones(self.observation_size)

        self.load_normalization_data()

    @abstractmethod
    def get_observation(self, flights: List[Flight]) -> List[np.ndarray]:
        pass
    
    @abstractmethod
    def normalize_observation(self, observation: List[np.ndarray]) -> List[np.ndarray]:
        pass
    
    @abstractmethod
    def add_normalization_data(self, observation: List[np.ndarray]) -> None:
        pass

    def load_normalization_data(self) -> None:

        if not self.normalize_data and not self.create_normalization_data:
            return
        
        if self.normalization_data_dir is None:
            raise('Cannot normalize data or create normalization dataset if no path is provided')
        
        if self.normalize_data:
            if self.create_normalization_data:
                print('Should not be normalizing data while data set has to be generated.')
                print('Setting create_normalization_data to False')
                self.create_normalization_data = False

            exist = fn.check_dir_exist(self.normalization_data_dir, False)
            if not exist:
                raise('No normalization data found, please run a create normalization data scenario first')
            else:
                with open(f'{self.normalization_data_dir}/normalization_data.p', 'rb') as handle:
                    self.means, self.stds = pickle.load(handle)

        elif self.create_normalization_data:
            exist = fn.check_dir_exist(self.normalization_data_dir, True)
            if not exist:
                print('Provided path to folder did not yet exist, creating path')

    def save_normalization_data(self) -> None:
        pass

class LocalObservation(Observation):

    def __init__(self,
                 num_ac_state: Optional[int] = 2):
        super().__init__()
        self.num_ac_state = 2

    def get_observation(self, flights: List[Flight]) -> List[np.ndarray]:

        

        if self.normalize_data:
            observation = self.normalize_observation(observation)
        if self.create_normalization_data:
            self.add_normalization_data(observation)
        

    def normalize_observation(self, observation: List[np.ndarray]) -> List[np.ndarray]:
        pass

    def add_normalization_data(self):
        pass

class GlobalObservation(Observation):

