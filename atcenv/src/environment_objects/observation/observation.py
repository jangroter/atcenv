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

class Local(Observation):

    def __init__(self,
                 observation_size: int,
                 normalize_data: bool,
                 create_normalization_data: Optional[bool] = False,
                 normalization_data_dir: Optional[str] = None,
                 normalization_samples: Optional[int] = 250000,
                 num_ac_state: Optional[int] = 2):
        super().__init__(observation_size, normalize_data, create_normalization_data, 
                         normalization_data_dir, normalization_samples)
        self.num_ac_state = num_ac_state

    def get_observation(self, flights: List[Flight]) -> List[np.ndarray]:

        observation = self.create_observation_vectors(flights)

        if self.normalize_data:
            observation = self.normalize_observation(observation)
        if self.create_normalization_data:
            self.add_normalization_data(observation)
        

    def normalize_observation(self, observation: List[np.ndarray]) -> List[np.ndarray]:
        pass

    def add_normalization_data(self):
        pass

    def create_observation_vectors(self, flights: List[Flight]) -> np.ndarray:
        
        dx_rel, dy_rel, distances, vx, vy, d_track = self.get_observation_matrices(flights)

        # Sort all arrays on minimum distances observed in the rows
        min_dist_index = np.argsort(distances)
        dx_rel = np.take_along_axis(dx_rel, min_dist_index, axis=1)
        dy_rel = np.take_along_axis(dy_rel, min_dist_index, axis=1)
        vx = np.take_along_axis(vx, min_dist_index, axis=1)
        vy = np.take_along_axis(vy, min_dist_index, axis=1)
        d_track = np.take_along_axis(d_track, min_dist_index, axis=1)
        distances = np.take_along_axis(distances, min_dist_index, axis=1)

        airspeed = np.sin(np.array([f.airspeed for f in flights])[:,np.newaxis])
        sin_drift = np.sin(np.array([f.drift for f in flights])[:,np.newaxis])
        cos_drift = np.sin(np.array([f.drift for f in flights])[:,np.newaxis])


        observation = np.hstack((airspeed,
                                 sin_drift,
                                 cos_drift,
                                 dx_rel[:,0:self.num_ac_state],
                                 dy_rel[:,0:self.num_ac_state],
                                 vx[:,0:self.num_ac_state],
                                 vy[:,0:self.num_ac_state],
                                 d_track[:,0:self.num_ac_state],
                                 distances[:,0:self.num_ac_state]))
        
        if len(observation[0]) != self.observation_size:
            raise(f'Observation vector contains {observation[0]} elements, but {self.observation_size} elements where specified in init')

        return observation

    def get_observation_matrices(self, flights: List[Flight]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                                       np.ndarray, np.ndarray, np.ndarray]:

        x = np.array([f.position.x for f in flights])
        y = np.array([f.position.y for f in flights])
        dx, dy, distances = self.get_distance_matrices(x, y)

        tracks = np.array([f.track for f in flights])
        dx_rel, dy_rel = self.get_relative_xy(dx, dy, tracks, distances)

        v = np.array([f.airspeed for f in flights])
        vx, vy = self.get_relative_vxvy(v, tracks)

        d_track = tracks - tracks[:,np.newaxis]

        dx_rel = fn.remove_diagonal(dx_rel)
        dy_rel = fn.remove_diagonal(dy_rel)
        distances = fn.remove_diagonal(distances)
        vx = fn.remove_diagonal(vx)
        vy = fn.remove_diagonal(vy)
        d_track = fn.remove_diagonal(d_track)

        return dx_rel, dy_rel, distances, vx, vy, d_track

    def get_distance_matrices(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        x_i = x[:, np.newaxis]
        x_j = x[np.newaxis, :]

        y_i = y[:, np.newaxis]
        y_j = y[np.newaxis, :]

        dx = x_j - x_i
        dy = y_j - y_i

        distances = np.sqrt((x_i-x_j)**2+(y_i-y_j)**2)

        return dx, dy, distances
    
    def get_relative_xy(self, dx: np.ndarray, dy: np.ndarray, tracks: np.ndarray, distances: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        bearing = np.arctan2(dx, dy)                         
        bearing_rel = bearing - tracks[:,np.newaxis]

        dx_rel = np.sin(bearing_rel) * distances # Coordinate vectors orthogonal to direction of flight
        dy_rel = np.cos(bearing_rel) * distances # Coordinate vectors in direction of flight

        return dx_rel, dy_rel
        
    def get_relative_vxvy(self, v: np.ndarray, tracks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        
        tracks_rel = tracks - tracks[:,np.newaxis]
        
        vx = v * np.sin(tracks_rel)  # Velocity vectors orthogonal to direction of flight
        vy = (v * np.cos(tracks_rel)) - v[:,np.newaxis]  # Velocity vectors in direction of flight

        return vx, vy


class Global(Observation):

    def get_observation(self, flights: List[Flight]) -> List[np.ndarray]:
        return super().get_observation(flights)

