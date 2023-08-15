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

        return observation
        

    def normalize_observation(self, observation: List[np.ndarray]) -> List[np.ndarray]:
        pass

    def add_normalization_data(self):
        pass

    def create_observation_vectors(self, flights: List[Flight]) -> np.ndarray:
        """ Uses a list of all flight objects to construct the observation vector
        for all flights from a local perspective. 
        
        Priority for the creation of the
        state vector is based on the distance between the aircraft, with lowest
        distance having the highest priority.

        If not enough flights are present to create a full observation vector
        the missing values will be padded with zeroes.

        Observation vector for each aircraft (each row) is:
            [airspeed,
            sin(drift),
            cos(drift),
            relative x position * number of aircraft in the state,
            relative y position * number of aircraft in the state,
            relative x velocity * number of aircraft in the state,
            relative y velocity * number of aircraft in the state,
            sin(relative track) * number of aircraft in the state,
            cos(relative track) * number of aircraft in the state,
            relative distance * number of aircraft in the state]

        Parameters
        __________
        flights: List[Flight]
            list of all the flight objects currently present in the airspace
        
        Returns
        __________
        observation: numpy array
            2D matrix, containing in each row the observation vector for a given flight
        """
        
        dx_rel, dy_rel, distances, vx, vy, d_track = self.get_relative_observation(flights)

        # Sort all arrays on minimum distances observed in the rows, pad with zeroes to ensure filled vector
        min_dist_index = np.argsort(distances)
        dx_rel = np.hstack((np.take_along_axis(dx_rel, min_dist_index, axis=1),np.zeros((len(flights),self.num_ac_state))))
        dy_rel = np.hstack((np.take_along_axis(dy_rel, min_dist_index, axis=1),np.zeros((len(flights),self.num_ac_state))))
        vx = np.hstack((np.take_along_axis(vx, min_dist_index, axis=1),np.zeros((len(flights),self.num_ac_state))))
        vy = np.hstack((np.take_along_axis(vy, min_dist_index, axis=1),np.zeros((len(flights),self.num_ac_state))))
        d_track = np.take_along_axis(d_track, min_dist_index, axis=1)
        sin_track = np.hstack((np.sin(d_track),np.zeros((len(flights),self.num_ac_state))))
        cos_track = np.hstack((np.cos(d_track),np.zeros((len(flights),self.num_ac_state))))
        distances = np.hstack((np.take_along_axis(distances, min_dist_index, axis=1),np.zeros((len(flights),self.num_ac_state))))

        # Create own information vectors
        airspeed = np.array([f.airspeed for f in flights])[:,np.newaxis]
        sin_drift = np.sin(np.array([f.drift for f in flights])[:,np.newaxis])
        cos_drift = np.cos(np.array([f.drift for f in flights])[:,np.newaxis])

        observation = np.hstack((airspeed,
                                 sin_drift,
                                 cos_drift,
                                 dx_rel[:,0:self.num_ac_state],
                                 dy_rel[:,0:self.num_ac_state],
                                 vx[:,0:self.num_ac_state],
                                 vy[:,0:self.num_ac_state],
                                 sin_track[:,0:self.num_ac_state],
                                 cos_track[:,0:self.num_ac_state],
                                 distances[:,0:self.num_ac_state]))
        
        if len(observation[0]) != self.observation_size:
            raise Exception(f"Observation vector contains {len(observation[0])} elements, but {self.observation_size} elements where specified in init")

        return observation

    def get_relative_observation(self, flights: List[Flight]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                                       np.ndarray, np.ndarray, np.ndarray]:
        """ Returns 6 matrices, dx_rel, dy_rel, distances, vx, vy and d_track, 
        these are the relative state arrays for all the aircraft in the airspace
        excluding the states relative to the ownship. (e.g. diagonals are removed)

        Indexing [i,j] gives corresponding state of j relative to i.

        Parameters
        __________
        flights: List[Flight]
            list of all the flight objects currently present in the airspace
        
        Returns
        __________
        dx_rel: numpy array
            2D numpy array with all the relative x positions w.r.t direction of flight

        dy_rel: numpy array
            2D numpy array with all the relative y positions w.r.t direction of flight
        
        distances: numpy array
            2D numpy array with all the relative distances between the aircraft
        
        vx: numpy array
            2D numpy array with all the relative x velocities w.r.t direction of flight

        vy: numpy array
            2D numpy array with all the relative y velocities w.r.t direction of flight

        d_track: numpy array
            2D numpy array with all the relative track angles of the aircraft in radians
        """

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
        """ Returns 3 square matrices, dx, dy and distances, where dx and dy are the 
        relative x and y positions in the earth references frame. distances is the 
        relative distance between all the aircraft.

        Indexing [i,j] gives speed of j relative to i.

        Parameters
        __________
        x: numpy array
            1D numpy array with all of the x positions in earth reference frame
        
        y: numpy array
            1D numpy array with all of the y positions in earth reference frame
        
        Returns
        __________
        dx: numpy array
            2D numpy array with all the relative x positions in the earth reference frame

        dy: numpy array
            2D numpy array with all the relative y positions in the earth reference frame
        
        distances: numpy array
            2D numpy array with all the relative distances between the aircraft
        """
        
        x_i = x[:, np.newaxis]
        x_j = x[np.newaxis, :]

        y_i = y[:, np.newaxis]
        y_j = y[np.newaxis, :]

        dx = x_j - x_i
        dy = y_j - y_i

        distances = np.sqrt((x_i-x_j)**2+(y_i-y_j)**2)

        return dx, dy, distances
    
    def get_relative_xy(self, dx: np.ndarray, dy: np.ndarray, tracks: np.ndarray, distances: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Returns 2 square matrices, dx_rel and dy_rel, with for each row the distances 
        of all aircraft, with positive y in the direction of flight.

        Indexing [i,j] gives speed of j relative to i.

        Parameters
        __________
        dx: numpy array
            2D numpy array with all of the relative x positions in earth reference frame
        
        dy: numpy array
            2D numpy array with all of the relative y positions in earth reference frame

        tracks: numpy array
            1D numpy array with all the track angles in radians, 0rad north.
        
        distances: numpy array
            2D numpy array with all the relative distances between the aircraft
        
        Returns
        __________
        dx_rel: numpy array
            2D numpy array with all the relative x positions w.r.t direction of flight

        dy_rel: numpy array
            2D numpy array with all the relative y positions w.r.t direction of flight
        """

        bearing = np.arctan2(dx, dy)                         
        bearing_rel = bearing - tracks[:,np.newaxis]

        dx_rel = np.sin(bearing_rel) * distances # Coordinate vectors orthogonal to direction of flight
        dy_rel = np.cos(bearing_rel) * distances # Coordinate vectors in direction of flight

        return dx_rel, dy_rel
        
    def get_relative_vxvy(self, v: np.ndarray, tracks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Returns 2 square matrices, vx and vy, with for each row the velocities of 
        all aircraft, with positive y in the direction of flight.

        Indexing [i,j] gives speed of j relative to i.

        Parameters
        __________
        v: numpy array
            1D numpy array with all of the velocities of all aircraft

        tracks: numpy array
            1D numpy array with all the track angles in radians, 0rad north.
        
        Returns
        __________
        vx: numpy array
            2D numpy array with all the relative x velocities w.r.t direction of flight

        vy: numpy array
            2D numpy array with all the relative y velocities w.r.t direction of flight
        """
        
        tracks_rel = tracks - tracks[:,np.newaxis]
        
        vx = v * np.sin(tracks_rel)  # Velocity vectors orthogonal to direction of flight
        vy = (v * np.cos(tracks_rel)) - v[:,np.newaxis]  # Velocity vectors in direction of flight

        return vx, vy


class Global(Observation):

    def get_observation(self, flights: List[Flight]) -> List[np.ndarray]:
        return super().get_observation(flights)

