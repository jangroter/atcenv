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
    """ Observation class

    Attributes
    ___________
    observation_size: int
        number of elements in each observation vector
    normalize_data: bool
        boolean variable on whether or not to normalize the observation vector
        actual normalization depends on the implemented method.
    create_normalization_data: bool
        boolean variable for using new data to construct the mean and std used
        for normalization
    normalization_data_dir: string
        directory to load from or save to for the normalization data
    normalization_samples: int
        number of samples required for constructing the normalization data

    Methods
    ___________
    get_observation(self, flights) -> observation 
        abstract method that given a list of flights returns 
        the observation dictionary, containing an observation vector
        and any other required information specific for the implementation
    normalize_observation(self, observation) -> observation
        abstract method that given an observation, normalizes it
    add_normalization_data(self, observation) -> None
        abstract method that stores the observation in an array for
        constructing the mean and std of the samples
    load_normalization_data(self) -> None
        loads the normalization data into self.means and self.stds
    save_normalization_data(self) -> None
        saves the means and stds to a pickle file when enough samples 
        have been obtained

    """

    def __init__(self,
                 observation_size: int,
                 normalize_data: bool,
                 create_normalization_data: Optional[bool] = False,
                 normalization_data_file: Optional[str] = None,
                 normalization_samples: Optional[int] = 20000):
        self.observation_size = observation_size
        self.normalize_data = normalize_data
        self.create_normalization_data = create_normalization_data
        self.normalization_data_file = normalization_data_file
        self.normalization_samples = normalization_samples

        self.means = np.zeros(self.observation_size)
        self.stds = np.ones(self.observation_size)
        
        self.normalization_counter = 0
        
        self.normalization_data = None
        self.load_normalization_data()

    @abstractmethod
    def get_observation(self, flights: List[Flight]) -> dict:
        """ return an observation vector for all flights in "flights"

        Parameters
        __________
        flights: List[Flight]
            list of Flight objects for which the observation vector needs to be constructed
        
        Returns
        __________
        observation: dict
            dictionary where observation["observation"] contains a numpy array with all the
            observations, any other information you might wish to pass can be given under
            personally defined keys
        """
        pass
    
    @abstractmethod
    def normalize_observation(self, observation: np.ndarray) -> np.ndarray:
        """ returns a normalized version of the input observation array

        Parameters
        __________
        observation: numpy array
            observation array to normalize
        
        Returns
        __________
        observation: numpy array
            normalized observation array
        """
        pass
    
    @abstractmethod
    def add_normalization_data(self, observation: np.ndarray) -> None:
        """ add observation data to a memory that can be used to 
        construct the mean and standard deviation of each sample for normalization

        Parameters
        __________
        observation: numpy array
            observation array to store in the memory

        Returns
        __________
        None

        """
        pass

    def load_normalization_data(self) -> None:
        """ loads normalization data into self.means and self.stds
        according to the data provided in normalization_data_dir

        Parameters
        __________
        None
        
        Returns
        __________
        None

        """

        if not self.normalize_data and not self.create_normalization_data:
            return
        
        if self.normalization_data_file is None:
            raise Exception('Cannot normalize data or create normalization dataset if no path is provided')
        
        if self.normalize_data:
            if self.create_normalization_data:
                print('Should not be normalizing data while data set has to be generated.')
                print('Setting create_normalization_data to False')
                self.create_normalization_data = False

            exist = fn.check_dir_exist(f'atcenv/src/observation/normalization_data/{self.normalization_data_file}', False)
            if not exist:
                raise Exception('No normalization data found, please run a create normalization data scenario first')
            else:
                with open(f'atcenv/src/observation/normalization_data/{self.normalization_data_file}', 'rb') as handle:
                    self.means, self.stds = pickle.load(handle)

                    print('means: ', self.means)
                    print('stds: ', self.stds)

    def save_normalization_data(self) -> None:
        """ construct the means and standard deviations generated 
        through add_normalization_data and saves them into normalization_data_dir

        Parameters
        __________
        None
        
        Returns
        __________
        None

        """
        
        means = np.mean(self.normalization_data, axis=1)
        stds = np.std(self.normalization_data, axis=1)

        with open(f'atcenv/src/observation/normalization_data/{self.normalization_data_file}', 'wb') as handle:
            pickle.dump((means,stds), handle, protocol=pickle.HIGHEST_PROTOCOL)

class BasicRel(Observation):
    
    def __init__(self,
                 observation_size: int,
                 normalize_data: bool,
                 num_ac_state: Optional[int] = 3,
                 create_normalization_data: Optional[bool] = False,
                 normalization_data_file: Optional[str] = None,
                 normalization_samples: Optional[int] = 20000):
        super().__init__(observation_size, normalize_data, create_normalization_data, 
                         normalization_data_file, normalization_samples)
        self.num_ac_state = num_ac_state
        
        # Basic uses 10 unique state parameters that require normalization
        self.normalization_data = np.zeros((10,normalization_samples))
    
    def get_observation(self, flights: List[Flight]) -> dict:
        observation = self.create_observation_vectors(flights)

        if self.normalize_data:
            observation = self.normalize_observation(observation)
        if self.create_normalization_data:
            self.add_normalization_data(observation)
        
        observation = {"observation": observation}

        return observation

    def create_observation_vectors(self, flights: List[Flight]) -> np.ndarray:
        """ Uses a list of all flight objects to construct the observation vector
        for all flights from a local perspective. 

        Different from Local, all elements in Basic are in the Earth reference
        frame instead of the flight reference frame.
        
        Priority for the creation of the
        state vector is based on the distance between the aircraft, with lowest
        distance having the highest priority.

        If not enough flights are present to create a full observation vector
        the missing values will be padded with zeroes.

        Observation vector for each aircraft (each row) is:
            [cos(drift),
            sin(drift),
            x,
            y,
            vx,
            vy,
            relative x * number of aircraft in the state,
            relative y * number of aircraft in the state,
            relative vx * number of aircraft in the state,
            relative vy * number of aircraft in the state]

        Parameters
        __________
        flights: List[Flight]
            list of all the flight objects currently present in the airspace
        
        Returns
        __________
        observation: numpy array
            2D matrix, containing in each row the observation vector for a given flight
        """
        
        drift = np.array([[np.sin(f.drift), np.cos(f.drift)] for f in flights])
        x = np.array([f.position.x for f in flights])
        y = np.array([f.position.y for f in flights])
        vx = np.array([f.components[0] for f in flights])
        vy = np.array([f.components[1] for f in flights])

        dx, dy, distances = fn.get_distance_matrices(x, y)
        dvx = vx[np.newaxis, :] - vx[:, np.newaxis]
        dvy = vy[np.newaxis, :] - vy[:, np.newaxis]

        distances = fn.remove_diagonal(distances)
        dx = fn.remove_diagonal(dx)
        dy = fn.remove_diagonal(dy)
        dvx = fn.remove_diagonal(dvx)
        dvy = fn.remove_diagonal(dvy)

        min_dist_index = np.argsort(distances)
        dx = np.hstack((np.take_along_axis(dx, min_dist_index, axis=1),np.zeros((len(flights),self.num_ac_state))))
        dy = np.hstack((np.take_along_axis(dy, min_dist_index, axis=1),np.zeros((len(flights),self.num_ac_state))))
        dvx = np.hstack((np.take_along_axis(dvx, min_dist_index, axis=1),np.zeros((len(flights),self.num_ac_state))))
        dvy = np.hstack((np.take_along_axis(dvy, min_dist_index, axis=1),np.zeros((len(flights),self.num_ac_state))))

        x = x[:,np.newaxis]
        y = y[:,np.newaxis]
        vx = vx[:,np.newaxis]
        vy = vy[:,np.newaxis]

        observation = np.hstack((drift,
                                 x,
                                 y,
                                 vx,
                                 vy,
                                 dx[:,0:self.num_ac_state],
                                 dy[:,0:self.num_ac_state],
                                 dvx[:,0:self.num_ac_state],
                                 dvy[:,0:self.num_ac_state]))

        if len(observation[0]) != self.observation_size:
            raise Exception(f"Observation vector contains {len(observation[0])} elements, but {self.observation_size} elements where specified in init")

        return observation
    
    def normalize_observation(self, observation: np.ndarray) -> np.ndarray:

        observation[:,2] /= self.stds[2]  # x
        observation[:,3] /= self.stds[3]  # y
        observation[:,4] /= self.stds[4]  # vx
        observation[:,5] /= self.stds[5]  # vy
        observation[:,6+self.num_ac_state*0:6+self.num_ac_state*1] /= self.stds[6]  # relative x position
        observation[:,6+self.num_ac_state*1:6+self.num_ac_state*2] /= self.stds[7]  # relative y position
        observation[:,6+self.num_ac_state*2:6+self.num_ac_state*3] /= self.stds[8]  # relative x velocity
        observation[:,6+self.num_ac_state*3:6+self.num_ac_state*4] /= self.stds[9]  # relative y velocity


        return observation
    
    def add_normalization_data(self, observation: np.ndarray) -> None:

        new_data = np.zeros((10,1))

        new_data[0] = np.mean(observation[:,0])  # sin(drift)
        new_data[1] = np.mean(observation[:,1])  # cos(drift)
        new_data[2] = np.mean(observation[:,2])  # x
        new_data[3] = np.mean(observation[:,3])  # y
        new_data[4] = np.mean(observation[:,4])  # vx
        new_data[5] = np.mean(observation[:,5])  # vy
        new_data[6] = np.mean(observation[:,6+self.num_ac_state*0:6+self.num_ac_state*1])  # relative x position
        new_data[7] = np.mean(observation[:,6+self.num_ac_state*1:6+self.num_ac_state*2])  # relative y position
        new_data[8] = np.mean(observation[:,6+self.num_ac_state*2:6+self.num_ac_state*3])  # relative x velocity
        new_data[9] = np.mean(observation[:,6+self.num_ac_state*3:6+self.num_ac_state*4])  # relative y velocity


        self.normalization_data[:,self.normalization_counter] = new_data[:,0]
        self.normalization_counter += 1
        
        if self.normalization_counter >= self.normalization_samples:
            self.save_normalization_data()
            print("finished generating normalization data, setting self.create_normalization_data to False")
            self.create_normalization_data = False

class BasicAbs(Observation):
    
    def __init__(self,
                 observation_size: int,
                 normalize_data: bool,
                 num_ac_state: Optional[int] = 3,
                 create_normalization_data: Optional[bool] = False,
                 normalization_data_file: Optional[str] = None,
                 normalization_samples: Optional[int] = 20000):
        super().__init__(observation_size, normalize_data, create_normalization_data, 
                         normalization_data_file, normalization_samples)
        self.num_ac_state = num_ac_state
        
        # Basic uses 10 unique state parameters that require normalization
        self.normalization_data = np.zeros((6,normalization_samples))
    
    def get_observation(self, flights: List[Flight]) -> dict:
        observation = self.create_observation_vectors(flights)

        if self.normalize_data:
            observation = self.normalize_observation(observation)
        if self.create_normalization_data:
            self.add_normalization_data(observation)
        
        observation = {"observation": observation}

        return observation

    def create_observation_vectors(self, flights: List[Flight]) -> np.ndarray:
        """ Uses a list of all flight objects to construct the observation vector
        for all flights from a local perspective. 

        Different from Local, all elements in Basic are in the Earth reference
        frame instead of the flight reference frame.
        
        Priority for the creation of the
        state vector is based on the distance between the aircraft, with lowest
        distance having the highest priority.

        If not enough flights are present to create a full observation vector
        the missing values will be padded with zeroes.

        Observation vector for each aircraft (each row) is:
            [cos(drift),
            sin(drift),
            x,
            y,
            vx,
            vy,
            x_int * number of aircraft in the state,
            y_int * number of aircraft in the state,
            vx_int * number of aircraft in the state,
            vy_int * number of aircraft in the state]

        Parameters
        __________
        flights: List[Flight]
            list of all the flight objects currently present in the airspace
        
        Returns
        __________
        observation: numpy array
            2D matrix, containing in each row the observation vector for a given flight
        """
        
        drift = np.array([[np.sin(f.drift), np.cos(f.drift)] for f in flights])
        x = np.array([f.position.x for f in flights])
        y = np.array([f.position.y for f in flights])
        vx = np.array([f.components[0] for f in flights])
        vy = np.array([f.components[1] for f in flights])

        dx, dy, distances = fn.get_distance_matrices(x, y)
        dvx = vx[np.newaxis, :] - vx[:, np.newaxis]
        dvy = vy[np.newaxis, :] - vy[:, np.newaxis]


        x = np.array([x for f in flights])
        y = np.array([y for f in flights])
        vx = np.array([vx for f in flights])
        vy = np.array([vy for f in flights])

        min_dist_index = np.argsort(distances)
        x = np.hstack((np.take_along_axis(x, min_dist_index, axis=1),np.zeros((len(flights),self.num_ac_state))))
        y = np.hstack((np.take_along_axis(y, min_dist_index, axis=1),np.zeros((len(flights),self.num_ac_state))))
        vx = np.hstack((np.take_along_axis(vx, min_dist_index, axis=1),np.zeros((len(flights),self.num_ac_state))))
        vy = np.hstack((np.take_along_axis(vy, min_dist_index, axis=1),np.zeros((len(flights),self.num_ac_state))))

        observation = np.hstack((drift,
                                 x[:,0:self.num_ac_state+1],
                                 y[:,0:self.num_ac_state+1],
                                 vx[:,0:self.num_ac_state+1],
                                 vy[:,0:self.num_ac_state+1]))

        if len(observation[0]) != self.observation_size:
            raise Exception(f"Observation vector contains {len(observation[0])} elements, but {self.observation_size} elements where specified in init")

        return observation
    
    def normalize_observation(self, observation: np.ndarray) -> np.ndarray:

        observation[:,2+(self.num_ac_state + 1)*0:6+(self.num_ac_state + 1)*1] /= self.stds[2]  # x position
        observation[:,2+(self.num_ac_state + 1)*1:6+(self.num_ac_state + 1)*2] /= self.stds[3]  # y position
        observation[:,2+(self.num_ac_state + 1)*2:6+(self.num_ac_state + 1)*3] /= self.stds[4]  # x velocity
        observation[:,2+(self.num_ac_state + 1)*3:6+(self.num_ac_state + 1)*4] /= self.stds[5]  # y velocity

        return observation
    
    def add_normalization_data(self, observation: np.ndarray) -> None:

        new_data = np.zeros((6,1))

        new_data[0] = np.mean(observation[:,0])  # sin(drift)
        new_data[1] = np.mean(observation[:,1])  # cos(drift)
        new_data[2] = np.mean(observation[:,2+(self.num_ac_state + 1)*0:2+(self.num_ac_state + 1)*1])  # x position
        new_data[3] = np.mean(observation[:,2+(self.num_ac_state + 1)*1:2+(self.num_ac_state + 1)*2])  # y position
        new_data[4] = np.mean(observation[:,2+(self.num_ac_state + 1)*2:2+(self.num_ac_state + 1)*3])  # x velocity
        new_data[5] = np.mean(observation[:,2+(self.num_ac_state + 1)*3:2+(self.num_ac_state + 1)*4])  # y velocity


        self.normalization_data[:,self.normalization_counter] = new_data[:,0]
        self.normalization_counter += 1
        
        if self.normalization_counter >= self.normalization_samples:
            self.save_normalization_data()
            print("finished generating normalization data, setting self.create_normalization_data to False")
            self.create_normalization_data = False

class Local(Observation):
    """ Local observation class, inherits from Observation

    This observation class uses local information relative from the 
    perspective of the ownship to construct the observation vector.
    This includes, position, speeds and relative heading (redundant with speeds)

    Attributes
    ___________
    observation_size: int
        number of elements in each observation vector
    normalize_data: bool
        boolean variable on whether or not to normalize the observation vector
        actual normalization depends on the implemented method.
    num_ac_state: int
        number of aircraft to include in the state vector
    create_normalization_data: bool
        boolean variable for using new data to construct the mean and std used
        for normalization
    normalization_data_dir: string
        directory to load from or save to for the normalization data
    normalization_samples: int
        number of samples required for constructing the normalization data
    

    Methods
    ___________
    get_observation(self, flights) -> observation 
        abstract method that given a list of flights returns 
        the observation vector for all flights
    normalize_observation(self, observation) -> observation
        abstract method that given an observation, normalizes it
    add_normalization_data(self, observation) -> None
        abstract method that stores the observation in an array for
        constructing the mean and std of the samples
    load_normalization_data(self) -> None
        loads the normalization data into self.means and self.stds
    save_normalization_data(self) -> None
        saves the means and stds to a pickle file when enough samples 
        have been obtained

    """

    def __init__(self,
                 observation_size: int,
                 normalize_data: bool,
                 num_ac_state: Optional[int] = 2,
                 create_normalization_data: Optional[bool] = False,
                 normalization_data_file: Optional[str] = None,
                 normalization_samples: Optional[int] = 20000):
        super().__init__(observation_size, normalize_data, create_normalization_data, 
                         normalization_data_file, normalization_samples)
        self.num_ac_state = num_ac_state
        
        # Local uses 10 unique state parameters that require normalization
        self.normalization_data = np.zeros((10,normalization_samples))

    def get_observation(self, flights: List[Flight]) -> dict:
        observation = self.create_observation_vectors(flights)

        if self.normalize_data:
            observation = self.normalize_observation(observation)
        if self.create_normalization_data:
            self.add_normalization_data(observation)
        
        observation = {"observation": observation}

        return observation
        
    def normalize_observation(self, observation: np.ndarray) -> np.ndarray:

        observation[:,0] -= self.means[0]  # Airspeed
        observation[:,3+self.num_ac_state*6:3+self.num_ac_state*7] -= self.means[9]  # relative distance

        observation[:,0] /= self.stds[0]  # Airspeed
        observation[:,3+self.num_ac_state*0:3+self.num_ac_state*1] /= self.stds[3]  # relative x position
        observation[:,3+self.num_ac_state*1:3+self.num_ac_state*2] /= self.stds[4]  # relative y position
        observation[:,3+self.num_ac_state*2:3+self.num_ac_state*3] /= self.stds[5]  # relative x velocity
        observation[:,3+self.num_ac_state*3:3+self.num_ac_state*4] /= self.stds[6]  # relative y velocity
        observation[:,3+self.num_ac_state*6:3+self.num_ac_state*7] /= self.stds[9]  # relative distance

        return observation

    def add_normalization_data(self, observation: np.ndarray) -> None:

        new_data = np.zeros((10,1))

        new_data[0] = np.mean(observation[:,0])  # Airspeed
        new_data[1] = np.mean(observation[:,1])  # sin(drift)
        new_data[2] = np.mean(observation[:,2])  # cos(drift)
        new_data[3] = np.mean(observation[:,3+self.num_ac_state*0:3+self.num_ac_state*1])  # relative x position
        new_data[4] = np.mean(observation[:,3+self.num_ac_state*1:3+self.num_ac_state*2])  # relative y position
        new_data[5] = np.mean(observation[:,3+self.num_ac_state*2:3+self.num_ac_state*3])  # relative x velocity
        new_data[6] = np.mean(observation[:,3+self.num_ac_state*3:3+self.num_ac_state*4])  # relative y velocity
        new_data[7] = np.mean(observation[:,3+self.num_ac_state*4:3+self.num_ac_state*5])  # relative sin(track)
        new_data[8] = np.mean(observation[:,3+self.num_ac_state*5:3+self.num_ac_state*6])  # relative cos(track)
        new_data[9] = np.mean(observation[:,3+self.num_ac_state*6:3+self.num_ac_state*7])  # relative distance

        self.normalization_data[:,self.normalization_counter] = new_data[:,0]
        self.normalization_counter += 1
        
        if self.normalization_counter >= self.normalization_samples:
            self.save_normalization_data()
            print("finished generating normalization data, setting self.create_normalization_data to False")
            self.create_normalization_data = False
        
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
        dx, dy, distances = fn.get_distance_matrices(x, y)

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

class LocalNoTrack(Observation):
    """ Local observation class, inherits from Observation

    This observation class uses local information relative from the 
    perspective of the ownship to construct the observation vector.
    This includes position and speeds

    Attributes
    ___________
    observation_size: int
        number of elements in each observation vector
    normalize_data: bool
        boolean variable on whether or not to normalize the observation vector
        actual normalization depends on the implemented method.
    num_ac_state: int
        number of aircraft to include in the state vector
    create_normalization_data: bool
        boolean variable for using new data to construct the mean and std used
        for normalization
    normalization_data_dir: string
        directory to load from or save to for the normalization data
    normalization_samples: int
        number of samples required for constructing the normalization data
    

    Methods
    ___________
    get_observation(self, flights) -> observation 
        abstract method that given a list of flights returns 
        the observation vector for all flights
    normalize_observation(self, observation) -> observation
        abstract method that given an observation, normalizes it
    add_normalization_data(self, observation) -> None
        abstract method that stores the observation in an array for
        constructing the mean and std of the samples
    load_normalization_data(self) -> None
        loads the normalization data into self.means and self.stds
    save_normalization_data(self) -> None
        saves the means and stds to a pickle file when enough samples 
        have been obtained

    """

    def __init__(self,
                 observation_size: int,
                 normalize_data: bool,
                 num_ac_state: Optional[int] = 2,
                 create_normalization_data: Optional[bool] = False,
                 normalization_data_file: Optional[str] = None,
                 normalization_samples: Optional[int] = 20000):
        super().__init__(observation_size, normalize_data, create_normalization_data, 
                         normalization_data_file, normalization_samples)
        self.num_ac_state = num_ac_state
        
        # Local uses 10 unique state parameters that require normalization
        self.normalization_data = np.zeros((10,normalization_samples))

    def get_observation(self, flights: List[Flight]) -> dict:
        observation = self.create_observation_vectors(flights)

        if self.normalize_data:
            observation = self.normalize_observation(observation)
        if self.create_normalization_data:
            self.add_normalization_data(observation)
        
        observation = {"observation": observation}

        return observation
        
    def normalize_observation(self, observation: np.ndarray) -> np.ndarray:

        observation[:,0] -= self.means[0]  # Airspeed
        observation[:,3+self.num_ac_state*6:3+self.num_ac_state*7] -= self.means[9]  # relative distance

        observation[:,0] /= self.stds[0]  # Airspeed
        observation[:,3+self.num_ac_state*0:3+self.num_ac_state*1] /= self.stds[3]  # relative x position
        observation[:,3+self.num_ac_state*1:3+self.num_ac_state*2] /= self.stds[4]  # relative y position
        observation[:,3+self.num_ac_state*2:3+self.num_ac_state*3] /= self.stds[5]  # relative x velocity
        observation[:,3+self.num_ac_state*3:3+self.num_ac_state*4] /= self.stds[6]  # relative y velocity
        observation[:,3+self.num_ac_state*4:3+self.num_ac_state*5] /= self.stds[9]  # relative distance

        return observation

    def add_normalization_data(self, observation: np.ndarray) -> None:
        print("Not implemented, uses 'Local' instead of LocalNoTrack to generate data.")
        self.create_normalization_data = False
        
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
        dx, dy, distances = fn.get_distance_matrices(x, y)

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

    def __init__(self,
                 observation_size: int,
                 normalize_data: bool,
                 create_normalization_data: Optional[bool] = False,
                 normalization_data_file: Optional[str] = None,
                 normalization_samples: Optional[int] = 20000):
        super().__init__(observation_size, normalize_data, create_normalization_data, 
                         normalization_data_file, normalization_samples)
        
        # Global uses 7 unique state parameters that require normalization
        self.normalization_data = np.zeros((7,normalization_samples))

    def get_observation(self, flights: List[Flight]) -> dict:
        
        x = np.array([[f.position.x] for f in flights])
        y = np.array([[f.position.y] for f in flights])
        vxy = np.array([f.components for f in flights])
        v = np.array([[f.airspeed] for f in flights])
        drift = np.array([[np.sin(f.drift), np.cos(f.drift)] for f in flights])

        observation = np.hstack((x,
                                 y,
                                 vxy,
                                 v,
                                 drift))
        
        if self.normalize_data:
            observation = self.normalize_observation(observation)
        if self.create_normalization_data:
            self.add_normalization_data(observation)

        observation = {"observation": observation}

        return observation
        
    def normalize_observation(self, observation: np.ndarray) -> np.ndarray:
        observation[:,0] -= self.means[0]  # x
        observation[:,1] -= self.means[1]  # y
        observation[:,2] -= self.means[2]  # vx
        observation[:,3] -= self.means[3]  # vy
        observation[:,4] -= self.means[4]  # v

        observation[:,0] /= self.stds[0]  # x
        observation[:,1] /= self.stds[1]  # y
        observation[:,2] /= self.stds[2]  # vx
        observation[:,3] /= self.stds[3]  # vy
        observation[:,4] /= self.stds[4]  # v
        
        return observation

    def add_normalization_data(self, observation: np.ndarray) -> None:
        new_data = np.zeros((7,1))

        new_data[0] = np.mean(observation[:,0])  # x
        new_data[1] = np.mean(observation[:,1])  # y
        new_data[2] = np.mean(observation[:,2])  # vx
        new_data[3] = np.mean(observation[:,3])  # vy
        new_data[4] = np.mean(observation[:,4])  # v
        new_data[5] = np.mean(observation[:,5])  # sin(drift)
        new_data[6] = np.mean(observation[:,6])  # cos(drift)

        self.normalization_data[:,self.normalization_counter] = new_data[:,0]
        self.normalization_counter += 1
        
        if self.normalization_counter >= self.normalization_samples:
            self.save_normalization_data()
            print("finished generating normalization data, setting self.create_normalization_data to False")
            self.create_normalization_data = False
        pass

class GlobalTrack(Global):

    def get_observation(self, flights: List[Flight]) -> dict:
        
        x = np.array([[f.position.x] for f in flights])
        y = np.array([[f.position.y] for f in flights])
        vxy = np.array([f.components for f in flights])
        v = np.array([[f.airspeed] for f in flights])
        drift = np.array([[np.sin(f.drift), np.cos(f.drift)] for f in flights])
        track = np.array([[f.track] for f in flights])

        observation = np.hstack((x,
                                 y,
                                 vxy,
                                 v,
                                 drift,
                                 track))
        
        if self.normalize_data:
            observation = self.normalize_observation(observation)
        if self.create_normalization_data:
            self.add_normalization_data(observation)

        observation = {"observation": observation}

        return observation

class StateBasedMVP(Observation):

    def get_observation(self, flights: List[Flight]) -> dict:
        drift = np.array([[f.drift] for f in flights])
        v_dif = np.array([[f.optimal_airspeed - f.airspeed] for f in flights]) 

        v = np.array([[f.airspeed] for f in flights])
        tracks = np.array([[f.track] for f in flights])

        x = np.array([[f.position.x] for f in flights])
        y = np.array([[f.position.y] for f in flights])
        vx, vy = self.get_vxvy(v, tracks)

        observation = np.hstack((drift,
                                 v_dif,
                                 x,
                                 y,
                                 vx,
                                 vy))
        
        observation = {"observation": observation, "flights": flights}
        return observation
    
    def normalize_observation(self, observation: np.ndarray) -> np.ndarray:
        return observation

    def add_normalization_data(self, observation: np.ndarray) -> None:
        pass

    def get_vxvy(self, v: np.ndarray, tracks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        vx = v * np.sin(tracks) 
        vy = v * np.cos(tracks)
        return vx, vy