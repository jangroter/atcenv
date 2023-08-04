import gym
from typing import List, Dict, Tuple

class Environment():
    def __init__(self,
                 dt: float = 5,
                 max_area: float = 63. * 63., # maybe makes more sense under the Airspace class
                 min_area: float = 40. * 40., 
                 max_speed: float = 500., # maybe makes more sense under the Flight class
                 min_speed: float = 400.,
                 min_distance: float = 5.,
                 distance_init_buffer: float = 2.) -> None:

        self.dt = dt
        self.max_area = max_area
        self.min_area = min_area
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.min_distance = min_distance
        self.distance_init_buffer = distance_init_buffer
    
    def step(self, actions: List) -> None:
        """ Perform  a single simulation step in the environment.

        Parameters
        __________
        actions: list of actions to perform 
            action[n][0]: float, heading change for aircraft n in radians 
            action[n][1]: float, speed change for aircraft n in knots
        
        Returns
        __________
        None
        """
        pass

    def reset(self):
        pass

    def get_scenario(self):
        """ Should probably be a class that gets called to allow easy change of scenario types """
        pass

    def load_scenario(self):
        pass

    def create_scenario(self):
        pass

    def save_scenario(self):
        pass

    def get_observation(self):
        """ Should probably be a class that gets called to allow easy change of observation vector """
        pass

    def get_score(self):
        """ Should probably be a class that gets called to allow easy change of score calculation """
        pass

    def update_position(self):
        pass

    def update_velocity(self):
        pass

    def render_viewer(self):
        pass

    def close_viewer(self):
        pass


