from shapely.geometry import Point, Polygon
from typing import Tuple, Optional, List
import math
import random
import numpy as np
import pickle
import time

from gym.envs.classic_control import rendering
from shapely.geometry import LineString

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import atcenv.src.functions as fn
import atcenv.src.units as u
from atcenv.src.environment_objects.flight import Flight
from atcenv.src.environment_objects.airspace import Airspace


class Environment(ABC):
    """ Environment base class

    Attributes
    ___________
    dt: float
        timestep used for the simulations, s
    max_episode_len: int
        maximum number of timesteps that a single episode takes
    max_speed_change: float
        maximum speed change per timestep for each flight, m/s
    max_heading_change: float
        maximum heading change per timestep, rad or deg
    use_degrees: bool
        boolean variable for if max_heading_change is in rad or deg
    render_frequency: int
        how often a scenario will be rendered, 0 means never
    
    Methods
    ___________
    create_environment(self, airspace, flights, episode) -> None
        initializes a new environment
    step(self, action) -> bool
        progresses the simulation to the next state using the provided 
        actions and simulation timestep dt, returns a done flag
    update_conflicts(self) -> None
        updates the conflict set object of all aircraft 
        that are currently too close
    render(self) -> None
        updates the render window with the newest state
    close(self) -> None
        closes the render window

    """

    def __init__(self,
                 dt: Optional[float] = 5.,
                 max_episode_len: int = 150,
                 max_speed_change: Optional[float] = 50.,
                 max_heading_change: Optional[float] = 25.,
                 use_degrees: bool = True,
                 render_frequency: int = 0):
        self.dt = dt
        self.max_episode_len = max_episode_len
        self.max_speed_change = max_speed_change
        self.max_heading_change = max_heading_change * (use_degrees * np.pi/180.)

        self.airspace = None
        self.flights = None
        self.done = False
        self.conflicts = set() 

        self.render_frequency = render_frequency
        self.episode = None
        self.viewer = None

        self.counter = 0

    def create_environment(self, airspace: Airspace, flights: List[Flight], episode: int = 0) -> None:
        """ Creates a new environment based on the provided initial conditions

        Parameters
        __________
        airspace: Airspace
            Airspace object for this scenario
        flights: List[Flight]
            list of Flight objects for this scenario
        episode: int
            episode number, used for determining rendering
        
        
        Returns
        __________
        None

        """
        self.airspace = airspace
        self.flights = flights
        self.episode = episode
        self.done = False
        self.counter = 0

    @abstractmethod
    def step(self, action: np.ndarray, transform_action: bool) -> bool:
        """ Progresses the environment to the next state

        Parameters
        __________
        action: numpy array
            action array of size = (number of flights, number of actions)
        
        Returns
        __________
        done: bool
            boolean variable for the done flag, if True, episode has terminated
            
        """
        pass

    def update_conflicts(self) -> None:
        """ Updates set of currently active conflicts

        Loops through all flights currently in the environment and checks the
        relative distance. If the distance is smaller than the minimum distance for 
        either one of the aircraft, they are added to the conflict set.

        Parameters
        __________
        None

        Returns
        __________
        None

        """

        self.conflicts = set()

        for i in range(len(self.flights)):
            for j in range(len(self.flights)):
                distance = self.flights[i].position.distance(self.flights[j].position)
                if distance < self.flights[i].aircraft.min_distance and i != j:
                    self.conflicts.update((i, j))

    def render(self) -> None:
        """ Creates a render of the current state of the environment

        Parameters
        __________
        None
        
        Returns
        __________
        None

        """

        if self.viewer is None:
            # initialise viewer
            screen_width, screen_height = 600, 600

            minx, miny, maxx, maxy = self.airspace.polygon.buffer(10 * u.nm).bounds
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.viewer.set_bounds(minx, maxx, miny, maxy)

            # fill background
            background = rendering.make_polygon([(minx, miny),
                                                 (minx, maxy),
                                                 (maxx, maxy),
                                                 (maxx, miny)],
                                                filled=True)
            background.set_color(0, 0, 0)
            self.viewer.add_geom(background)

            # display airspace
            sector = rendering.make_polygon(self.airspace.polygon.boundary.coords, filled=False)
            sector.set_linewidth(1)
            sector.set_color(0, 255, 0)
            self.viewer.add_geom(sector)

        # add current positions
        for i, f in enumerate(self.flights):

            if i in self.conflicts:
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)

            circle = rendering.make_circle(radius=f.aircraft.min_distance / 2.0,
                                           res=10,
                                           filled=False)
            circle.add_attr(rendering.Transform(translation=(f.position.x,
                                                             f.position.y)))
            circle.set_color(0, 0, 255)

            plan = LineString([f.position, f.target])
            self.viewer.draw_polyline(plan.coords, linewidth=1, color=color)
            prediction = LineString([f.position, f.prediction])
            self.viewer.draw_polyline(prediction.coords, linewidth=4, color=color)

            self.viewer.add_onetime(circle)

        self.viewer.render()
        time.sleep(0.01)

    def close(self) -> None:
        """ Closes the current render

        Parameters
        __________
        None
        
        Returns
        __________
        None

        """
        if self.viewer is not None:
            try:
                self.viewer.close()
                self.viewer = None
            except AttributeError as e:
                pass

class DefaultEnvironment(Environment):
    """ Default environment, inherits from Environment

    Attributes
    ___________
    dt: float
        timestep used for the simulations, s
    max_episode_len: int
        maximum number of timesteps that a single episode takes
    max_speed_change: float
        maximum speed change per timestep for each flight, m/s
    max_heading_change: float
        maximum heading change per timestep, rad or deg
    use_degrees: bool
        boolean variable for if max_heading_change is in rad or deg
    render_frequency: int
        how often a scenario will be rendered, 0 means never
    
    Methods
    ___________
    create_environment(self, airspace, flights, episode) -> None
        initializes a new environment
    step(self, action) -> bool
        progresses the simulation to the next state using the provided 
        actions and simulation timestep dt, returns a done flag
    update_conflicts(self) -> None
        updates the conflict set object of all aircraft 
        that are currently too close
    render(self) -> None
        updates the render window with the newest state
    close(self) -> None
        closes the render window

    """

    def step(self, action: np.ndarray, transform_action: bool) -> bool:
        """ Progresses the environment to the next state

        This implementation uses direct propagation of the state
        and next state adaption. No dynamics involved.

        Parameters
        __________
        action: numpy array
            action array of size = (number of flights, number of actions)
        transform: boolean
            boolean variable on whether or not to map the input action to
            the environment action space
        
        Returns
        __________
        done: bool
            boolean variable for the done flag, if True, episode has terminated
            
        """
        self.counter += 1

        action = self.transform_action(action, transform_action)

        d_heading = np.clip(action[:,0], -self.max_heading_change, self.max_heading_change)
        d_velocity = np.clip(action[:,1], -self.max_speed_change, self.max_speed_change)
        
        for flight, dh, dv in zip(self.flights, d_heading, d_velocity):
            flight.track = ((flight.track + dh) + u.circle) % u.circle  # Bound the new track between 0 and 2*pi

            flight.airspeed = np.clip(flight.airspeed+dv, flight.aircraft.min_speed, flight.aircraft.max_speed)

            vx, vy = flight.components
            position = flight.position

            new_x = position.x + vx * self.dt
            new_y = position.y + vy * self.dt

            flight.position = Point(new_x,new_y)
        
        self.update_conflicts()

        if self.render_frequency != 0 and self.episode % self.render_frequency == 0:
            self.render()
        
        if self.counter >= self.max_episode_len:
            self.done = True
            self.close()
        
        return self.done
    
    def transform_action(self, action: np.ndarray, transform_action: bool) -> np.ndarray:

        if transform_action:
            action[:,0] *= self.max_heading_change
            action[:,1] *= self.max_speed_change
        return action





