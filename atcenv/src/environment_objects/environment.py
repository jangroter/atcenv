from shapely.geometry import Point, Polygon
from typing import Tuple, Optional, List
import math
import random
import numpy as np
import pickle

from gym.envs.classic_control import rendering
from shapely.geometry import LineString

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import atcenv.src.functions as fn
import atcenv.src.units as u
from atcenv.src.environment_objects.flight import Flight
from atcenv.src.environment_objects.airspace import Airspace


class Environment(ABC):

    def __init__(self,
                 dt: Optional[float] = 5.,
                 max_episode_len: int = 150,
                 max_speed_change: Optional[float] = 50.,
                 max_heading_change: Optional[float] = 25.,
                 use_degrees: bool = True,
                 render_frequency: int = 0,
                 episode: int = 0):
        self.dt = dt
        self.max_episode_len = max_episode_len
        self.max_speed_change = max_speed_change
        self.max_heading_change = max_heading_change * (use_degrees * np.pi/180.)

        self.airspace = None
        self.flights = None
        self.done = False
        self.conflicts = set() 

        self.render_frequency = render_frequency
        self.episode = episode
        self.viewer = None

        self.counter = 0

    def create_environment(self, airspace: Airspace, flights: List[Flight], episode: int = 0):
        self.airspace = airspace
        self.flights = flights
        self.episode = episode
        self.done = False
        self.counter = 0

    @abstractmethod
    def step(self, action: np.ndarray) -> bool:
        pass

    def update_conflicts(self) -> None:

        self.conflicts = set()

        for i in range(len(self.flights)):
            for j in range(len(self.flights)):
                distance = self.flights[i].position.distance(self.flights[j].position)
                if distance < self.flights[i].aircraft.min_distance and i != j:
                    self.conflicts.update((i, j))

    def render(self, mode=None) -> None:

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

    def close(self) -> None:

        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

class DefaultEnvironment(Environment):

    def step(self, action: np.ndarray) -> bool:

        self.counter += 1

        d_heading = np.clip(action[:,0], -self.max_heading_change, self.max_heading_change)
        d_velocity = np.clip(action[:,1], -self.max_speed_change, self.max_speed_change)
        
        for flight, dh, dv in zip(self.flights, d_heading, d_velocity):

            old_track = flight.track
            flight.track = ((flight.track + dh) + u.circle) % u.circle  # Bound the new track between 0 and 2*pi

            flight.airspeed = np.clip(flight.airspeed+dv, flight.aircraft.min_speed, flight.aircraft.max_speed)

            vx, vy = flight.components
            position = flight.position

            new_x = position.x + vx * self.dt
            new_y = position.y + vy * self.dt

            flight.position._set_coords(new_x, new_y)
        
        self.update_conflicts()

        if self.render_frequency != 0 and self.episode % self.render_frequency == 0:
            self.render()
        
        if self.counter > self.max_episode_len:
            self.done = True
            self.close()
        
        return self.done



