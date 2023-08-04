from shapely.geometry import Point, Polygon
from typing import Tuple, Optional
import math
import random

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import atcenv.src.functions as fn

@dataclass
class Airspace(ABC):
    """ Airspace class

    Attributes
    ___________
    polygon: shapely.geomtry.Polygon 
        object that describes the airspace geometry
    category: string
        airspace category, determines the type of operations being flown

    Methods
    ___________
    random(min_area, max_area) -> Airspace object
        class method that returns a random Airspace object 
    random_flight(self) -> position, target, speed
        returns random initial conditions for a flight 
        in accordance with the airspace type
    get_waypoint(self, position, flight_type, target):
        returns a new waypoint for an aircraft in accordance
        with the airspace 

    """
    polygon: Optional[Polygon] = None
    category: Optional[str] = None

    @classmethod
    @abstractmethod
    def random(cls, min_area: float, max_area: float):
        """ Create a random airspace object with min_area < area < max_area

        Parameters
        __________
        min_area: float
            minimum area of the sector (nm^2)
        max_area: float
            maximum area of the sector (nm^2)
        
        Returns
        __________
        airspace: Airspace
            randomly created airspace object
        """
        pass
        
    @abstractmethod
    def random_flight(self, min_speed: float, max_speed: float, tol: float = 0) -> Tuple[Point, Point, float, str]:
        """ Get initial conditions for random flight in the airspace

        Parameters
        __________
        airspace: Airspace
            Airspace object in which the aircraft has to spawn
        min_speed: float
            minimum speed the created aircraft should fly
        max_speed: float
            maximum speed the created aircraft should fly
        tol: float = 0
            minimum distance between the aircraft and its target after spawning
        
        Returns
        __________
        position: Point
            initial position of the aircraft in the airspace
        target: Point
            initial waypoint of the aircraft in the airspace
        airspeed: float
            initial speed of the aircraft
        flight_type: string
            category of the flight, depends on the operations flown in the airspace

        """
        pass
    
    @abstractmethod
    def get_waypoint(self, position: Point, flight_type: str, target: Optional[Point] = None) -> Point:
        """ Determine the waypoint for the aircraft 

        Parameters
        __________
        position: Point
            current location of the aircraft in the airspace
        flight_type: str
            classifier for the type of flight that is being conducted
        target: Point (optional)
            location of the current (previous) waypoint
        
        Returns
        __________
        target: Point
            new waypoint that the flight should direct to

        """
        pass
    
@dataclass
class EnrouteAirspace(Airspace):
    """ Enroute Airspace class, inherits from Airspace

    Attributes
    ___________
    polygon: shapely.geomtry.Polygon 
        object that describes the airspace geometry
    category: string
        airspace category, determines the type of operations being flown

    Methods
    ___________
    random(min_area, max_area) -> Airspace object
        class method that returns a random Airspace object 
    random_flight(self) -> position, target, speed, flight type
        returns random initial conditions for a flight 
        in accordance with the airspace type
    get_waypoint(self, position, flight_type, target):
        returns a new waypoint for an aircraft in accordance
        with the airspace 
    """
    #TODO add possibility to spawn on the airspace border

    @classmethod
    def random(cls, min_area: float, max_area: float):
        R = math.sqrt(max_area / math.pi)
        p = [fn.random_point_in_circle(R) for _ in range(3)]
        polygon = Polygon(p).convex_hull

        while polygon.area < min_area:
            p.append(fn.random_point_in_circle(R))
            polygon = Polygon(p).convex_hull

        return cls(polygon = polygon, category = "enroute")
    
    def random_flight(self, min_speed: float, max_speed: float, tol: float = 0) -> Tuple[Point,Point,float,str]:
        flight_type = "enroute"
        position = fn.random_point_in_polygon(self.polygon)
        target = self.get_waypoint(position, flight_type)

        airspeed = random.uniform(min_speed, max_speed)
        return position, target, airspeed, flight_type
    
    def get_waypoint(self, position: Point, flight_type: str, target: Point | None = None) -> Point:
        #TODO add more sensible get waypoint method
        if target is not None:
            return target

        d = random.uniform(0, self.polygon.boundary.length)
        target = self.polygon.boundary.interpolate(d)
        target = Point(target.x * 10, target.y * 10)
        return target

@dataclass
class MixedAirspace(Airspace):
    """ Mixed Airspace class, inherits from Airspace

    Attributes
    ___________
    polygon: shapely.geomtry.Polygon 
        object that describes the airspace geometry
    category: string
        airspace category, determines the type of operations being flown

    Methods
    ___________
    random(min_area, max_area) -> Airspace object
        class method that returns a random Airspace object 
    random_flight(self) -> position, target, speed, flight type
        returns random initial conditions for a flight 
        in accordance with the airspace type
    """
    #TODO actually implement the airspace class
    enroute_airspace: Optional[Airspace] = None

    @classmethod
    def random(cls, min_area: float, max_area: float):
        R = math.sqrt(max_area / math.pi)
        p = [fn.random_point_in_circle(R) for _ in range(3)]
        polygon = Polygon(p).convex_hull

        while polygon.area < min_area:
            p.append(fn.random_point_in_circle(R))
            polygon = Polygon(p).convex_hull

        return cls(polygon = polygon, category = "mixed", enroute_airspace = EnrouteAirspace(polygon,"enroute"))
    
    def random_flight(self, min_speed: float, max_speed: float, tol: float = 0) -> Tuple[Point,Point,float,str]:
        position = fn.random_point_in_polygon(self.polygon)
        boundary = self.polygon.boundary

        while True:
            d = random.uniform(0, self.polygon.boundary.length)
            target = boundary.interpolate(d)
            target = Point(target.x * 10, target.y * 10)
            if target.distance(position) > tol:
                break

        airspeed = random.uniform(min_speed, max_speed)
        return position, target, airspeed, "enroute"
    
    def get_waypoint(self, position: Point, flight_type: str, target: Point | None = None) -> Point:
        pass