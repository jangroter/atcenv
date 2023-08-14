import math
import random
import os
from typing import Optional
from shapely.geometry import Polygon, Point
import numpy as np

def bound_angle_positive_negative_pi(angle_radians: float) -> float:
    """ maps any angle in radians to the [-pi,pi] interval 
    Parameters
    __________
    angle_radians: float
        angle that needs to be mapped (in radians)
    
    Returns
    __________
    angle_radians: float
        input angle mapped to the interval [-pi,pi] (in radians)
    """

    if angle_radians > math.pi:
        return -(2*math.pi -angle_radians)
    elif angle_radians < -math.pi:
        return (2*math.pi + angle_radians)
    else:
        return angle_radians
def random_point_in_polygon(polygon: Polygon) -> Point:
    """ Get a random point within a polygon
    Parameters
    __________
    polygon: Polygon
        input polygon, should be of type shapely.geometry.Polygon
    
    Returns
    __________
    point: Point
        randomly sampled shapely.geometry.Point
    """
    minx, miny, maxx, maxy = polygon.bounds
    while True:
        point = Point(random.uniform(minx,maxx), random.uniform(miny,maxy))
        if polygon.contains(point):
            return point
def random_point_in_circle(radius: float) -> Point:
    """ Get a random point in a circle with given radius
    Parameters
    __________
    radius: float
        radius for the circle (nm)
    
    Returns
    __________
    point: Point
        randomly sampled shapely.geometry.Point
    """
    alpha = 2 * math.pi * random.uniform(0., 1.)
    r = radius * math.sqrt(random.uniform(0., 1.))
    x = r * math.cos(alpha)
    y = r * math.sin(alpha)
    return Point(x, y)
def count_files_in_dir(directory: str) -> int:
    count = 0
    for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isfile(item_path):
                count += 1
    return count
def check_dir_exist(directory: str, mkdir: Optional[bool] = False) -> bool:
    exist = os.path.exists(directory)
    if not exist and mkdir:
        os.mkdir(directory)
    return exist
def remove_diagonal(matrix: np.ndarray) -> np.ndarray:
    """ Remove the diagonal from a square matrix
    Parameters
    __________
    matrix: numpy array
        input matrix, should be a square numpy array
    
    Returns
    __________
    matrix: numpy array
        returns a new matrix with all diagonal elements removed
    """
    return matrix[~np.eye(matrix.shape[0],dtype=bool)].reshape(matrix.shape[0],-1)