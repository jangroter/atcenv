"""
Script for generating the training and testing scenarios
"""

import pickle
from typing import Dict, List
from atcenv.definitions import *


import math as m
import numpy as np

# Scenario parameters
num_scenarios = 100
num_flights = 15
max_area = 63. * 63. * (u.nm ** 2)
min_area = 40. * 40. * (u.nm ** 2)

scenario_name = 'test_scenario_high_density/'

# Fixed parameters
distance_init_buffer = 2.
min_distance = 5. * distance_init_buffer * u.nm
min_speed = 400. * u.kt
max_speed = 500. * u.kt
tol = max_speed * 1.05 * 5 * distance_init_buffer

for i in range(0,num_scenarios):
    airspace = Airspace.random(min_area, max_area)
    flights = []
    print(i)
    counter = 0

    while len(flights) < num_flights:
        valid = True
        candidate = Flight.random(airspace, min_speed, max_speed, tol)
        # ensure that candidate is not in conflict
        for f in flights:
            if counter == 250:
                airspace = Airspace.random(min_area, max_area)
                flights = []
                counter = 0

            if candidate.position.distance(f.position) < min_distance:
                valid = False
                counter += 1
                break
        if valid:
            flights.append(candidate)

    with open(f'{scenario_name}airspace_{i}.p', 'wb') as handle:
        pickle.dump(airspace, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'{scenario_name}flights_{i}.p', 'wb') as handle:
        pickle.dump(flights, handle, protocol=pickle.HIGHEST_PROTOCOL)


            

