from atcenv.src.environment_objects.airspace import Airspace
from atcenv.src.environment_objects.flight import Flight, Aircraft
import atcenv.src.units as u
import atcenv.src.functions as fn

from typing import Tuple, List, Optional
from dataclasses import dataclass, field

import pickle
import math
import random

@dataclass
class Scenario():
    """ Scenario class

    Main class responsible for determining the type of experiments
    and simulations that are being run. Handles the creation,
    saving and loading of scenarios that can be used by the environment 
    class. 

    Attributes
    ___________
    num_episodes: int
        number of random episodes that have to be run

    Of the following 3 attributes, only 2 out of 3 have to be non-zero
    the remaining will be calculated based on the others.
        
    num_flights: int
        number of flights that should be created in each scenario
    airspace_area: float
        area of the airspace in NM2 
    traffic_density: float
        traffic density for the scenarios in AC / 100 NM2

    test_scenario_dir: str
        directory for the pregenerated test scenarios
    num_test_episodes: int
        number of test episodes that have to be run, will take 
        them in alphabatical order from the provided directory
    test_frequency: int
        evey 'test_frequency' number of random episodes all
        test episodes will be run for evaluation of the model

    random_seed: int (optional) = 1
        sets the random seed for the scenario generator
    save_scenarios: bool (optional) = False
        boolean for if the newly created random scenarios should be saved 
        inside 'test_scenario_dir'
    
    airspace_area_margin: float (optional) = 1.1
        margin for the airspace area to set area_min and area_max for the
        get airspace method
    
    Methods
    ___________
    get_scenario(self, airspace: Airspace, aircraft_type: Aircraft) -> Tuple[Airspace, List[Flight], bool]
        returns a scenario, uses an internal counter to determine if one should be created
        or loaded from pre-existing scenarios

    create_scenario(self, airspace: Airspace, aircraft_type: Aircraft) -> Tuple[Airspace, List[Flight]]
        creates a new scenario

    load_scenario(self, file_name: str) -> Tuple[Airspace, List[Flight]]
        loads pre-existing scenario
    
    save_scenario(self, airspace: Airspace, flights: List[Flight]) -> None
        saves given airspace and list of flights as a scenario
    
    get_filename_from_counter(self) -> str
        uses the internal test counter to determine which scenario to load
    
    get_airspace(self, airspace: Airspace) -> Airspace
        returns newly created airspace based on the given airspace class 
        and scenario parameters

    get_flights(self, airspace: Airspace, aircraft_type: Aircraft) -> List[Flight]
        returns a list of flights of length self.num_flights based on the provided
        airspace object and aircraft type
    
    reset_counters(self) -> None
        resets the test counter after all test episodes have been completed 
        and increments the training counter with 1

    set_missing_attributes(self) -> None
        sets the correct number of flights, airspace area and traffic density
        depending on the initialization values of the class
        
    """
    num_episodes: int

    num_flights: int
    airspace_area: float
    traffic_density: float
    
    test_scenario_dir: str
    num_test_episodes: int
    test_frequency: int

    random_seed: Optional[int] = 1
    save_scenarios: Optional[bool] = False

    airspace_area_margin: Optional[float] = 1.25

    def __post_init__(self) -> None:
        self.episode_counter = 0
        self.test_counter = 0
        self.set_missing_attributes()
        self.check_num_test_episodes()
        random.seed(self.random_seed)

    def get_scenario(self, airspace: Airspace, aircraft_type: Aircraft) -> Tuple[Airspace, List[Flight], bool]:
        """ returns an airspace, list of flights and boolean for the new scenario

        The method uses an internal counter to determine whether a scenario should be 
        created or loaded from a pre-existing file. To control whether a new scenario 
        will be created or to load a scenario, see;

            create_scenario()
            load_scenario()

        Parameters
        __________
        airspace: Airspace
            base airspace class used as reference for determining the scenario airspace
        aircraft_type: Aircraft
            type of aircraft to be used for the scenario, currently only 1 aircraft 
            type is supported when creating a scenario using this method
        
        Returns
        __________
        airspace: Airspace
            randomly created airspace object based on the reference airspace object
        flights: List[Flight]
            list of flights located in the airspace, has length self.num_flights
        test: bool
            boolean variable that returns True when it is a loaded/test scenario and 
            False when it is a newly created scenario

        """
        if self.test_frequency == 0:
            self.episode_counter += 1
            test = False
            return *self.create_scenario(airspace, aircraft_type), test
        
        elif self.episode_counter % self.test_frequency != 0:
            self.episode_counter += 1
            test = False
            return *self.create_scenario(airspace, aircraft_type), test
        
        else:
            self.test_counter += 1
            self.reset_counters()
            test = True
            return *self.load_scenario(), test

    def create_scenario(self, airspace: Airspace, aircraft_type: Aircraft) -> Tuple[Airspace, List[Flight]]:
        """ creates a random scenario, based on the reference airspace and aircraft type

        Parameters
        __________
        airspace: Airspace
            base airspace class used as reference for determining the scenario airspace
        aircraft_type: Aircraft
            type of aircraft to be used for the scenario, currently only 1 aircraft 
            type is supported when creating a scenario using this method
        
        Returns
        __________
        airspace: Airspace
            randomly created airspace object based on the reference airspace object
        flights: List[Flight]
            list of flights located in the airspace, has length self.num_flights
        """
        flights_list = []
        internal_counter = 0
        while len(flights_list) < self.num_flights:
            airspace = self.get_airspace(airspace)
            flights_list = self.get_flights(airspace, aircraft_type)
            internal_counter += 1
            if internal_counter == 50:
                raise("Unable to create scenario, traffic density is likely too high to create conflict free initial conditions")
        if self.save_scenarios:
            self.save_scenario(airspace,flights_list)
        return airspace, flights_list

    def load_scenario(self, file_name: Optional[str] = None) -> Tuple[Airspace, List[Flight]]:

        if file_name == None:
            file_name = f'scenario_{self.test_counter}.p'
        with open(f'{self.test_scenario_dir}/{file_name}', 'rb') as handle:
            airspace, flights = pickle.load(handle)
        return airspace, flights
        
    def save_scenario(self, airspace: Airspace, flights: List[Flight], file_name: Optional[str] = None) -> None:

        if file_name == None:
            file_name = f'scenario_{self.episode_counter}.p'
        with open(f'{self.test_scenario_dir}/{file_name}', 'wb') as handle:
            pickle.dump((airspace,flights), handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get_filename_from_counter(self) -> str:
        pass
    
    def get_airspace(self, airspace: Airspace) -> Airspace:
        """ returns an airspace for the new scenario

        the method uses self.airspace_area together with self.airspace_area_margin
        to determine the min_area and max_area parameters required for the airspace.random() method

        Parameters
        __________
        airspace: Airspace
            base airspace class used as reference for determining the scenario airspace
        
        Returns
        __________
        airspace: Airspace
            randomly created airspace object based on the reference airspace object
            
        """
        min_area = self.airspace_area / self.airspace_area_margin
        max_area = self.airspace_area * self.airspace_area_margin
        return airspace.random(min_area, max_area)
    
    def get_flights(self, airspace: Airspace, aircraft_type: Aircraft) -> List[Flight]:
        """ returns list of flights for the new scenario

        Uses the airspace objects to spawn aircraft within the airspace using Flight.random()
        it is then checked if the aircraft spawned is not currently in conflict with any other
        aircraft in the airspace before proceeding.
        If it takes more that 250 tries to fit all aircraft it returns the list of all 
        validly spawned aircraft.

        Parameters
        __________
        airspace: Airspace
            base airspace class used as reference for determining the scenario airspace
        aircraft_type: Aircraft
            type of aircraft to be used for the scenario, currently only 1 aircraft 
            type is supported when creating a scenario using this method
        
        Returns
        __________
        flights: List[Flight]
            list of flights located in the airspace, has length self.num_flights
            
        """
        counter = 0
        flights_list = []
        while len(flights_list) < self.num_flights:
            valid = True
            candidate = Flight.random(airspace, aircraft_type)
            for f in flights_list:
                if counter == 250:
                    return flights_list
                if candidate.position.distance(f.position) < aircraft_type.min_distance:
                    valid = False
                    counter += 1
                    break
            if valid:
                flights_list.append(candidate)
        return flights_list

    def reset_counters(self) -> None:
        """ resets the counters after finishing all testing scenarios

        Parameters
        __________
        
        None

        Returns
        __________

        None
            
        """
        if self.test_counter >= self.num_test_episodes:
            self.test_counter = 0
            self.episode_counter += 1
    
    def set_missing_attributes(self) -> None:
        """ sets missing attributes depending on initialization values

        if all parameters 'num_flights, airspace_area and traffic_density'
        are non-zero, it will prioritize using num_flights and traffic_density
        to set the correct airspace area. Otherwise it will fill the missing 
        parameter.

        Parameters
        __________
        
        None

        Returns
        __________
        
        None
            
        """
        if self.num_flights != 0 and self.traffic_density !=0:
            self.airspace_area = u.traffic_density_to_NM2 * (self.num_flights / self.traffic_density)

        elif self.num_flights != 0 and self.airspace_area != 0:
            self.traffic_density = u.traffic_density_to_NM2 * (self.num_flights / self.airspace_area)

        elif self.airspace_area != 0 and self.traffic_density != 0:
            self.num_flights = math.ceil((self.airspace_area * self.traffic_density) / u.traffic_density_to_NM2)

        else:
            raise Exception("Atleast 2 out of 3 attributes of num_flights, traffic_density and airspace_area must be non-zero")
        
    def check_num_test_episodes(self) -> None:
        if self.num_test_episodes != 0:
            count = fn.count_files_in_dir(self.test_scenario_dir)
            if self.num_test_episodes > count:
                print(f"number of test episodes is higher than number of scenario's, setting num_test_episodes to {count}")
                self.num_test_episodes = count
