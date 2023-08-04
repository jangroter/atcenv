from jsonargparse import CLI
from atcenv.src.scenarios.scenario import Scenario
from atcenv.src.environment_objects.airspace import Airspace

import sys


def main(scenario: Scenario, airspace: Airspace):
    """
    Main should start the scenario's and run them, as input it should take the CR class, the environment class
    and all other important parameters required for running it. 

    It should start a thread from which the simulations will be run
    """
    print(airspace.random(100,200))
    
    # Initiliaze folder structure for saving and loading if neccessary
    # Initiliaze classes with proper values

    # Start scenario


if __name__ == '__main__':
    if '--config' in sys.argv:
        CLI(main, as_positional=False)
    else:
        CLI(main, args=['--config', 'config.yaml'])