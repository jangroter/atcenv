from jsonargparse import CLI

from atcenv.src.atcenv import AtcEnv

from atcenv.src.environment_objects.airspace import Airspace
from atcenv.src.environment_objects.flight import Aircraft, Flight
from atcenv.src.environment_objects.environment import Environment
from atcenv.src.models.model import Model
from atcenv.src.observation.observation import Observation
from atcenv.src.reward.reward import Reward
from atcenv.src.scenarios.scenario import Scenario
from atcenv.src.logger.logger import Logger
import sys


def main(environment: Environment,
        model: Model,
        scenario: Scenario,
        airspace: Airspace,
        aircraft: Aircraft,
        observation: Observation,
        reward: Reward,
        logger: Logger):
    """
    Main should start the scenario's and run them, as input it should take the CR class, the environment class
    and all other important parameters required for running it. 

    It should start a thread from which the simulations will be run
    """

    atcenv = AtcEnv(environment=environment,
                    model=model,
                    scenario=scenario,
                    airspace=airspace,
                    aircraft=aircraft,
                    observation=observation,
                    reward=reward,
                    logger=logger)
    
    atcenv.run_scenario()
    
    # Initiliaze folder structure for saving and loading if neccessary
    # Initiliaze classes with proper values

    # Start scenario


if __name__ == '__main__':
    if '--config' in sys.argv:
        CLI(main, as_positional=False)
    else:
        CLI(main, args=['--config', 'config.yaml'])