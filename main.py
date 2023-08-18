from jsonargparse import CLI
import sys

from atcenv.src.atcenv import AtcEnv

from atcenv.src.environment_objects.airspace import Airspace
from atcenv.src.environment_objects.flight import Aircraft, Flight
from atcenv.src.environment_objects.environment import Environment
from atcenv.src.models.model import Model
from atcenv.src.observation.observation import Observation
from atcenv.src.reward.reward import Reward
from atcenv.src.scenarios.scenario import Scenario
from atcenv.src.logger.logger import Logger

import atcenv.src.functions as fn

def main(experiment_name: str,
        environment: Environment,
        model: Model,
        scenario: Scenario,
        airspace: Airspace,
        aircraft: Aircraft,
        observation: Observation,
        reward: Reward,
        logger: Logger):

    experiment_folder = logger.setup_experiment(experiment_name, config_file = sys.argv[2])
    model.setup_model(experiment_folder)

    atcenv = AtcEnv(environment=environment,
                    model=model,
                    scenario=scenario,
                    airspace=airspace,
                    aircraft=aircraft,
                    observation=observation,
                    reward=reward,
                    logger=logger)
    
    atcenv.run_scenario()

if __name__ == '__main__':
    if '--config' in sys.argv:
        CLI(main, as_positional=False)
    else:
        CLI(main, args=['--config', 'config.yaml'])