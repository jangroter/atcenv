from atcenv.src.environment_objects.airspace import Airspace
from atcenv.src.environment_objects.flight import Aircraft, Flight
from atcenv.src.environment_objects.environment import Environment
from atcenv.src.models.model import Model
from atcenv.src.observation.observation import Observation
from atcenv.src.reward.reward import Reward
from atcenv.src.scenarios.scenario import Scenario
from atcenv.src.logger.logger import Logger

from typing import Tuple, Optional, List

class AtcEnv():
    def __init__(self,
                 environment: Environment,
                 model: Model,
                 scenario: Scenario,
                 airspace: Airspace,
                 aircraft: Aircraft,
                 observation: Observation,
                 reward: Reward,
                 logger: Logger) -> None:
                 
        self.environment = environment  # Currently OK
        self.model = model  # Have to implement MVP and different DRL models
        self.scenario = scenario  # Currently OK
        self.airspace_template = airspace  # Currently OK, have to implement Merging
        self.aircraft = aircraft  # Currently OK
        self.observation = observation  # Have to implement Global for transformers, 
        self.reward = reward  # Have to implement basic reward function
        self.logger = logger  # Currently OK

    def run_scenario(self) -> None:
        while self.scenario.episode_counter < self.scenario.num_episodes:
            airspace, flights, test = self.scenario.get_scenario(self.airspace_template,self.aircraft)
            self.model.new_episode(test)
            self.environment.create_environment(airspace, flights, episode = self.scenario.episode_counter)
            self.run_episode(test)

    def run_episode(self, test: bool) -> None:
        self.logger.initialize_episode(self.scenario.episode_counter)
        new_observation = self.observation.get_observation(self.environment.flights)
        while not self.environment.done:
            observation = new_observation
            action = self.model.get_action(observation)
            done = self.environment.step(action, self.model.transform_action)
            new_observation = self.observation.get_observation(self.environment.flights)
            reward = self.reward.get_reward(self.environment.flights)
            self.store_transition(observation["observation"],action,new_observation["observation"],reward,done)
        self.store_episode()

    def store_episode(self):
        self.logger.store_episode()
    
    def store_transition(self, observation, action, new_observation, reward, done):
        self.logger.store_transition(self.environment, reward)
        self.model.store_transition(observation,action,new_observation,reward,done)

        