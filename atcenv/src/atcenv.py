
class AtcEnv():
    def __init__(self,
                 env,
                 model,
                 scenario) -> None:
        self.env = env
        self.model = model
        self.scenario = scenario

    def run_scenario(self):
        # while not finished all episodes:
            # get new episode -> from scenario class -> should return flights and airspace class corresponding to scenario
            # run episode
        pass

    def run_episode(self):
        # reset episode with new flight and airspace class
        # while episode not finished
            # get_observation -> return observation vector (own class, also handles normalization)
            # get_action -> from model class, list[n][i], n aircraft, i actions
            # update environment to next state -> requires action
            # get_new_observation -> same function as get_observation
            # get_reward -> return score scalar (maybe own class)
            # store_transition -> model specific, give environment class copy + observations and rewards
        # store_episode -> stores information of the episode for logging
        pass

    def start_episode(self):
        pass

    def store_episode(self):
        pass

    def load_scenario(self):
        pass