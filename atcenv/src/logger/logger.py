from abc import ABC, abstractmethod
import numpy as np
import shutil
import torch

from atcenv.src.environment_objects.environment import Environment
from atcenv.src.models.model import Model
import atcenv.src.functions as fn

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')

class Logger(ABC):
    """ Logger class

    Attributes
    ___________
    log_frequency: int
        how often to save the logged data away
    verbose: bool
        should information be printed to the terminal while running

    Methods
    ___________
    initialize_episode(self, episode) -> None 
        initiliazes the data arrays for this episode
    store_transition(self, environment, reward) -> None
        stores the information corresponding to the provided state
    store_episode(self) -> None
        stores a summary of this episode, if necessary also saves 
        a copy of the data arrays
    setup_experiment(self, experiment_name, config_file) -> str
        creates the required folder structure and saves a copy of the
        config file. Returns the path to the created experiment folder

    """

    def __init__(self,
                 log_frequency: int,
                 verbose: bool):
        self.log_frequency = log_frequency
        self.verbose = verbose

        self.output_folder = None
        self.results_folder = None

        self.log_experiment = False
        if self.log_frequency != 0:
            self.log_experiment = True
    
    @abstractmethod
    def initialize_episode(self, episode: int) -> None:
        """ initializes required parameters for logging new episode

        Parameters
        __________
        epsiode: int
            current episode
        
        Returns
        __________
        None

        """
        pass
    
    @abstractmethod
    def store_transition(self, environment: Environment, reward: np.ndarray) -> None:
        """ stores relevant state information in the corresponding arrays

        Parameters
        __________
        environment: Environment
            Environment object that contains the entire current state
        reward: numpy array
            reward obtained for all of the flights based on the previous
            state transition
        
        Returns
        __________
        None

        """
        pass

    @abstractmethod
    def store_episode(self, model: Model) -> None:
        """ stores (a summary of) the episode data in their corresponding arrays

        Parameters
        __________
        None
        
        Returns
        __________
        None

        """
        pass

    def setup_experiment(self, experiment_name: str, config_file: str) -> str:
        """ Creates the required folder structure for saving the experiment data

        Raises an Exception if there already is an experiment with this name
        and self.log_experiment is True (self.log_frequency > 0)

        Parameters
        __________
        experiment_name: str
            name of the to be conducted experiment
        config_file: str
            path to the used config file, can be obtained using sys.argvs
        
        Returns
        __________
        output_folder: str
            path to the main folder for this experiment

        """
        if self.log_experiment:
            output_folder = "atcenv/output/" + experiment_name
            exist = fn.check_dir_exist(output_folder, mkdir = True)
            if exist:
                raise Exception("Experiment with this name already exists, please use a different name, or delete the previous experiment")
            else:
                config_folder = output_folder + '/config'
                _ = fn.check_dir_exist(config_folder, mkdir = True)

                results_folder = output_folder + '/results'
                _ = fn.check_dir_exist(results_folder, mkdir = True)

                shutil.copyfile(config_file, config_folder + "/config.yaml")

            self.output_folder = output_folder
            self.results_folder = results_folder

            return output_folder

class NoLogger(Logger):

    def initialize_episode(self, *args):
        pass

    def store_transition(self, *args):
        pass

    def store_episode(self, *args):
        pass

class BasicLogger(Logger):

    """ BasicLogger class, inherits from Logger

    BasicLogger logs the mean reward, number of conflicts (normalized
    with respect to number of aircraft) and mean drift angle of th flights

    Attributes
    ___________
    log_frequency: int
        how often to save the logged data away
    verbose: bool
        should information be printed to the terminal while running

    Methods
    ___________
    initialize_episode(self, episode) -> None 
        initiliazes the data arrays for this episode
    store_transition(self, environment, reward) -> None
        stores the information corresponding to the provided state
    store_episode(self) -> None
        stores a summary of this episode, if necessary also saves 
        a copy of the data arrays
    setup_experiment(self, experiment_name, config_file) -> str
        creates the required folder structure and saves a copy of the
        config file. Returns the path to the created experiment folder

    """

    def __init__(self,
                 log_frequency: int,
                 verbose: bool):
        super().__init__(log_frequency, verbose)
        self.episode = 0
        
        self.cur_reward = np.array([])
        self.cur_conflicts = np.array([])
        self.cur_drift_angle = np.array([])

        self.reward = np.array([])
        self.conflicts = np.array([])
        self.drift_angle = np.array([])
    
    def initialize_episode(self, episode: int) -> None:

        self.episode = episode

        self.cur_reward = np.array([])
        self.cur_conflicts = np.array([])
        self.cur_drift_angle = np.array([])

    def store_transition(self, environment: Environment, reward: np.ndarray) -> None:
        """ stores relevant state information in the corresponding arrays

        BasicLogger logs the mean reward, number of conflicts (normalized
        with respect to number of aircraft) and mean drift angle of th flights

        Parameters
        __________
        environment: Environment
            Environment object that contains the entire current state
        reward: numpy array
            reward obtained for all of the flights based on the previous
            state transition
        
        Returns
        __________
        None

        """

        self.cur_reward = np.append(self.cur_reward, np.mean(reward))
        self.cur_conflicts = np.append(self.cur_conflicts, (len(environment.conflicts)/len(environment.flights)))
        self.cur_drift_angle = np.append(self.cur_drift_angle, np.mean(np.abs(np.array([f.drift for f in environment.flights]))))

    def store_episode(self, model: Model) -> None:

        self.reward = np.append(self.reward, np.sum(self.cur_reward))
        self.conflicts = np.append(self.conflicts, np.mean(self.cur_conflicts))
        self.drift_angle = np.append(self.drift_angle, np.mean(self.cur_drift_angle))

        if self.log_frequency != 0 and self.episode % self.log_frequency == 0:
            np.savetxt(self.results_folder+'/reward.csv', self.reward)
            np.savetxt(self.results_folder+'/conflicts.csv', self.conflicts)
            np.savetxt(self.results_folder+'/drift_angle.csv', self.drift_angle)
        
        if self.verbose:
            print(f"Episode {self.episode} completed, average of {np.mean(self.reward[-100:])} reward (ao100), average of {np.mean(self.conflicts[-100:])} conflicts")

class RLLogger(Logger):

    """ BasicLogger class, inherits from Logger

    BasicLogger logs the mean reward, number of conflicts (normalized
    with respect to number of aircraft) and mean drift angle of th flights

    Attributes
    ___________
    log_frequency: int
        how often to save the logged data away
    verbose: bool
        should information be printed to the terminal while running

    Methods
    ___________
    initialize_episode(self, episode) -> None 
        initiliazes the data arrays for this episode
    store_transition(self, environment, reward) -> None
        stores the information corresponding to the provided state
    store_episode(self) -> None
        stores a summary of this episode, if necessary also saves 
        a copy of the data arrays
    setup_experiment(self, experiment_name, config_file) -> str
        creates the required folder structure and saves a copy of the
        config file. Returns the path to the created experiment folder

    """

    def __init__(self,
                 log_frequency: int,
                 verbose: bool):
        super().__init__(log_frequency, verbose)
        self.episode = 0
        
        self.best_reward = -999

        self.cur_reward = np.array([])
        self.cur_conflicts = np.array([])
        self.cur_drift_angle = np.array([])

        self.reward = np.array([])
        self.conflicts = np.array([])
        self.drift_angle = np.array([])
    
    def initialize_episode(self, episode: int) -> None:

        self.episode = episode

        self.cur_reward = np.array([])
        self.cur_conflicts = np.array([])
        self.cur_drift_angle = np.array([])

    def store_transition(self, environment: Environment, reward: np.ndarray) -> None:
        """ stores relevant state information in the corresponding arrays

        BasicLogger logs the mean reward, number of conflicts (normalized
        with respect to number of aircraft) and mean drift angle of th flights

        Parameters
        __________
        environment: Environment
            Environment object that contains the entire current state
        reward: numpy array
            reward obtained for all of the flights based on the previous
            state transition
        
        Returns
        __________
        None

        """

        self.cur_reward = np.append(self.cur_reward, np.mean(reward))
        self.cur_conflicts = np.append(self.cur_conflicts, (len(environment.conflicts)/len(environment.flights)))
        self.cur_drift_angle = np.append(self.cur_drift_angle, np.mean(np.abs(np.array([f.drift for f in environment.flights]))))

    def store_episode(self, model: Model) -> None:

        self.reward = np.append(self.reward, np.sum(self.cur_reward))
        self.conflicts = np.append(self.conflicts, np.mean(self.cur_conflicts))
        self.drift_angle = np.append(self.drift_angle, np.mean(self.cur_drift_angle))

        if self.log_frequency != 0 and self.episode % self.log_frequency == 0:
            np.savetxt(self.results_folder+'/reward.csv', self.reward)
            np.savetxt(self.results_folder+'/conflicts.csv', self.conflicts)
            np.savetxt(self.results_folder+'/drift_angle.csv', self.drift_angle)
            np.savetxt(self.results_folder+'/q_loss.csv', model.qf1_lossarr)

            self.plot_figures(model)
        
        if self.log_frequency != 0 and np.mean(self.reward[-100:]) > self.best_reward:
            self.best_reward = np.mean(self.reward[-100:])
            self.save_models(model)
        
        if self.verbose:
            print(f"Episode {self.episode} completed")

    def save_models(self, model):
        torch.save(model.actor.state_dict(), self.weights_folder+"/actor.pt")
        torch.save(model.critic_q_1.state_dict(), self.weights_folder+"/qf1.pt")
        torch.save(model.critic_q_2.state_dict(), self.weights_folder+"/qf2.pt")
        torch.save(model.critic_v.state_dict(), self.weights_folder+"/vf.pt")    
        torch.save(model.critic_v_target.state_dict(), self.weights_folder+"/vf_target.pt")    
    
    def plot_figures(self, model):
        fig, ax = plt.subplots()
        ax.plot(model.qf1_lossarr, label='qf1')
        ax.plot(model.qf2_lossarr, label='qf2')
        ax.plot(fn.moving_average(model.qf2_lossarr,500))
        ax.set_yscale('log')
        fig.savefig(self.output_folder+'/qloss.png')
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.plot(self.reward, label='reward')
        ax.plot(fn.moving_average(self.reward,500))
        fig.savefig(self.output_folder+'/reward.png')
        plt.close(fig)

    def setup_experiment(self, experiment_name: str, config_file: str) -> str:
        """ Creates the required folder structure for saving the experiment data

        Raises an Exception if there already is an experiment with this name
        and self.log_experiment is True (self.log_frequency > 0)

        Parameters
        __________
        experiment_name: str
            name of the to be conducted experiment
        config_file: str
            path to the used config file, can be obtained using sys.argvs
        
        Returns
        __________
        output_folder: str
            path to the main folder for this experiment

        """
        if self.log_experiment:
            output_folder = "atcenv/output/" + experiment_name
            exist = fn.check_dir_exist(output_folder, mkdir = True)
            if exist:
                raise Exception("Experiment with this name already exists, please use a different name, or delete the previous experiment")
            else:
                config_folder = output_folder + '/config'
                _ = fn.check_dir_exist(config_folder, mkdir = True)

                results_folder = output_folder + '/results'
                _ = fn.check_dir_exist(results_folder, mkdir = True)

                weights_folder = output_folder + '/weights'
                _ = fn.check_dir_exist(weights_folder, mkdir = True)

                shutil.copyfile(config_file, config_folder + "/config.yaml")

            self.output_folder = output_folder
            self.results_folder = results_folder
            self.weights_folder = weights_folder

            return output_folder

class RLLoggerV2(RLLogger):
    def __init__(self,
                 log_frequency: int,
                 verbose: bool):
        super().__init__(log_frequency, verbose)

    def save_models(self, model):
        torch.save(model.actor.state_dict(), self.weights_folder+"/actor.pt")
        torch.save(model.critic_q_1.state_dict(), self.weights_folder+"/qf1.pt")
        torch.save(model.critic_q_2.state_dict(), self.weights_folder+"/qf2.pt")
        torch.save(model.critic_q_target_1.state_dict(), self.weights_folder+"/qf1_target.pt")    
        torch.save(model.critic_q_target_2.state_dict(), self.weights_folder+"/qf2_target.pt")    
    
class RLLoggerV3(RLLogger):
    def __init__(self,
                 log_frequency: int,
                 verbose: bool):
        super().__init__(log_frequency, verbose)

    def save_models(self, model):
        torch.save(model.actor.state_dict(), self.weights_folder+"/actor.pt")
        torch.save(model.critic_q.state_dict(), self.weights_folder+"/qf.pt")
        torch.save(model.critic_q_target.state_dict(), self.weights_folder+"/qf_target.pt")

    



        


