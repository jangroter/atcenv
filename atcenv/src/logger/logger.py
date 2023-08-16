from abc import ABC, abstractmethod

class Logger(ABC):
    def __init__(self,
                 verbose: bool):
        self.verbose = verbose
    
    @abstractmethod
    def initialize_episode(self, *args):
        pass
    
    @abstractmethod
    def store_transition(self, *args):
        pass

    @abstractmethod
    def store_episode(self, *args):
        pass

class NoLogger(Logger):

    def initialize_episode(self, *args):
        pass

    def store_transition(self, *args):
        pass

    def store_episode(self, *args):
        pass