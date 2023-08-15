from abc import ABC, abstractmethod

class Logger(ABC):
    def __init__(self,
                 verbose):
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

    