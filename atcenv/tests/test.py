from abc import ABC
from jsonargparse import lazy_instance

class Dataset(ABC):
    def __init__(self, data_dir: str = 'bye', batch_size: int = 128, augment: bool = True):
        """Generates a dataset from image files in a directory.

        Args:
            data_dir: Directory where the images are stored
            batch_size: Number of images in a single batch
            augment: Whether to do data augmentation (horizontal flipping, random cropping)
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.augment = augment

    def __repr__(self):
        key_values = [f"{key}={value}" for key, value in vars(self).items()]
        return f"{self.__class__.__name__}: {', '.join(key_values)}"