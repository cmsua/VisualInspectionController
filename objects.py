import abc

import numpy as np


class Camera(abc.ABC):
    def __init__():
        pass

    @abc.abstractmethod
    def get_image() -> np.typing.ArrayLike:
        pass


class Machine(abc.ABC):
    def __init__():
        pass

    def __enter__():
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    @abc.abstractmethod
    def get_images(self, camera: Camera) -> np.typing.ArrayLike:
        pass
