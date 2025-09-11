import logging
import os

import numpy as np

from objects import Camera, Machine

logger = logging.getLogger("ReuseMachine")


class ReuseMachine(Machine):
    def __init__(self, config: dict):
        self._path = config["path"] if "path" in config else None
        self._output_dir = config["output_dir"] if "output_dir" in config else "data"

    # Load images from latest
    def __enter__(self):
        path = self._path

        if path is None:
            all_subdirs = [
                os.path.join(self._output_dir, d) for d in os.listdir(self._output_dir)
            ]
            path = os.path.join(max(all_subdirs, key=os.path.getmtime), 'images.npy')

        logger.info(f"Loading images from folder {path}")
        self._images = np.load(path)
        
        return self

    # Return images
    def get_images(self, camera: Camera) -> np.typing.ArrayLike:
        return self._images
