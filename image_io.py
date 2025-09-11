import logging
import os
import multiprocessing

import numpy as np
import cv2

logger = logging.getLogger("image_io")

images_subdir = "raw_images"
data_file = "images.npy"


def write_image(image, path):
    logger.debug(f"Saving image to {path}")
    cv2.imwrite(path, image)


def write_images(images, dir, write_raws: bool = False):
    os.makedirs(dir)
    logger.debug(f"Created folder {dir}")

    np.save(os.path.join(dir, data_file), images)
    logger.debug(f"")

    if write_raws:
        images_dir = os.path.join(dir, images_subdir)
        os.makedirs(images_dir)
        logger.debug(f"Created folder {images_dir}")

        # Save Images
        logger.debug("Saving Images")
        files = []
        for y, row in enumerate(images):
            for x, image in enumerate(row):
                if image is None:
                    continue

                logger.debug(f"Queueing image X{x}Y{y}.png")
                raw_path = os.path.join(images_dir, f"X{x}Y{y}.png")
                files.append((image, raw_path))

        logger.debug("Queued all images")

        with multiprocessing.Pool(processes=16) as pool:
            pool.starmap(write_image, files)
