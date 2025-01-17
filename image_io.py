import logging
import os
import multiprocessing

import numpy as np
import cv2
import tqdm

logger = logging.getLogger('image_io')

images_subdir = 'raw_images'
data_file = 'images.npy'

def write_image(image, path):
    logger.debug(f'Saving image to {path}')
    cv2.imwrite(path, image)

def write_images(images, dir, x_start, x_inc, y_start, y_inc):
    os.makedirs(dir)
    logger.debug(f'Created folder {dir}')

    np.save(os.path.join(dir, data_file), images)
    logger.debug(f'')

    images_dir = os.path.join(dir, images_subdir)
    os.makedirs(images_dir)
    logger.debug(f'Created folder {images_dir}')


    # Save Images
    logger.debug('Saving Images')
    files = []
    for y_num, row in enumerate(images):
        for x_num, image in enumerate(row):
            if image is None:
                continue

            x = x_start + x_num * x_inc
            y = y_start + y_num * y_inc
            
            logger.debug(f'Queueing image X{x}Y{y}.png')
            raw_path = os.path.join(images_dir, f'X{x}Y{y}.png')
            files.append((image, raw_path))

    logger.debug('Queued all images')
    
    with multiprocessing.Pool(processes=16) as pool:
        pool.starmap(write_image, files)

@DeprecationWarning
def load_images_form_files(dir, x_start, x_inc, x_end, y_start, y_inc, y_end):
    logger.info(f'Loading to Numpy from {dir}')

    rows = len(range(y_start, y_end + y_inc, y_inc))
    cols = len(range(x_start, x_end + x_inc, x_inc))
    
    np_images = None
    
    total_images = len(range(y_start, y_end + y_inc, y_inc)) * len(range(x_start, x_end + x_inc, x_inc))
    pbar = tqdm.tqdm(desc='Loading Images', total=total_images)

    for row_id, y in enumerate(range(y_start, y_end + y_inc, y_inc)):
        row_strip = None
        for col_id, x in enumerate(range(x_start, x_end + x_inc, x_inc)):
            logger.debug(f'Loading image X{x}Y{y}.png')
            file = os.path.join(dir, 'raw', f'X{x}Y{y}.png')
            image = cv2.imread(file)

            if row_strip is None:
                row_strip = np.zeros((cols, *image.shape), dtype=np.uint8)
            row_strip[col_id] = np.asarray(image, dtype=np.uint8)
            pbar.update()
            
        if np_images is None:
            np_images = np.zeros((rows, *row_strip.shape), dtype=np.uint8)
        np_images[row_id] = row_strip

        logger.debug(f'Loaded row {row_id}')
        
    pbar.close()
    return np_images

def load_images(dir):
    logger.info(f'Loading to Numpy from {dir}')
    return np.load(os.path.join(dir, data_file))
