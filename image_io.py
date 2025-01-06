import logging
import os
import multiprocessing

from PIL import Image


logger = logging.getLogger('image_io')

def write_image(image, paths):
    for path in paths:
        image.save(path)

def write_images(images, dir, x_start, x_inc, y_start, y_inc):
    os.makedirs(dir)

    raw_dir = os.path.join(dir, 'raw')
    os.makedirs(raw_dir)

    raw_unsorted_dir = os.path.join(dir, 'raw_unsorted')
    os.makedirs(raw_unsorted_dir)

    logger.info(f'Created folder {dir} and subfolders raw, raw_unsorted')

    # Save Grid
    logger.info('Saving Images')
    counter = 0

    files = []
    for y_num, row in enumerate(images):
        for x_num, image in enumerate(row):
            if image is None:
                continue

            x = x_start + x_num * x_inc
            y = y_start + y_num * y_inc
            
            logger.debug(f'Queueing image X{x}Y{y}.png ({counter}.png)')
            raw_path = os.path.join(raw_dir, f'X{x}Y{y}.png')
            raw_unsorted_path = os.path.join(raw_unsorted_dir, f'{counter}.png')
            files.append((image, [raw_path, raw_unsorted_path]))

            counter = counter + 1

    logger.info('Queued all images')
    
    with multiprocessing.Pool(processes=8) as pool:
        pool.starmap(write_image, files)

    logger.info('Saved images')

def load_images(dir, x_start, x_inc, x_end, y_start, y_inc, y_end):
    logger.info(f'Loading from {dir}')

    files = os.listdir(os.path.join(dir, 'raw'))
    with multiprocessing.Pool(processes=8) as pool:
        raw_images = pool.map(Image.open, [os.path.join(dir, 'raw', file) for file in files])
    logger.info('Loaded all files')

    images = []
    for y in range(y_start, y_end + y_inc, y_inc):
        row = []
        for x in range(x_start, x_end + x_inc, x_inc):
            logger.debug(f'Loading image X{x}Y{y}.png')
            row += [raw_images[files.index(f'X{x}Y{y}.png')]]
        
        images += [row]

    return images