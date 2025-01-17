import logging
import datetime
import time

import moonrakerpy as moonpy
import numpy as np
import tqdm

from camera import CameraWrapper

x_max = 240
y_max = 400

logger = logging.getLogger('machine')

def create_images(x_start, x_inc, x_end, y_start, y_inc, y_end, stabilize_delay, skipped_points=[]):
    # Open Printer
    logger.info('Opening printer...')
    printer = moonpy.MoonrakerPrinter('http://localhost')

    # Open Webcam
    logger.info('Opening webcam...')
    camera = CameraWrapper()

    ##
    ## HOMING
    ##

    logger.info('Homing X+Y')
    printer.send_gcode('G28')
    printer.send_gcode('M400')

    ##
    ## START CAPTURE
    ##


    start_time = datetime.datetime.now()

    images = None

    forward = True
    # Iterate over a grid
    rows = len(range(y_start, y_end + y_inc, y_inc))
    cols = len(range(x_start, x_end + x_inc, x_inc))
    pbar = tqdm.tqdm(desc='Capturing Images', total=rows * cols)
    
    for row_index, y in enumerate(range(y_start, y_end + y_inc, y_inc)):
        # Handle direction switching
        x_points = range(x_start, x_end + x_inc, x_inc)
        if not forward:
            x_points = reversed(x_points)

        for col_index, x in enumerate(x_points):
            logger.debug(f'Capturing image {row_index * cols + col_index} out of {rows * cols}')
            # Check for skipped points
            skip = False
            for point in skipped_points:
                if x == point[0] and y == point[1]:
                    skip = True
                    break

            if skip:
                logger.debug(f'Skipping point {x}, {y}')
                continue

            if x > x_max:
                raise ValueError(f'x > x_max ({x} > {x_max})')
            if y > y_max:
                raise ValueError(f'y > y_max ({x} > {x_max})')

            # Move machine
            logger.debug(f'Moving to {x}, {y} (row: {row_index}, image captured: {col_index})')
            printer.send_gcode(f'G1 X{x} Y{y} F12000')
            printer.send_gcode('M400')

            # Wait
            logger.debug('Waiting for camera to stabilize')
            time.sleep(stabilize_delay)

            # Write image
            logger.debug('Reading frame')
            frame = camera.get_image()

            if images is None:
                logger.debug('Creating images')
                images = np.zeros((rows, cols, *frame.shape), dtype=np.uint8)
            
            col_index_actual = col_index if forward else cols - col_index - 1
            logger.debug(f'Using col index {col_index_actual}')

            images[row_index][col_index_actual] = frame

            pbar.update()

        forward = not forward

    pbar.close()

    seconds = (datetime.datetime.now() - start_time).seconds
    logger.info(f'Finished in {seconds}s')
    logger.info(f'Finished in {seconds - (len(images) * len(images[0]) * stabilize_delay)}s (no sleep)')

    camera.close()


    # Disable Motors
    # printer.send_gcode('M18')


    return images