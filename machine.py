import logging
import datetime
import time

import moonrakerpy as moonpy

from camera import CameraWrapper

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

    images = []

    forward = True
    # Iterate over a grid
    for y in range(y_start, y_end + y_inc, y_inc):
        # Handle direction switching
        x_points = range(x_start, x_end + x_inc, x_inc)
        if not forward:
            x_points = reversed(x_points)

        row = []
        for x in x_points:
            # Check for skipped points
            skip = False
            for point in skipped_points:
                if x == point[0] and y == point[1]:
                    skip = True
                    break

            if skip:
                logger.info(f'Skipping point {x}, {y}')
                row += [None]
                continue

            # Move machine
            logger.info(f'Moving to {x}, {y} out of {x_end + x_inc}, {y_end + y_inc}')
            printer.send_gcode(f'G1 X{x} Y{y} F12000')
            printer.send_gcode('M400')

            # Wait
            logger.debug('Waiting for camera to stabilize')
            time.sleep(stabilize_delay)

            # Write image
            logger.debug('Reading frame')
            frame = camera.get_image()
            row += [frame]

        if not forward:
            row.reverse()

        images += [row]
        forward = not forward

    seconds = (datetime.datetime.now() - start_time).seconds
    logger.info(f'Finished in {seconds}s')
    logger.info(f'Finished in {seconds - (len(images) * len(images[0]) * stabilize_delay)}s (no sleep)')

    camera.close()


    # Disable Motors
    # printer.send_gcode('M18')


    return images