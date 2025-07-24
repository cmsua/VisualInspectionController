import logging
import datetime
import time
import os

import moonrakerpy as moonpy
import numpy as np
import tqdm
import cv2

from camera import CameraWrapper

# CV2 Params
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

if not os.path.exists('aruco.png'):
    marker_image = cv2.aruco.generateImageMarker(aruco_dict, 0, 200)
    cv2.imwrite('aruco.png', marker_image)

# Logger
logger = logging.getLogger('machine')

# Get status (if restart needed)
def get_status(printer: moonpy.MoonrakerPrinter) -> str:
    return printer.get('/server/info')['result']['klippy_state']

# Wait for printer to come online
def wait_for_printer(printer: moonpy.MoonrakerPrinter) -> None:
    # Check for activity
    if get_status(printer) != "ready":
        logger.warning('Printer status is not "ready", restarting...')
        printer.post('/printer/restart')
        time.sleep(3)

        if get_status(printer) != "ready":
            logger.debug('Restarting printer firmware')
            printer.post('/printer/firmware_restart')
            time.sleep(4)

            # Make sure it worked
            if get_status(printer) != "ready":
                logger.critical('Printer failed to come online. Exiting.')
                raise RuntimeError('Printer failed to initialize')

def home_printer(printer: moonpy.MoonrakerPrinter, camera: CameraWrapper, stabilize_delay: float) -> None:
    logger.info('Homing Printer')

    # Mechanical homing
    logger.debug('Homing X+Y mechanically')
    printer.send_gcode('G28 X')
    printer.send_gcode('G0 X100')
    printer.send_gcode('M400')
    printer.send_gcode('G28 Y')
    printer.send_gcode('M400')

    printer.send_gcode('G90')

    return

def create_images(x_start: int, x_inc: int, x_end: int, y_start: int, y_inc: int, y_end: int, stabilize_delay: float, skipped_points=[]) -> None:
    # Open Printer
    logger.info('Opening printer...')
    printer = moonpy.MoonrakerPrinter('http://localhost')
    wait_for_printer(printer)

    # Open Webcam
    logger.info('Opening webcam...')
    camera = CameraWrapper()

    home_printer(printer, camera, stabilize_delay)

    ##
    ## START CAPTURE
    ##
    start_time = datetime.datetime.now()
    images = None

    # Iterate over a grid
    rows = len(range(y_start, y_end + y_inc, y_inc))
    cols = len(range(x_start, x_end + x_inc, x_inc))
    pbar = tqdm.tqdm(desc='Capturing Images', total=rows * cols)
    
    forward = True
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
    return images
