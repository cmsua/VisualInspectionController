import logging
import os
import datetime
import argparse
import time
import numpy as np

import tqdm.contrib.logging 

from non_stitch_prepro import main
from grid import create_grid
from machine import create_images
from image_io import write_images, load_images

##
## SETTINGS
##
output_dir = 'Pictures'

x_start = 49
x_end = 225
x_inc = 22

y_start = 43
y_end = 186
# y_end = 56
y_inc = 13

kernel_size = 340

stabilize_delay = 2.2

stitched_scale = 1

skipped_points = []

vertical_clip_fraction = .265
horizontal_clip_fraction = .3

##
## LOAD RESOURCES
##

parser = argparse.ArgumentParser(
                    prog='Visual Inspection Control',
                    description='Runs and manages the visual inspection machine',
                    epilog='Contact Nathan Nguyen for script help')

parser.add_argument('-r', '--reuse', action='store_true', help='Reuse the latest folder')
parser.add_argument('-d', '--dir', type=str, help='Load a specific folder')
parser.add_argument('-g', '--grid', action='store_true', help='Enable raw grid creation')
parser.add_argument('-s', '--silent', action='store_true', help='Disable beeping when done')
parser.add_argument('-v', '--verbose',
                    action='store_true')  # on/off flag
parser.add_argument('-b', '--baseline_path', type=str, default=None, help='Baseline board image paths. ' + \
                    'Only use if saving baseline boards.')
parser.add_argument('-n', '--numpy', action='store_true', help='Use images from numpy directly rather than loading from pngs')

if __name__ == '__main__':
  args = parser.parse_args()

  logging.basicConfig(format='%(asctime)s - %(name)-24s - %(levelname)-7s - %(message)s (%(filename)s:%(lineno)d)', level=logging.DEBUG if args.verbose else logging.INFO)
  logger = logging.getLogger('main')

  with tqdm.contrib.logging.logging_redirect_tqdm():
    output_dir = os.path.expanduser(output_dir)
    if args.reuse:
      all_subdirs = [os.path.join(output_dir, d) for d in os.listdir(output_dir)]
      folder = max(all_subdirs, key=os.path.getmtime)

      logger.info(f'Loading images from folder {folder}')
      if args.numpy:
        np_images = np.load(os.path.join(folder, 'images.npy'))
      else:
        np_images = load_images(folder)
    elif args.dir:
      folder = args.dir
      logger.info(f'Loading images from folder {folder}')
      np_images = load_images(folder)
    else:
      logger.info('Scanning images')
      start_time = datetime.datetime.now()
      np_images = create_images(x_start, x_inc, x_end, y_start, y_inc, y_end, stabilize_delay, skipped_points)

      if args.baseline_path is not None:
        # Make Dirs for baseline path
        folder = os.path.join(output_dir, args.baseline_path)
      else:
        # Make Dirs
        folder = os.path.join(output_dir, str(start_time))

      # Saving images
      logger.info('Saving images')
      write_images(np_images, folder, x_start, x_inc, y_start, y_inc, args.verbose)

    logger.info('Loaded images')

    ##
    ## RUN ANALYSIS
    ##
    if args.grid:
      logger.info('Creating grid')
      create_grid(np_images, os.path.join(folder, 'grid.jpg'), stitched_scale, x_start, x_inc, y_start, y_inc)


    logger.info('Adjusting and cropping images')
    main(np_images, vertical_clip_fraction, horizontal_clip_fraction,
         kernel_size=kernel_size, output_dir=folder, is_baseline=args.baseline_path is not None)

    # Beep
    logger.info('Finished, exiting...')
    if not args.silent:
      for i in range(5):
        print('\a')
        time.sleep(1)
