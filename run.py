import logging
import os
import datetime
import argparse

import numpy as np

from stitcher import main
from grid import create_grid
from machine import create_images
from image_io import write_images, load_images

parser = argparse.ArgumentParser(
                    prog='Visual Inspection Control',
                    description='Runs and manages the visual inspection machine',
                    epilog='Contact Nathan Nguyen for script help')

parser.add_argument('-r', '--reuse', action='store_true', help='Reuse the latest folder')
parser.add_argument('-d', '--dir', type=str, help='Load a specific folder')
parser.add_argument('-g', '--no-grid', action='store_true', help='Disable raw grid creation')
parser.add_argument('-v', '--verbose',
                    action='store_true')  # on/off flag
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s - %(name)-24s - %(levelname)-7s - %(message)s (%(filename)s:%(lineno)d)', level=logging.DEBUG if args.verbose else logging.INFO)
logger = logging.getLogger('main')

##
## SETTINGS
##
output_dir = 'Pictures'

x_start = 0
x_end = 190
x_inc = 20 # 45

y_start = 45
y_end = 205
y_inc = 10 # 25

stabilize_delay = 1 # 2.2

stitched_scale = 16

skipped_points = [
  # (0, 45),
  # (0, 55),
  # (0, 65),
  # (0, 75),
  # (0, 85),
  # (20, 45),

  # (0, 175),
  # (0, 185),
  # (0, 195),
  # (0, 205),
  # (20, 205),

  # (200, 155),
  # (200, 165),
  # (200, 175),
  # (200, 185),
  # (200, 195),
  # (200, 205),

  # (180, 45),
  # (180, 55),
  # (180, 65),
  # (200, 45),
  # (200, 55),
  # (200, 65),
  # (200, 75),
  # (200, 85),
  # (200, 95)
]

##
## LOAD RESOURCES
##
output_dir = os.path.expanduser(output_dir)
if args.reuse:
  all_subdirs = [os.path.join(output_dir, d) for d in os.listdir(output_dir)]
  folder = max(all_subdirs, key=os.path.getmtime)

  logger.info(f"Loading images from folder {folder}")
  images = load_images(folder, x_start, x_inc, x_end, y_start, y_inc, y_end)
elif args.dir:
  folder = args.dir
  logger.info(f"Loading images from folder {folder}")
  images = load_images(folder, x_start, x_inc, x_end, y_start, y_inc, y_end)
else:
  logger.info("Scanning images")
  start_time = datetime.datetime.now()
  images = create_images(x_start, x_inc, y_end, y_start, y_inc, y_end, stabilize_delay, skipped_points)

  # Make Dirs
  folder = os.path.join(output_dir, str(start_time))

  write_images(images, folder, x_start, x_inc, y_start, y_inc)

logger.info("Loaded images")

##
## RUN ANALYSIS
##
if not args.no_grid:
  logger.info("Creating grid")
  create_grid(images, os.path.join(folder, 'grid.jpg'), stitched_scale, x_start, x_inc, y_start, y_inc)

logger.info("Converting images for stitching")
for row_num, row in enumerate(images):
    for col_num, image in enumerate(row):
        images[row_num][col_num] = np.array(image.convert('RGB'))
        image.close()

logger.info("Stitchng images")
main(images, 0.2, 0.2, os.path.join(folder, 'stitched.png'))

logger.info("Finished, exiting...")