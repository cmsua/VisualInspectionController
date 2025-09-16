import argparse
import datetime
import logging
import os
import time
import sys

import tqdm.contrib.logging
import yaml

from barcodes import scan_data_matrix
from cameras.logitech import LogitechWebcam
from cameras.microscope import Microscope
from grid import create_grid
from image_io import write_images
from machines.printer import MoonrakerMachine, SerialMachine
from machines.reuse import ReuseMachine
from non_stitch_prepro import main

##
## SETTINGS
##

kernel_size = 340

stitched_scale = 1

vertical_clip_fraction = 0.265
horizontal_clip_fraction = 0.3

##
## LOAD RESOURCES
##

parser = argparse.ArgumentParser(
    prog="Visual Inspection Control",
    description="Runs and manages the visual inspection machine",
    epilog="Contact Nathan Nguyen for script help",
)

parser.add_argument("-c", "--config", help="App config file")

parser.add_argument(
    "-r", "--reuse", action="store_true", help="Reuse the latest folder"
)
parser.add_argument(
    "-g", "--grid", action="store_true", help="Enable raw grid creation"
)
parser.add_argument(
    "-s", "--silent", action="store_true", help="Disable beeping when done"
)
parser.add_argument(
    "-d", "--debug", action="store_true", help="Output verbose logging"
)
parser.add_argument(
    "--ref", action="store_true", help="Whether the board is a reference board"
)

if __name__ == "__main__":
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(name)-24s - %(levelname)-7s - %(message)s (%(filename)s:%(lineno)d)",
        level=logging.DEBUG if args.debug else logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger("main")

    with tqdm.contrib.logging.logging_redirect_tqdm():
        # Load Configs
        logger.debug(f"Loading config file {args.config}")
        with open(args.config, "r") as file:
            config = yaml.safe_load(file)

        # Overrides
        if args.reuse:
            config["machine"]["type"] = "reuse"
            if type(args.reuse) == str:
                config["machine"]["path"] = args.reuse

        # Load Machine
        machine_type = config["machine"]["type"]
        if machine_type == "moonraker":
            machine = MoonrakerMachine(config["machine"])
        elif machine_type == "serial":
            machine = SerialMachine(config["machine"])
        elif machine_type == "reuse":
            machine = ReuseMachine(config["machine"])
        else:
            raise ValueError(f"Invalid machine {machine_type}")

        # Load Camera
        camera_type = config["camera"]["type"]
        if camera_type == "logitech":
            camera = LogitechWebcam(config["camera"])
        elif camera_type == "microscope":
            camera = Microscope(config["camera"])
        else:
            raise ValueError(f"Invalid camera {camera_type}")

        with camera as camera:
            with machine as machine:
                logger.info("Scanning images")
                images = machine.get_images(camera)

        barcode = None
        try:
            barcode = scan_data_matrix(images[10, 3])
        except Exception as e:
            logger.critical(e)
        if barcode is not None:
            logger.info(f"Identified barcode {barcode}")
        else:
            barcode = "Not Found"
            logger.critical("Could not find barcode in image")

        start_time = datetime.datetime.now()
        folder = os.path.join(
            config["output_directory"], barcode + "_" + str(start_time)
        )

        # Saving images
        logger.info("Saving images")
        write_images(images, folder, args.debug)

        logger.info("Loaded images")

        ##
        ## RUN ANALYSIS
        ##

        if args.grid:
            logger.info("Creating grid")
            create_grid(images, os.path.join(folder, "grid.jpg"), stitched_scale)

        logger.info("Adjusting and cropping images")
        main(
            images,
            vertical_clip_fraction,
            horizontal_clip_fraction,
            kernel_size=kernel_size,
            output_dir=folder,
            is_baseline=args.ref,
        )

        # Beep
        logger.info("Finished, exiting...")
        if not args.silent:
            for i in range(5):
                print("\a")
                time.sleep(1)
