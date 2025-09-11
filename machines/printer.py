import abc
import datetime
import logging
import os
import time

import moonrakerpy as moonpy
import numpy as np
import serial
import tqdm

from objects import Camera, Machine

logger = logging.getLogger("PrinterMachines")


class PrinterMachine(Machine):
    def __init__(self, config: dict) -> None:
        # Save coordinates
        coordinates = config["coordinates"]
        self._x_start = coordinates["x_start"]
        self._x_inc = coordinates["x_inc"]
        self._x_end = coordinates["x_end"]
        self._y_start = coordinates["y_start"]
        self._y_inc = coordinates["y_inc"]
        self._y_end = coordinates["y_end"]

        # Save misc
        self._speed = config["speed"]
        self._stabilize_delay = config["stabilize_delay"]
        self._home = config["home"]

    def __enter__(self) -> Machine:
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    # Mechanical homing
    def home(self) -> None:
        logger.debug("Homing X+Y mechanically")
        for line in self._home:
            self.send_gcode(line)

    @abc.abstractmethod
    def send_gcode(self, str) -> None:
        pass

    def get_images(self, camera: Camera) -> np.typing.ArrayLike:
        x_images = int((self._x_end - self._x_start) / self._x_inc + 1)
        y_images = int((self._y_end - self._y_start) / self._y_inc + 1)
        images = None

        start_time = datetime.datetime.now()

        # Snake logic
        forward = True
        for y in tqdm.trange(y_images):
            # Handle direction switching
            x_points = range(x_images)
            if not forward:
                x_points = reversed(x_points)

            for x in tqdm.tqdm(x_points):
                # Go to image and capture
                logger.debug(
                    f"Capturing image {y * y_images + x} out of {y_images * x_images}"
                )
                self.go_to_image(x, y)

                logger.debug("Reading frame")
                frame = camera.get_image()

                if images is None:
                    logger.debug("Creating images object")
                    images = np.zeros(
                        (y_images, x_images, *frame.shape), dtype=np.uint8
                    )

                images[y][x] = frame

            forward = not forward

        seconds = (datetime.datetime.now() - start_time).seconds
        logger.info(f"Finished in {seconds}s")

        return images

    def go_to_image(self, x_index: int, y_index: int) -> np.typing.ArrayLike:
        x = self._x_start + (self._x_inc * x_index)
        y = self._y_start + (self._y_inc * y_index)
        self.send_gcode(f"G1 X{x} Y{y} F{self._speed}")
        self.send_gcode(f"M400")
        time.sleep(self._stabilize_delay)


class MoonrakerMachine(PrinterMachine):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._address = config["address"]

    def __enter__(self) -> PrinterMachine:
        super().__enter__()
        # Connect
        logger.debug(f"Connecting to {self._address}")
        self._printer = moonpy.MoonrakerPrinter(self._address)

        # Check for online
        if self.get_status() != "ready":
            logger.warning('Printer status is not "ready", restarting...')
            self._printer.post("/printer/restart")
            time.sleep(3)

            logger.debug("Restarting printer firmware")
            self._printer.post("/printer/firmware_restart")
            time.sleep(4)

            # Make sure it worked
            if self.get_status() != "ready":
                logger.critical("Printer failed to come online. Exiting.")
                raise RuntimeError("Printer failed to initialize")

        # Home Machine
        self.home()

        return self

    def send_gcode(self, gcode: str) -> None:
        self._printer.send_gcode(gcode)

    def __exit__(self, exc_type, exc_value, traceback):
        self._printer.send_gcode("G0 X0 Y0")
        
    # For startup
    def get_status(self) -> str:
        return self._printer.get("/server/info")["result"]["klippy_state"]


class SerialMachine(PrinterMachine):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._port = config["port"]
        self._baud = config["baud"]

    def __enter__(self) -> PrinterMachine:
        super().__enter__()
        # Connect
        logger.debug(f"Connecting to {self._port} (baud {self._baud})")
        self._serial = serial.Serial(self._port, self._baud, timeout=1)

        # Home Machine
        self.home()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._serial.write("G0 X0 Y0\n".encode())
        self._serial.close()

    def send_gcode(self, gcode: str) -> None:
        logger.debug(f'Sending gcode {gcode}')
        self._serial.write((gcode + '\n').encode())
        
        # Wait for 'ok'
        line = ''
        while line != 'ok':
            line = self._serial.readline().decode().strip()
            logger.debug(f'Received line {line}')
