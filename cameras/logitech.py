import logging
import os
import queue
import sys
import threading

import cv2 as cv2

from objects import Camera

logger = logging.getLogger("camera")


class LogitechWebcam(Camera):
    def __init__(self, config):
        self._id = config["id"]
        self.commands = [
            f"v4l2-ctl -d {self._id} -c zoom_absolute=381,focus_automatic_continuous=0",
            f"v4l2-ctl -d {self._id} -c focus_absolute=90",
            f"v4l2-ctl -d {self._id} -c backlight_compensation=0,white_balance_automatic=0,auto_exposure=1,exposure_dynamic_framerate=0",
            f"v4l2-ctl -d {self._id} -c exposure_time_absolute=1007",
            f"v4l2-ctl -d {self._id} -c gain=40",
            f"v4l2-ctl -d {self._id} -c saturation=102",
            f"python3 ~/cameractrls/cameractrls.py -d {self._id} -c logitech_led1_mode=off",
        ]

    def __enter__(self):
        # Kill other camera uses
        os.system("killall cheese")

        # Settings
        for command in self.commands:
            logger.debug(f"Calling {command}")
            rc = os.system(command)
            if rc != 0:
                raise RuntimeError(f"Command returned with code {rc}: {command}")

        # Open camera
        self.cap = cv2.VideoCapture(self._id)
        if not self.cap or not self.cap.isOpened():
            logger.critical("Cannot open camera!")
            sys.exit(1)

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4096)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self.run_reader)
        self.thread.daemon = True

        self.run = True
        self.thread.start()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        logger.info("Closing...")
        self.run = False
        self.thread.join()
        self.cap.release()

    def run_reader(self):
        while self.run:
            ret, frame = self.cap.read()
            if not ret:
                logger.critical("Failed to capture image!")
                break
            if not self.queue.empty():
                try:
                    self.queue.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.queue.put(frame)

    def get_image(self):
        segment = self.queue.get()
        logger.debug("Image captured")

        segment = cv2.rotate(segment, cv2.ROTATE_180)

        logger.debug("Image converted and rotated")
        return segment
