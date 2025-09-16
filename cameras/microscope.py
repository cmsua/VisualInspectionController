import logging
import os
import queue
import sys
import threading

import cv2 as cv2

from objects import Camera

logger = logging.getLogger("camera")


class Microscope(Camera):
    def __init__(self, config):
        self._id = config["id"]

    def __enter__(self):
        # Kill other camera uses
        os.system("killall cheese")

        # Open camera
        self.cap = cv2.VideoCapture(self._id)
        if not self.cap or not self.cap.isOpened():
            logger.critical("Cannot open camera!")
            sys.exit(1)

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
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

        # segment = cv2.rotate(segment, cv2.ROTATE_180)
        # logger.debug('Image converted and rotated')

        return segment
