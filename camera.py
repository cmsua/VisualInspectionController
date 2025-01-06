import queue
import threading
import logging

import cv2 as cv2
from PIL import Image

logger = logging.getLogger('camera')

class CameraWrapper():
  def __init__(self):
    self.cap = cv2.VideoCapture(0)
    if not self.cap or not self.cap.isOpened():
        logger.critical('Cannot open camera!')
        exit()

    self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4096)
    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    self.queue = queue.Queue()
    self.thread = threading.Thread(target=self.run_reader)
    self.thread.daemon = True

    self.run = True
    self.thread.start()

  def run_reader(self):
    while self.run:
      ret, frame = self.cap.read()
      if not ret:
        logger.critical('Failed to capture image!')
        break
      if not self.queue.empty():
        try:
          self.queue.get_nowait() # discard previous (unprocessed) frame
        except queue.Empty:
          pass
      self.queue.put(frame)
  
  def get_image(self):
    segment = Image.fromarray(cv2.cvtColor(self.queue.get(), cv2.COLOR_BGR2RGB))
    segment = segment.transpose(Image.ROTATE_180)
    return segment

  def close(self):
    logger.info("Closing...")
    self.run = False
    self.thread.join()
    self.cap.release()
