import logging

import cv2 as cv2
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger('stitcher')

def create_grid(images, output_file, stitched_scale, x_start, x_inc, y_start, y_inc):
  logger.info('Creating Grid')
  width = images[0][0].shape[1] // stitched_scale
  height = images[0][0].shape[0] // stitched_scale
  total_width = width * len(images[0])
  total_height = height * len(images)

  image = Image.new('RGB', (total_width, total_height))

  for y_num, row in enumerate(images):
    for x_num, captured in enumerate(row):
      if captured is None:
        continue
      
      image_pos = (x_num * width, (len(images) - 1 - y_num) * height)

      segment = Image.fromarray(cv2.cvtColor(captured, cv2.COLOR_BGR2RGB), mode='RGB').resize((width, height))
      image.paste(segment, image_pos)

  draw = ImageDraw.Draw(image)
  font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', 16)

  counter = 0
  for y_num, row in enumerate(images):
    for x_num, captured in enumerate(row):
      machine_x = x_start + x_num * x_inc
      machine_y = y_start + y_num * y_inc
      image_pos = (x_num * width, (len(images) - 1 - y_num) * height)
      
      draw.text(image_pos, f'({y_num}, {x_num}) - {counter} - X{machine_x}Y{machine_y}', (255,255,255), font=font)
      counter = counter + 1

  logger.info('Saving Grid')
  image.save(output_file)