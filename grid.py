import logging

import cv2 as cv2
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger("stitcher")


def create_grid(images, output_file: str, stitched_scale: int):
    logger.debug("Creating Grid")

    # Calculate width and height based on the first image
    width = images[0][0].shape[1] // stitched_scale
    height = images[0][0].shape[0] // stitched_scale

    # Calculate the total width and height of the final grid
    total_width = width * len(images[0])
    total_height = height * len(images)

    # Create a new image with the calculated total dimensions
    image = Image.new("RGB", (total_width, total_height))

    # Loop through the rows and columns to place images
    for y_num, row in enumerate(images):
        for x_num, captured in enumerate(row):
            if captured is None:
                continue

            # Calculate the position of the image in the final grid
            image_pos = (x_num * width, y_num * height)

            # Resize and convert the captured image to match the grid
            segment = cv2.resize(
                cv2.cvtColor(captured, cv2.COLOR_BGR2RGB), (width, height)
            )
            segment = Image.fromarray(segment, mode="RGB")
            image.paste(segment, image_pos)

    # Set up the drawing for adding text annotations
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", height // 20
    )

    for y_num, row in enumerate(images):
        for x_num, captured in enumerate(row):
            image_pos = (x_num * width, y_num * height)

            # Determine if we should add custom machine coordinates
            draw.text(image_pos, f"{y_num}, {x_num}", (255, 255, 255), font=font)

    logger.debug("Saving Grid")
    image.save(output_file)
