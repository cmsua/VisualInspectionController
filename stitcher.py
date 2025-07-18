import math
import logging
import os

import numpy as np
import cv2
import tqdm

from grid import create_grid

logger = logging.getLogger('main')

def compute_gradients(img):
    # Assume img is grayscale for simplicity
    grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    return grad_mag

def find_best_overlap(imgA, imgB,
                    #   min_overlap=50,
                    #   max_overlap=50,
                      min_overlap=0,
                      max_overlap=1,
                      pixel_weight=0.5,
                      grad_weight=0.5):
    # Ensure same height
    assert imgA.shape[0] == imgB.shape[0], 'Images must have the same height for horizontal stitching.'
    
    # Convert to grayscale if needed
    if len(imgA.shape) == 3:
        grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    else:
        grayA = imgA.astype(np.uint8)
    if len(imgB.shape) == 3:
        grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)
    else:
        grayB = imgB.astype(np.uint8)
    
    # Compute gradients
    gradA = compute_gradients(grayA)
    gradB = compute_gradients(grayB)
    
    _, wA = grayA.shape
    _, wB = grayB.shape
    
    if max_overlap is None:
        max_overlap = min(wA, wB) // 2  # heuristic
    
    best_score = float('inf')
    best_overlap = None
    
    # Scan possible overlaps
    for overlap in range(min_overlap, max_overlap+1):
        # Extract overlap strips
        stripA = grayA[:, wA - overlap:]   # right side of A
        stripB = grayB[:, :overlap]        # left side of B
        
        # Extract corresponding gradients
        gradStripA = gradA[:, wA - overlap:]
        gradStripB = gradB[:, :overlap]
        
        # Compute pixel difference (mean absolute difference)
        pixel_diff = np.mean(np.abs(stripA.astype(np.float32) - stripB.astype(np.float32)))
        
        # Compute gradient difference
        grad_diff = np.mean(np.abs(gradStripA - gradStripB))
        
        # Weighted score
        score = pixel_weight * pixel_diff + grad_weight * grad_diff
        
        # Track best
        if score < best_score:
            best_score = score
            best_overlap = overlap
    
    return best_overlap
    
def compute_gradients(img):
    # Assume img is grayscale
    grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    return grad_mag

def find_best_vertical_overlap(imgTop, imgBottom,
                            #    min_overlap=50,
                            #    max_overlap=50,
                               min_overlap=0,
                               max_overlap=1,
                               pixel_weight=0.5,
                               grad_weight=0.5):
    # Ensure same width
    assert imgTop.shape[1] == imgBottom.shape[1], 'Images must have the same width for vertical stitching.'

    # Convert to grayscale if needed
    if len(imgTop.shape) == 3:
        grayTop = cv2.cvtColor(imgTop, cv2.COLOR_BGR2GRAY)
    else:
        grayTop = imgTop.astype(np.uint8)
    if len(imgBottom.shape) == 3:
        grayBottom = cv2.cvtColor(imgBottom, cv2.COLOR_BGR2GRAY)
    else:
        grayBottom = imgBottom.astype(np.uint8)

    # Compute gradients
    gradTop = compute_gradients(grayTop)
    gradBottom = compute_gradients(grayBottom)

    hTop, _ = grayTop.shape
    hBottom, _ = grayBottom.shape

    if max_overlap is None:
        max_overlap = min(hTop, hBottom) // 2  # heuristic

    best_score = float('inf')
    best_overlap = None

    # Scan possible overlaps vertically
    for overlap in range(min_overlap, max_overlap + 1):
        # Bottom strip of top image
        stripTop = grayTop[hTop - overlap:, :]
        gradStripTop = gradTop[hTop - overlap:, :]

        # Top strip of bottom image
        stripBottom = grayBottom[:overlap, :]
        gradStripBottom = gradBottom[:overlap, :]

        # Compute pixel difference
        pixel_diff = np.mean(np.abs(stripTop.astype(np.float32) - stripBottom.astype(np.float32)))

        # Compute gradient difference
        grad_diff = np.mean(np.abs(gradStripTop - gradStripBottom))

        # Weighted score
        score = pixel_weight * pixel_diff + grad_weight * grad_diff

        if score < best_score:
            best_score = score
            best_overlap = overlap

    return best_overlap
    
def clip_image(img, top_clip, bottom_clip, left_clip, right_clip):
    return img[top_clip:img.shape[0]-bottom_clip, left_clip:img.shape[1]-right_clip]

def blend_images(imgA, imgB, overlap_width, direction='horizontal'):
    if direction == 'horizontal':
        # Horizontal blending as before
        overlapA = imgA[:, -overlap_width:]
        overlapB = imgB[:, :overlap_width]
        
        alpha = np.linspace(1, 0, overlap_width).reshape(1, -1, 1)
        blended_region = (alpha * overlapA + (1 - alpha) * overlapB).astype(np.uint8)
        
        stitched = np.concatenate([imgA[:, :-overlap_width], blended_region, imgB[:, overlap_width:]], axis=1)
        return stitched
    else:
        # Vertical blending
        overlapA = imgA[-overlap_width:, :]
        overlapB = imgB[:overlap_width, :]
        
        alpha = np.linspace(1, 0, overlap_width).reshape(-1, 1, 1)
        blended_region = (alpha * overlapA + (1 - alpha) * overlapB).astype(np.uint8)
        
        stitched = np.concatenate([imgA[:-overlap_width, :], blended_region, imgB[overlap_width:, :]], axis=0)
        return stitched

def main(images, vert_clip_fraction: float, horz_clip_fraction: float, output_dir: str, write_intermediates: bool = False):
    total_image_shape = images[0][0].shape
    vert_clip = math.floor(total_image_shape[0]*vert_clip_fraction)
    horz_clip = math.floor(total_image_shape[1]*horz_clip_fraction)
    rows = len(images)
    columns = len(images[0])

    logger.debug(f'Clipping images, from {total_image_shape} to {vert_clip}, {horz_clip} (fractions {vert_clip_fraction}, {horz_clip_fraction})')
    pbar = tqdm.tqdm(desc='Clipping Images', total=rows*columns)

    clipped_images = np.zeros((rows, columns, total_image_shape[0] - 2 * vert_clip, total_image_shape[1] - 2 * horz_clip, 3), dtype=np.uint8)
    for row_num, row in enumerate(images):
        for col_num, image in enumerate(row):
            clipped_img = clip_image(image,
                                     top_clip=vert_clip,
                                     bottom_clip=vert_clip,
                                     left_clip=horz_clip,
                                     right_clip=horz_clip)
            clipped_images[rows - row_num - 1][col_num] = clipped_img
            pbar.update()
    pbar.close()

    logger.debug(f'Clipped image shape: {clipped_images[0][0].shape}')
    if write_intermediates and output_dir is not None:
        create_grid(clipped_images, os.path.join(output_dir, 'stitcher-clipped.png'), 4)

    # Memory cleanup
    images = None

    #center_x = len(clipped_images) // 2
    #center_y = len(clipped_images[0]) // 2
    center_x = 0
    center_y = 5
    logger.info(f'Using center {center_x}, {center_y}')

    logger.info('Finding best overlaps')
    # Compute horizontal overlap using the first two images in the top row
    horiz_overlap = find_best_overlap(clipped_images[center_x][center_y], clipped_images[center_x][center_y + 1])
    logger.info(f'Found horizontal overlap {horiz_overlap}')

    # Compute vertical overlap using the first two images in the first column
    vert_overlap = find_best_vertical_overlap(clipped_images[center_x][center_y], clipped_images[center_x + 1][center_y])
    logger.info(f'Found vertical overlap {vert_overlap}')

    logger.debug(f'Stitching {rows} rows')
    # Now use horiz_overlap for stitching each row horizontally
    stitched_rows = None
    for row_index in tqdm.tqdm(range(rows), desc='Stitching Rows'):
        logger.debug(f'Stitching row {row_index}')
        row_strip = clipped_images[row_index][0]
        for col_index in range(1, columns):
            # Use the determined horizontal overlap width
            row_strip = blend_images(row_strip,
                                     clipped_images[row_index][col_index],
                                     overlap_width=horiz_overlap,
                                     direction='horizontal')
        
        if stitched_rows is None:
            stitched_rows = np.zeros((rows, *row_strip.shape), dtype=np.uint8)
        stitched_rows[row_index] = row_strip
    
    # Memory cleanup
    clipped_images = None

    logger.debug('Stitching to final image')
    # Now stitch the rows together vertically using the determined vert_overlap
    final_image = stitched_rows[0]
    for r in tqdm.tqdm(range(1, rows), desc='Stitching Columns'):
        logger.debug(f'Stitching column {r}')
        final_image = blend_images(final_image,
                                   stitched_rows[r],
                                   overlap_width=vert_overlap,
                                   direction='vertical')
        
    if output_dir is not None:
        logger.debug('Saving...')
        if write_intermediates:
            cv2.imwrite(os.path.join(output_dir, 'stitcher-out.png'), final_image)

        np.save(os.path.join(output_dir, 'stitcher-out.npy'), final_image)
    return final_image