import math
import logging
import os

import numpy as np
import cv2
import tqdm

from grid import create_grid

logger = logging.getLogger('main')

def compute_gradients(img):
    # Assume img is grayscale
    grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    return grad_mag

def find_best_overlap(imgA, imgB,
                      min_overlap=1,
                      max_overlap=10,
                      pixel_weight=0.5,
                      grad_weight=0.5):
    """Find the best horizontal overlap between imgA and imgB."""
    # Ensure same height
    assert imgA.shape[0] == imgB.shape[0], \
        'Images must have the same height for horizontal stitching.'
    
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
    best_overlap = 1  # fallback

    # Scan possible overlaps
    for overlap in range(min_overlap, max_overlap + 1):
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

def find_best_vertical_overlap(imgTop, imgBottom,
                               min_overlap=10,
                               max_overlap=None,
                               pixel_weight=0.5,
                               grad_weight=0.5):
    """
    Find the best vertical overlap between imgTop and imgBottom
    by cropping both to their common width.
    """

    # 1. Determine common width
    common_width = min(imgTop.shape[1], imgBottom.shape[1])

    # 2. Crop both images to the same (common) width
    imgTop_cropped = imgTop[:, :common_width]
    imgBottom_cropped = imgBottom[:, :common_width]

    # Convert to grayscale if needed
    if len(imgTop_cropped.shape) == 3:
        grayTop = cv2.cvtColor(imgTop_cropped, cv2.COLOR_BGR2GRAY)
    else:
        grayTop = imgTop_cropped.astype(np.uint8)
    if len(imgBottom_cropped.shape) == 3:
        grayBottom = cv2.cvtColor(imgBottom_cropped, cv2.COLOR_BGR2GRAY)
    else:
        grayBottom = imgBottom_cropped.astype(np.uint8)

    # Compute gradients
    gradTop = compute_gradients(grayTop)
    gradBottom = compute_gradients(grayBottom)

    hTop, _ = grayTop.shape
    hBottom, _ = grayBottom.shape

    # If max_overlap is None, use a heuristic
    if max_overlap is None:
        max_overlap = min(hTop, hBottom) // 2

    best_score = float('inf')
    best_overlap = min_overlap

    # Scan possible overlaps
    for overlap in range(min_overlap, max_overlap + 1):
        # Bottom strip of top image
        stripTop = grayTop[hTop - overlap:, :]
        gradStripTop = gradTop[hTop - overlap:, :]

        # Top strip of bottom image
        stripBottom = grayBottom[:overlap, :]
        gradStripBottom = gradBottom[:overlap, :]

        # Compute pixel difference
        pixel_diff = np.mean(
            np.abs(stripTop.astype(np.float32) - stripBottom.astype(np.float32))
        )
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
    """Blend two images (either horizontally or vertically) using simple alpha blending."""
    if direction == 'horizontal':
        # Overlapping region is width-based
        overlapA = imgA[:, -overlap_width:]
        overlapB = imgB[:, :overlap_width]
        
        alpha = np.linspace(1, 0, overlap_width).reshape(1, -1, 1)
        blended_region = (alpha * overlapA + (1 - alpha) * overlapB).astype(np.uint8)
        
        stitched = np.concatenate([imgA[:, :-overlap_width], blended_region, imgB[:, overlap_width:]], axis=1)
        return stitched
    else:
        # Overlapping region is height-based
        overlapA = imgA[-overlap_width:, :]
        overlapB = imgB[:overlap_width, :]
        
        alpha = np.linspace(1, 0, overlap_width).reshape(-1, 1, 1)
        blended_region = (alpha * overlapA + (1 - alpha) * overlapB).astype(np.uint8)
        
        stitched = np.concatenate([imgA[:-overlap_width, :], blended_region, imgB[overlap_width:, :]], axis=0)
        return stitched

def main(images, 
         vert_clip_fraction: float, 
         horz_clip_fraction: float, 
         output_dir: str, 
         write_intermediates: bool = False
         ):
    """
    Modified main pipeline:
      1. Clip images.
      2. For each row, stitch horizontally from left to right, 
         computing best overlap for each pair of neighbors.
      3. Stitch the resulting row strips top to bottom, again computing
         best vertical overlap for each pair of adjacent rows.

    Args:
        images: 2D list of images [row][col]
        vert_clip_fraction: fraction to clip from top/bottom
        horz_clip_fraction: fraction to clip from left/right
        output_dir: Directory to save outputs
        write_intermediates: if True, save some intermediate results
    """
    
    rows = len(images)
    columns = len(images[0])

    # Assume all images have the same shape initially:
    total_image_shape = images[0][0].shape
    vert_clip = math.floor(total_image_shape[0] * vert_clip_fraction)
    horz_clip = math.floor(total_image_shape[1] * horz_clip_fraction)
    
    logger.debug(f'Clipping images, from {total_image_shape} '
                 f'to {vert_clip}, {horz_clip} (fractions {vert_clip_fraction}, {horz_clip_fraction})')

    # Prepare array to store clipped images
    new_h = total_image_shape[0] - 2 * vert_clip
    new_w = total_image_shape[1] - 2 * horz_clip
    clipped_images = np.zeros((rows, columns, new_h, new_w, 3), dtype=np.uint8)

    # Clip all images
    pbar = tqdm.tqdm(desc='Clipping Images', total=rows * columns)
    for row_num in range(rows):
        for col_num in range(columns):
            image = images[row_num][col_num]
            clipped_img = clip_image(
                image,
                top_clip=vert_clip,
                bottom_clip=vert_clip,
                left_clip=horz_clip,
                right_clip=horz_clip
            )
            # Store in the same (row, col) positions
            clipped_images[row_num, col_num] = clipped_img
            pbar.update()
    pbar.close()

    if write_intermediates and output_dir is not None:
        create_grid(clipped_images, 
                    os.path.join(output_dir, 'stitcher-clipped.png'), 
                    4)

    # Free up memory from the original images
    images = None

    logger.info('Stitching horizontally within each row...')
    stitched_rows_list = []
    
    # For each row, do a left-to-right stitch:
    for row_index in tqdm.tqdm(range(rows), desc='Stitching Rows'):
        row_strip = clipped_images[row_index, 0]
        
        for col_index in range(1, columns):
            # Determine best horizontal overlap for these two images
            overlap = find_best_overlap(row_strip, 
                                        clipped_images[row_index, col_index],
                                        min_overlap=1,
                                        max_overlap=None)
            
            row_strip = blend_images(row_strip, 
                                     clipped_images[row_index, col_index],
                                     overlap_width=overlap,
                                     direction='horizontal')
        
        stitched_rows_list.append(row_strip)
    
    # Now we have a list of stitched rows (each row is one full panorama for that row).

    logger.info('Stitching all rows vertically...')
    final_image = stitched_rows_list[0]
    for row_index in tqdm.tqdm(range(1, rows), desc='Stitching Columns'):
        # For each adjacent pair of row strips, find best vertical overlap
        overlap = find_best_vertical_overlap(final_image, 
                                             stitched_rows_list[row_index],
                                             min_overlap=1,
                                             max_overlap=None)
        final_image = blend_images(final_image, 
                                   stitched_rows_list[row_index],
                                   overlap_width=overlap,
                                   direction='vertical')

    # Save final results
    if output_dir is not None:
        logger.debug('Saving final stitched image...')
        if write_intermediates:
            cv2.imwrite(os.path.join(output_dir, 'stitcher-out.png'), final_image)
        # Also save as .npy for convenience
        np.save(os.path.join(output_dir, 'stitcher-out.npy'), final_image)

    return final_image
