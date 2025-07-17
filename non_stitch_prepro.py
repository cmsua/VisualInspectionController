import math
import logging
import os

import numpy as np
import cv2
import tqdm
import matplotlib.pyplot as plt

from PIL import Image

logger = logging.getLogger('main')

def align_image_general(
    image: Image.Image,
    src_pts: np.ndarray,
    dst_pts: np.ndarray
) -> Image.Image:
    """
    Align an image by computing either a similarity (2-point) or full affine (3+ points) transform.

    Args:
        image: PIL Image to transform.
        src_pts: np.ndarray of shape (N,2): detected keypoints in the input image.
        dst_pts: np.ndarray of shape (N,2): corresponding target positions in the reference frame.

    Returns:
        A new PIL Image that has been aligned to match dst_pts.
    """
    # Convert to float32
    src = np.array(src_pts, dtype=np.float32)
    dst = np.array(dst_pts, dtype=np.float32)

    if src.shape[0] == 2:
        # Similarity transform (rotation + uniform scale + translation)
        src_pt0, src_pt1 = src
        dst_pt0, dst_pt1 = dst

        # Compute input and output angles
        angle_in = np.arctan2(src_pt1[1] - src_pt0[1], src_pt1[0] - src_pt0[0])
        angle_out = np.arctan2(dst_pt1[1] - dst_pt0[1], dst_pt1[0] - dst_pt0[0])
        angle = (angle_out - angle_in) * 180.0 / np.pi

        # Compute uniform scale
        len_in = np.linalg.norm(src_pt1 - src_pt0)
        len_out = np.linalg.norm(dst_pt1 - dst_pt0)
        scale = len_out / len_in if len_in > 0 else 1.0

        # Build rotation matrix about the first source point
        center = tuple(src_pt0)
        M = cv2.getRotationMatrix2D(center, angle, scale)

        # Translate so that src_pt0 maps exactly to dst_pt0
        dx = float(dst_pt0[0] - src_pt0[0])
        dy = float(dst_pt0[1] - src_pt0[1])
        M[0, 2] += dx
        M[1, 2] += dy

    else:
        # Full affine transform (allows skew) from 3+ points
        M, _ = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC)
        if M is None:
            raise ValueError("Could not compute affine transform with given points")

    # Apply the affine/similarity warp
    arr = np.array(image)
    h, w = arr.shape[:2]
    warped = cv2.warpAffine(arr, M, (w, h), flags=cv2.INTER_LINEAR)
    return Image.fromarray(warped)


def get_circle_pattern(size_px: int, dpi: int = 200) -> np.ndarray:
    fig = plt.figure(figsize=(size_px / dpi, size_px / dpi), dpi=dpi)
    fig.patch.set_facecolor("black")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("black")
    ax.set_axis_off()

    ax.scatter(
        [0.5], [0.5],
        marker='o',
        s=size_px ** 2 / 20,
        c='white',
        linewidths=0
    )

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()) # (H, W, 4)
    plt.close(fig)

    mask = (rgba[:, :, 0] > 128).astype(np.float32)
    return mask

def keep_central_circles(circles, img, x_clip, y_clip):
    h, w = img.shape[:2]

    xmin, xmax = x_clip, w - x_clip                             # inner window x-range
    ymin, ymax = y_clip, h - y_clip                             # inner window y-range

    # boolean masks for the inner window
    in_x = (circles[:, 0] >= xmin) & (circles[:, 0] < xmax)
    in_y = (circles[:, 1] >= ymin) & (circles[:, 1] < ymax)
    mask = in_x & in_y

    circles_filt = circles[mask]

    return circles_filt

def crop_image(img, x_clip, y_clip):
    h, w = img.shape[:2]
    xmin, xmax = x_clip, w - x_clip                             # inner window x-range
    ymin, ymax = y_clip, h - y_clip                             # inner window y-range

    # crop image to the same inner window
    img_cropped = img[ymin:ymax, xmin:xmax].copy()

    return img_cropped

def blob_centers(det_mask: np.ndarray,
                 approx_marker_area: int = 363,
                 split_large: bool = True) -> np.ndarray:
    det_uint = det_mask.astype(np.uint8)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        det_uint, connectivity=8)

    centres = []

    for lbl in range(1, n_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        blob = (labels == lbl).astype(np.uint8)

        if split_large and area > 3 * approx_marker_area:
            dist = cv2.distanceTransform(blob, cv2.DIST_L2, 3)

            local_max = (dist == cv2.dilate(dist, None))
            local_max &= dist > 0.4 * dist.max()

            seeds = np.zeros_like(dist, np.int32)
            seeds[local_max] = np.arange(1, np.count_nonzero(local_max) + 1)

            blob_rgb = cv2.merge([blob * 255] * 3)
            cv2.watershed(blob_rgb, seeds)

            for sub_lbl in range(1, seeds.max() + 1):
                ys, xs = np.where(seeds == sub_lbl)
                if xs.size == 0:
                    continue
                idx = np.argmax(dist[ys, xs])
                centres.append([xs[idx], ys[idx]])
            continue

        dist = cv2.distanceTransform(blob, cv2.DIST_L2, 3)
        ys, xs = np.where(blob)
        idx = np.argmax(dist[ys, xs])
        centres.append([xs[idx], ys[idx]])

    return np.asarray(centres, dtype=int)   # shape (N, 2)

def get_circles(pattern, img, pos_thresh, neg_thresh, pixels_to_shrink=3):
    shr_pat = shrink_pattern(pattern, pixels_to_shrink)

    ker = shr_pat
    thr = ker.sum() * pos_thresh

    ker_inv = 1.0 - pattern
    inv_thr  = ker_inv.sum() * neg_thresh

    resp = cv2.filter2D(img, -1, ker)
    resp_inv = cv2.filter2D(img, -1, ker_inv)

    det = (resp < thr) & (resp_inv > inv_thr)
    circle_coords   = blob_centers(det,
                                 approx_marker_area=363,
                                 split_large=False)

    return circle_coords

def shrink_pattern(pat: np.ndarray, pixels: int = 1) -> np.ndarray:
    se = np.ones((3, 3), np.uint8)
    pat_uint8 = (pat * 255).astype(np.uint8)
    pat_eroded = cv2.erode(pat_uint8, se, iterations=pixels)
    return (pat_eroded > 0).astype(pat.dtype)

def main(images, vert_clip_fraction: float, horz_clip_fraction: float, positive_threshold: float, negative_threshold: float, kernel_size:int, output_dir: str):
    circles_ref = np.load('a')
    circle_kernel = get_circle_pattern(kernel_size)
    total_image_shape = images[0][0].shape
    vert_clip = math.floor(total_image_shape[0]*vert_clip_fraction)
    horz_clip = math.floor(total_image_shape[1]*horz_clip_fraction)
    rows = len(images)
    columns = len(images[0])

    logger.debug(f'Clipping images, from {total_image_shape} to {vert_clip}, {horz_clip} (fractions {vert_clip_fraction}, {horz_clip_fraction})')
    pbar = tqdm.tqdm(desc='Clipping Images', total=rows*columns)

    adjusted_clipped_images = np.zeros((rows, columns, total_image_shape[0] - 2 * vert_clip, total_image_shape[1] - 2 * horz_clip, 3), dtype=np.uint8)
    for row_num, row in enumerate(images):
        for col_num, image in enumerate(row):
            circle_coords = get_circles(circle_kernel, image, pos_thresh=positive_threshold, neg_thresh=negative_threshold, pixels_to_shrink=10)
            circle_coords = keep_central_circles(circle_coords, image, x_clip=horz_clip, y_clip=vert_clip)
            circle_coords = circle_coords[np.lexsort((circle_coords[:, 1], circle_coords[:, 0]))]
            aligned_image = align_image_general(image, src_pts=np.array(circle_coords[:3]), dst_pts=np.array(circles_ref[row_num][col_num][:3]))
            clipped_img = crop_image(np.array(aligned_image), horz_clip, vert_clip)
            adjusted_clipped_images[rows - row_num - 1][col_num] = clipped_img
            pbar.update()
    pbar.close()

    if output_dir is not None:
        logger.debug('Saving...')
        np.save(os.path.join(output_dir, 'non-stitch-prepro-out.npy'), adjusted_clipped_images)

    return adjusted_clipped_images