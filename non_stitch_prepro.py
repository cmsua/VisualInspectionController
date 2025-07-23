import math
import logging
import os

import numpy as np
import cv2
import tqdm
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from PIL import Image

import pickle as pkl

logger = logging.getLogger('main')

def pil_to_gray_array(image: Image.Image) -> np.ndarray:
    """
    Convert a PIL image to a 2D grayscale numpy array.
    """
    arr = np.array(image.convert('L'), dtype=np.float32)
    return arr

def align_image_general(
    image: Image.Image,
    src_pts: np.ndarray,
    dst_pts: np.ndarray
) -> Image.Image:
    """
    Align an image by computing either a similarity (2-point) transform.

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

    M, _ = cv2.estimateAffinePartial2D(src, dst, None, cv2.RANSAC,
                                       ransacReprojThreshold=2.0,
                                       maxIters=5000, confidence=0.999)
    if M is None:
        raise ValueError("Could not compute similarity transform")

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
                 approx_marker_area: int = 340,
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
        centres.append([int(xs.mean()), int(ys.mean())])

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
                                   approx_marker_area=340,
                                   split_large=False)

    return circle_coords

def shrink_pattern(pat: np.ndarray, pixels: int = 1) -> np.ndarray:
    se = np.ones((3, 3), np.uint8)
    pat_uint8 = (pat * 255).astype(np.uint8)
    pat_eroded = cv2.erode(pat_uint8, se, iterations=pixels)
    return (pat_eroded > 0).astype(pat.dtype)

def bidirectional_match(a: np.ndarray, b: np.ndarray, radius: float = 200):
    tree_a, tree_b = cKDTree(a), cKDTree(b)

    # a -> b
    dist_ab, idx_ab = tree_b.query(a, distance_upper_bound=radius)
    # b -> a
    dist_ba, idx_ba = tree_a.query(b, distance_upper_bound=radius)

    keep_ref, keep_new = [], []
    used_b = set()

    for i, (d, j) in enumerate(zip(dist_ab, idx_ab)):
        if d <= radius and j < len(b):           # ai found a neighbour
            # require mutual nearest and that bj not reused
            if idx_ba[j] == i and dist_ba[j] <= radius and j not in used_b:
                keep_ref.append(a[i])
                keep_new.append(b[j])
                used_b.add(j)

    if keep_ref:   # convert to arrays; else return empty (0, 2)
        return np.vstack(keep_ref), np.vstack(keep_new)
    return np.empty((0, 2), a.dtype), np.empty((0, 2), b.dtype)

def main(images, vert_clip_fraction: float, horz_clip_fraction: float, kernel_size:int, output_dir: str, is_baseline: bool = False):
    circle_kernel = get_circle_pattern(kernel_size)
    total_image_shape = images[0][0].shape
    vert_clip = math.floor(total_image_shape[0]*vert_clip_fraction)
    horz_clip = math.floor(total_image_shape[1]*horz_clip_fraction)
    rows = len(images)
    columns = len(images[0])
    print(f"num cols: {columns}")
    positive_thresholds = np.linspace(110, 145, columns)
    positive_thresholds[-1] = 165
    negative_thresholds = np.linspace(150, 180, columns)
    positive_thresholds[-1] = 195
    skip_set = {(0,0),(0,1),(0,7),(0,8),(1,0),(1,8),(2,0),(2,8),(3,0),(3,8),
                (7,0),(7,8),(8,0),(8,8),(9,0),(9,8),(10,0),(10,1),(10,7),(10,8),
                (11,0),(11,1),(11,7),(11,8)}
    
    if not is_baseline:
        with open('./circles_ref.pkl', 'rb') as f:
            circles_ref = pkl.load(f)
        print(f"Circles ref length: {len(circles_ref)}")
    else:
        circles_ref = []

    logger.debug(f'Clipping images, from {total_image_shape} to {vert_clip}, {horz_clip} (fractions {vert_clip_fraction}, {horz_clip_fraction})')
    pbar = tqdm.tqdm(desc='Clipping Images', total=rows*columns)


    # try:
    adjusted_clipped_images = np.zeros((rows, columns, total_image_shape[0] - 2 * vert_clip, total_image_shape[1] - 2 * horz_clip, 3), dtype=np.uint8)
    for row_num, row in enumerate(images):
        for col_num, image in enumerate(row):
            image = Image.fromarray(images[row_num, col_num].astype(np.uint8))
            if (row_num, col_num) in skip_set:
                clipped_img = crop_image(np.array(image), horz_clip, vert_clip)
                print(f"image [{row_num}, {col_num}] skipped")
                if is_baseline:
                    circles_ref.append(np.array([[0,0]]))
            else:
                bw_image = pil_to_gray_array(image)
                circle_coords = get_circles(circle_kernel, bw_image, pos_thresh=positive_thresholds[col_num], neg_thresh=negative_thresholds[col_num], pixels_to_shrink=10)
                circle_coords = keep_central_circles(circle_coords, bw_image, x_clip=150, y_clip=150)
                circle_coords = circle_coords[np.lexsort((circle_coords[:, 1], circle_coords[:, 0]))]
                if is_baseline:
                    clipped_img = crop_image(np.array(image), horz_clip, vert_clip)
                    circles_ref.append(circle_coords)
                else:
                    c_coords, c_ref = bidirectional_match(np.array(circle_coords), np.array(circles_ref[row_num*columns+col_num]))
                    aligned_image = align_image_general(image, src_pts=c_coords, dst_pts=c_ref)
                    clipped_img = crop_image(np.array(aligned_image), horz_clip, vert_clip)
            adjusted_clipped_images[rows - row_num - 1][col_num] = clipped_img
            pbar.update()
    # except:
    #     print(f"Failed for {row_num}, {col_num}")
    pbar.close()


    if is_baseline:
        np.save(os.path.join('./','ref_image_array.npy'), adjusted_clipped_images)
        with open('circles_ref.pkl', 'wb') as file:
            pkl.dump(circles_ref, file)
    elif output_dir is not None:
        print("output dir exists")
        logger.debug('Saving...')
        np.save(os.path.join(output_dir, 'non-stitch-prepro-out.npy'), adjusted_clipped_images)
        print("image saved to output dir")

    return adjusted_clipped_images