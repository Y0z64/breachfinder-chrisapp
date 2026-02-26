#!/usr/bin/env python
import numpy as np
import cv2
from scipy import ndimage
from src.data.constants import LEFT_CP, RIGHT_CP, LEFT_INNER, RIGHT_INNER


def extract_slice(volume, slice_idx, axis):
    if axis == 0:
        return volume[slice_idx, :, :]
    elif axis == 1:
        return volume[:, slice_idx, :]
    else:
        return volume[:, :, slice_idx]


def _flood_from_borders(fillable):
    h, w = fillable.shape
    filled = np.ascontiguousarray(fillable.copy())
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    for y in range(h):
        if fillable[y, 0] == 1:
            cv2.floodFill(filled, mask, (0, y), 2)
        if fillable[y, w - 1] == 1:
            cv2.floodFill(filled, mask, (w - 1, y), 2)
    for x in range(w):
        if fillable[0, x] == 1:
            cv2.floodFill(filled, mask, (x, 0), 2)
        if fillable[h - 1, x] == 1:
            cv2.floodFill(filled, mask, (x, h - 1), 2)

    return filled == 2


def _detect_breaches_for_hemisphere(cp, inner, contra_inner):
    if cp.sum() == 0 or inner.sum() == 0:
        return np.zeros_like(cp), False

    fillable = (1 - cp).astype(np.uint8)
    outside_reachable = _flood_from_borders(fillable)

    # Exclude voxels where the contralateral inner region lives —
    # that is inter-region contact, not a true breach.
    outside_reachable &= ~contra_inner.astype(bool)

    labeled_inner, n_regions = ndimage.label(inner)
    breach_mask = np.zeros_like(cp, dtype=np.uint8)
    has_breach = False

    for region_id in range(1, n_regions + 1):
        region = labeled_inner == region_id
        if (outside_reachable & region).any():
            dilated = ndimage.binary_dilation(region, iterations=1)
            boundary = dilated & outside_reachable & ~region
            breach_mask |= boundary.astype(np.uint8)
            has_breach = True

    return breach_mask, has_breach


def _cluster_and_locate(cluster_mask, grouping_factor=3):

    merged = ndimage.binary_dilation(cluster_mask, iterations=grouping_factor)
    labeled, count = ndimage.label(merged)

    centroids = []
    radii = []
    for i in range(1, count + 1):
        region = labeled == i
        area = (region & cluster_mask.astype(bool)).sum()
        centroids.append(ndimage.center_of_mass(region))
        radii.append(max(5, np.sqrt(area) * 2))

    return {"count": count, "centroids": centroids, "radii": radii}


def detect_breaches(
    seg_slice, label_values=(LEFT_CP, RIGHT_CP), detect_weakpoints=True
):
    left_cp = (seg_slice == LEFT_CP).astype(np.uint8)
    right_cp = (seg_slice == RIGHT_CP).astype(np.uint8)
    left_inner = (seg_slice == LEFT_INNER).astype(np.uint8)
    right_inner = (seg_slice == RIGHT_INNER).astype(np.uint8)

    if left_cp.sum() == 0 and right_cp.sum() == 0:
        return None

    breach_mask = np.zeros_like(left_cp)
    weakpoint_mask = np.zeros_like(left_cp)
    has_breach = False
    has_weakpoint = False

    # weakpoint detection in eroded masks
    left_cp_eroded = ndimage.binary_erosion(left_cp, iterations=1)
    right_cp_eroded = ndimage.binary_erosion(right_cp, iterations=1)

    # Right hemisphere
    bm, found = _detect_breaches_for_hemisphere(right_cp, right_inner, left_inner)
    wm, wfound = _detect_breaches_for_hemisphere(
        left_cp_eroded, right_inner, left_inner
    )
    if found:
        breach_mask |= bm
        has_breach = True

    if wfound:
        weakpoint_mask |= wm
        has_weakpoint = True

    # Left hemisphere
    bm, found = _detect_breaches_for_hemisphere(left_cp, left_inner, right_inner)
    wm, wfound = _detect_breaches_for_hemisphere(
        right_cp_eroded, left_inner, right_inner
    )
    if found:
        breach_mask |= bm
        has_breach = True

    if wfound:
        weakpoint_mask |= wm
        has_weakpoint = True

    if not has_breach and (not has_weakpoint and detect_weakpoints):
        return None

    cp_mask = np.isin(seg_slice, list(label_values)).astype(np.uint8)

    # Cluster the breach voxels for annotation
    holes = _cluster_and_locate(breach_mask)
    weakpoints = _cluster_and_locate(weakpoint_mask)

    return {
        "breach_mask": breach_mask,
        "cp_mask": cp_mask,
        "holes": holes,
        "weakpoints": weakpoints,
    }


# TODO: Make fix width parametrized
def propose_fix(seg_slice, breach_mask):
    fixed = seg_slice.copy()

    left_inner = seg_slice == LEFT_INNER
    right_inner = seg_slice == RIGHT_INNER

    labeled_breaches, n = ndimage.label(breach_mask)

    left_dilated = ndimage.binary_dilation(left_inner, iterations=1)
    right_dilated = ndimage.binary_dilation(right_inner, iterations=1)

    for bid in range(1, n + 1):
        region = labeled_breaches == bid
        left_contact = (region & left_dilated).sum()
        right_contact = (region & right_dilated).sum()

        if left_contact >= right_contact:
            fixed[region] = LEFT_CP
        else:
            fixed[region] = RIGHT_CP

    return fixed


def scan_volume(seg_data, axis, label_values=(LEFT_CP, RIGHT_CP)):
    n_slices = seg_data.shape[axis]
    for i in range(n_slices):
        seg_slice = extract_slice(seg_data, i, axis)
        result = detect_breaches(seg_slice, label_values)
        if result is not None:
            yield i, result
