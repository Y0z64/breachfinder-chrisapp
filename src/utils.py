from data.constants import _ROT_PLANES
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
  from napari.utils import DirectLabelColormap

def load_freesurfer_colormap(lut_path) -> 'DirectLabelColormap':
    from napari.utils.colormaps import DirectLabelColormap

    color_dict = {0: np.array([0, 0, 0, 0], dtype=np.float32)}
    with open(lut_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 5:
                try:
                    idx = int(parts[0])
                    color_dict[idx] = np.array(
                        [
                            int(parts[2]) / 255,
                            int(parts[3]) / 255,
                            int(parts[4]) / 255,
                            1.0,
                        ],
                        dtype=np.float32,
                    )
                except (ValueError, IndexError):
                    continue
    return DirectLabelColormap(color_dict=color_dict)


def orient_for_display(data, axis):
    """Rotate a 3-D volume 90° CCW in the display plane of *axis*."""
    return np.rot90(data, k=1, axes=_ROT_PLANES[axis])


def orient_for_save(data, axis):
    """Undo ``orient_for_display`` so we can write back to NIfTI."""
    return np.rot90(data, k=-1, axes=_ROT_PLANES[axis])


def ellipse_bbox_3d(center_rc, radius, slice_idx, axis):
    """4×3 bounding box for a napari 3-D ellipse in *display* volume coords."""
    r, c = center_rc
    if axis == 0:
        vol = [float(slice_idx), r, c]
        rd, cd = 1, 2
    elif axis == 1:
        vol = [r, float(slice_idx), c]
        rd, cd = 0, 2
    else:
        vol = [r, c, float(slice_idx)]
        rd, cd = 0, 1
    corners = []
    for dr, dc in [(-1, -1), (-1, 1), (1, 1), (1, -1)]:
        pt = list(vol)
        pt[rd] += dr * radius
        pt[cd] += dc * radius
        corners.append(pt)
    return np.array(corners, dtype=float)
