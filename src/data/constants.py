import numpy as np

#  label constants 
LEFT_CP = 1
RIGHT_CP = 42
LEFT_INNER = 161
RIGHT_INNER = 160

FREESURFER_LUT = "src/data/FreeSurferColorLUT.txt"

AXIS_NAMES = {0: "Sagittal", 1: "Coronal", 2: "Axial"}

DIMS_ORDER = {
    0: (0, 1, 2),
    1: (1, 0, 2),
    2: (2, 0, 1),
}

_ROT_PLANES = {0: (1, 2), 1: (0, 2), 2: (0, 1)}


def make_breach_colormap():
    from napari.utils.colormaps import DirectLabelColormap

    return DirectLabelColormap(
        color_dict={
            0: np.array([0, 0, 0, 0], dtype=np.float32),
            1: np.array([1, 0, 0, 0.85], dtype=np.float32),   # breach = red
            2: np.array([0, 1, 0, 0.6], dtype=np.float32),    # cp = green
            3: np.array([1, 1, 0, 0.85], dtype=np.float32),   # weakpoint = yellow
        }
    )
