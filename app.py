#!/usr/bin/env python
from widgets.ortho_viewer import BreachFinderOrthoViewer
from widgets.controls import BreachFinderControls
from src.widgets.correction_viewer import BreachFinderCorrectionViewer

from pathlib import Path
from argparse import ArgumentParser, Namespace, ArgumentDefaultsHelpFormatter
import os

import napari


from data.constants import FREESURFER_LUT

from chris_plugin import chris_plugin

__version__ = '1.0.0'

DISPLAY_TITLE = r"""
ChRIS T2 Breachfinder
"""


parser = ArgumentParser(
    description="Find breaches in cortical plate segmentation by floodfilling and finding incorrectly exposed inner regions."
    "Speeds up simple manual corrections of cp and sp segmentation in subjects with high complexity and low segmentation quality requirements."
    "Specifically tuned for FNNDSC fetal brain reconstruction pipeline.",
    formatter_class=ArgumentDefaultsHelpFormatter,
)
parser.add_argument('-i', "--input", "--t2", type=str, default="recon_to31_nuc.nii",
                    help='Name of input file')
parser.add_argument('-o', "--output", "--seg", type=str, default="segmentation_to31_final.nii",
                    help='Name of input file')

parser.add_argument('-l', '--labels', nargs=2, type=int, default=[1,42],
                    help='Label value pair of target region.')
parser.add_argument('-ax', '--axis', "--view" , type=str, choices=["axial", "saggital", "coronal", "ax", "sg", "cr"] , default=1,
                    help="Default view axis")
parser.add_argument('-w', '--weakpoints', "--show-weakpoints", action='store_true',
                    help="Detect weakpoints")
parser.add_argument('-V', '--version', action='version',
                    version=f'%(prog)s {__version__}')


# TODO: Add a way to bypass inputdir and outputdir optionally for ease of use when outside of plugin
# The main function of this *ChRIS* plugin is denoted by this ``@chris_plugin`` "decorator."
# Some metadata about the plugin is specified here. There is more metadata specified in setup.py.
#
# documentation: https://fnndsc.github.io/chris_plugin/chris_plugin.html#chris_plugin
@chris_plugin(
    parser=parser,
    title="Breachfinder",
    category="ts",                 # ref. https://chrisstore.co/plugins
    min_memory_limit='100Mi',    # supported units: Mi, Gi
    min_cpu_limit='1000m',       # millicores, e.g. "1000m" = 1 CPU core
    min_gpu_limit=0              # set min_gpu_limit=1 to enable GPU
)
def main(options: Namespace, inputdir: Path, outputdir: Path) -> None:
    """
    *ChRIS* plugins usually have two positional arguments: an **input directory** containing
    input files and an **output directory** where to write output files. Command-line arguments
    are passed to this main method implicitly when ``main()`` is called below without parameters.

    :param options: non-positional arguments parsed by the parser given to @chris_plugin
    :param inputdir: directory containing (read-only) input files
    :param outputdir: directory where to write output files
    """

    print(DISPLAY_TITLE)

    BASE_PATH = inputdir
    OUTPUT_PATH = outputdir if outputdir else BASE_PATH
    
    t2_path = os.path.join(BASE_PATH, "recon_segmentation/", options.input)
    seg_path = os.path.join(OUTPUT_PATH, "recon_segmentation/", options.output)

    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QApplication
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    
    viewer = napari.Viewer(title="Breach Finder")
    
    control_panel = BreachFinderControls(
        axis=options.axis, show_weakpoints=options.weakpoints,
    )
    
    # correction_viewer = BreachFinderCorrectionViewer(
    #     viewer,
    #     t2_path=t2_path,
    #     seg_path=seg_path,
    #     lut_path=FREESURFER_LUT,
    #     controls=control_panel,
    #     label_values=tuple(options.labels),
    #     axis=options.axis,
    #     show_weakpoints=options.weakpoints,
    # )
    
    ortho = BreachFinderOrthoViewer(
        viewer,
        t2_path=t2_path,
        seg_path=seg_path,
        lut_path=FREESURFER_LUT,
        controls=control_panel,
    )
    # freeviewViewer = napari.Viewer(title="Freeview-style Viewer")

    # Viewers replace the main canvas; controls go in the dock
    viewer.window._qt_window.setCentralWidget(ortho)
    viewer.window.add_dock_widget(ortho, name='Breach Finder', area='right')
    
    napari.run()


if __name__ == '__main__':
    main()
