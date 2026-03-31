#!/usr/bin/env python
from utils import add_central_viewer
from chris_plugin import chris_plugin

from widgets.controls import BreachFinderControls

from pathlib import Path
from argparse import ArgumentParser, Namespace, ArgumentDefaultsHelpFormatter
import os

import napari

from data.constants import FREESURFER_LUT


# Required for the ortho viewer
from widgets.ortho_viewer import BreachFinderOrthoViewer
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication
QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)

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
                    help='Path of T2 image.')
parser.add_argument('-o', "--output", "-s", "--seg", type=str, default="segmentation_to31_final.nii",
                    help='Path of output segmentation image.')

parser.add_argument('-l', '--labels', nargs=2, type=int, default=[1,42],
                    help='Label value pair of target region.')
parser.add_argument('-ax', '--axis', "--view" , type=str, choices=["axial", "saggital", "coronal", "ax", "sg", "cr"] , default=1,
                    help="Default view axis")
parser.add_argument('-w', '--weakpoints', "--show-weakpoints", action='store_true',
                    help="Detect weakpoints")
parser.add_argument('-V', '--version', action='version',
                    version=f'%(prog)s {__version__}')



# TODO: Add a way to bypass inputdir and outputdir optionally for ease of use when outside of plugin
def main(options: Namespace, inputdir: Path | str = os.getcwd(), outputdir: Path | str = os.getcwd()) -> None:
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
    #     axis=options.axis,gu
    #     show_weakpoints=options.weakpoints,
    # )
    
    ortho = BreachFinderOrthoViewer(
        viewer,
        t2_path=options.input if options.input else t2_path,
        seg_path=options.output if options.output else seg_path,
        lut_path=FREESURFER_LUT,
        controls=control_panel,
        show_cross=False
    )
    
    add_central_viewer(viewer, ortho)    
    viewer.window.add_dock_widget(control_panel, name='Breach Finder', area='right')
    
    napari.run()

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
def chris_main(options: Namespace, inputdir: Path, outputdir: Path):
    main(options, inputdir, outputdir)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args, )
