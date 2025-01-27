# [![AITV](https://raw.githubusercontent.com/eth-ait/aitviewer/main/aitv_logo.svg)](https://github.com/eth-ait/aitviewer) AITViewer

A set of tools to visualize and interact with sequences of 3D data with cross-platform support on Windows, Linux, and Mac OS X.

![AITV Sample](https://raw.githubusercontent.com/eth-ait/aitviewer/main/aitv_sample.png)


## Features
* Easy to use Python interface.
* Load [SMPL[-H | -X]](https://smpl.is.tue.mpg.de/) / [MANO](https://mano.is.tue.mpg.de/) / [FLAME](https://flame.is.tue.mpg.de/) sequences and display them in an interactive viewer.
* Support for the [STAR model](https://github.com/ahmedosman/STAR).
* Manually editable SMPL sequences.
* Render 3D data on top of images via weak-perspective or OpenCV camera models.
* Built-in extensible GUI (based on Dear ImGui).
* Export the scene to a video (mp4/gif) via the GUI or render videos/images in headless mode.
* Animatable camera paths.
* Prebuilt renderable primitives (cylinders, spheres, point clouds, etc).
* Support live data feeds and rendering (e.g., webcam).
* Modern OpenGL shader-based rendering pipeline for high performance (via ModernGL / ModernGL Window).

https://user-images.githubusercontent.com/5639197/188625409-1c86b12e-4e91-48ba-ab7e-b050508d586b.mp4

![AITV SMPL Editing](https://user-images.githubusercontent.com/5639197/188625764-351100e9-992e-430c-b170-69d4f142f5dd.gif)

## Installation
Basic Installation:
```commandline
pip install aitviewer
```

Or install locally (if you need to extend or modify code)
```commandline
git clone git@github.com:eth-ait/aitviewer.git
cd aitviewer
pip install -e .
```

Note that this does not install the GPU-version of PyTorch automatically. If your environment already contains it, you should be good to go, otherwise install it manually.

If you would like to visualize STAR, please install the package manually via
```commandline
pip install git+https://github.com/ahmedosman/STAR.git
```
and download the respective body models from the official website.

## Configuration
The viewer loads default configuration parameters from [`aitvconfig.yaml`](aitviewer/aitvconfig.yaml). There are three ways how to override these parameters:
  - Create a file named `aitvconfig.yaml` and have the environment variable `AITVRC` point to it. Alternatively, you can point `AITVRC` to the directory containing `aitvconfig.yaml`.
  - Create a file named `aitvconfig.yaml` in your current working directory, i.e. from where you launch your python program.
  - Pass a `config` parameter to the `Viewer` constructor.

Note that the configuration files are loaded in this order, i.e. the config file in your working directory overrides all previous parameters.

The configuration management is using [OmegaConf](https://omegaconf.readthedocs.io/en/2.1_branch/index.html). You will probably want to override the following parameters at your convenience:
- `datasets.amass`: where [AMASS](https://amass.is.tue.mpg.de/) is stored if you want to load AMASS sequences.
- `smplx_models`: where SMPLX models are stored, preprocessed as required by the [`smplx` package](https://github.com/vchoutas/smplx).
- `star_models`: where the [STAR model](https://github.com/ahmedosman/STAR) is stored if you want to use it.
- `export_dir`: where videos and other outputs are stored by default.


## Quickstart
Display the SMPL T-pose:
```py
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer

if __name__ == '__main__':
    v = Viewer()
    v.scene.add(SMPLSequence.t_pose())
    v.run()
```

## Examples

Check out the [examples](examples/) for a few examples how to use the viewer:
 * [`animation.py`](examples/animation.py): Example of how 3D primitives can be animated.

 * [`camera_path.py`](examples/camera_path.py): Example how to use camera paths.

 * [`headless_rendering.py`](examples/headless_rendering.py): Example how to render a video in headless mode.

 * [`load_3DPW.py`](examples/load_3DPW.py): Loads an SMPL sequence from the 3DPW dataset and displays it in the viewer.

 * [`load_AMASS.py`](examples/load_AMASS.py): Loads an SMPL sequence from the AMASS dataset and displays it in the viewer.

 * [`load_DIP.py`](examples/load_DIP.py): Loads an SMPL and IMU sequence taken from the TotalCapture dataset as used by [DIP](https://github.com/eth-ait/dip18).

 * [`load_GLAMR.py`](examples/load_GLAMR.py): Loads a result obtained from [GLAMR](https://github.com/NVlabs/GLAMR) and displays it in the viewer both for 3D and 2D inspection.

 * [`load_obj.py`](examples/load_obj.py): Loads meshes from OBJ files.

 * [`load_ROMP.py`](examples/load_ROMP.py): Loads the result of [ROMP](https://github.com/Arthur151/ROMP) and overlays it on top of the input image using the OpenCV camera model.

 * [`load_template.py`](examples/load_template.py): Loads the template meshes of SMPL-H, MANO, and FLAME.

 * [`load_VIBE.py`](examples/load_VIBE.py): Loads the result of [VIBE](https://github.com/mkocabas/VIBE) and overlays it on top of the input image.

 * [`missing_frames.py`](examples/missing_frames.py): Example how sequences with intermittent missing frames can be visualized.

 * [`quickstart.py`](examples/quickstart.py): The above quickstart example.

 * [`render_primitives.py`](examples/render_primitives.py): Renders a bunch of spheres and lines.

 * [`stream.py`](examples/stream.py): Streams your webcam into the viewer.

 * [`vertex_clicking.py`](examples/vertex_clicking.py): An example how to subclass the basic Viewer class for custom interaction.

## Keyboard shortcuts

The viewer supports the following keyboard shortcuts, all of this functionality is also accessible from the menus and windows in the GUI.
This list can be shown directly in the viewer by clicking on the `Help -> Keyboard shortcuts` menu.

- `SPACE` Start/stop playing animation.
- `.` Go to next frame.
- `,` Go to previous frame.
- `X` Center view on the selected object.
- `O` Enable/disable orthographic camera.
- `T` Show the camera target in the scene.
- `C` Save the camera position and orientation to disk.
- `L` Load the camera position and orientation from disk.
- `K` Lock the selection to the currently selected object.
- `S` Show/hide shadows.
- `D` Enabled/disable dark mode.
- `P` Save a screenshot to the the `export/screenshots` directory.
- `I` Change the viewer mode to `inspect`.
- `V` Change the viewer mode to `view`.
- `E` If a mesh is selected, show the edges of the mesh.
- `F` If a mesh is selected, switch between flat and smooth shading.
- `Z` Show a debug visualization of the object IDs.
- `ESC` Exit the viewer.

## Projects using the AITViewer
The following projects have used the AITViewer:
- Dong et al., [Shape-aware Multi-Person Pose Estimation from Multi-view Images](https://ait.ethz.ch/projects/2021/multi-human-pose/), ICCV 2021
- Kaufmann et al., [EM-POSE: 3D Human Pose Estimation from Sparse Electromagnetic Trackers](https://ait.ethz.ch/projects/2021/em-pose/), ICCV 2021
- Vechev et al., [Computational Design of Kinesthetic Garments](https://ait.ethz.ch/projects/2022/cdkg/), Eurographics 2021
- Guo et al., [Human Performance Capture from Monocular Video in the Wild](https://ait.ethz.ch/projects/2021/human-performance-capture/index.php), 3DV 2021
- Dong and Guo et al., [PINA: Learning a Personalized Implicit Neural Avatar from a Single RGB-D Video Sequence](https://zj-dong.github.io/pina/), CVPR 2022

## Citation
If you use this software, please cite it as below.
```commandline
@software{Kaufmann_Vechev_AITViewer_2022,
  author = {Kaufmann, Manuel and Vechev, Velko and Mylonopoulos, Dario},
  doi = {10.5281/zenodo.1234},
  month = {7},
  title = {{AITViewer}},
  url = {https://github.com/eth-ait/aitviewer},
  year = {2022}
}
```

## Contact & Contributions
This software was developed by [Manuel Kaufmann](mailto:manuel.kaufmann@inf.ethz.ch), [Velko Vechev](mailto:velko.vechev@inf.ethz.ch) and Dario Mylonopoulos.
For questions please create an issue.
We welcome and encourage module and feature contributions from the community.
