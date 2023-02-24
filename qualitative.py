from examples.load_MB import MBSMPL
from aitviewer.viewer import Viewer
import imgui
import glob
from aitviewer.configuration import CONFIG as C
import os
import numpy as np
from aitviewer.renderables.spheres import Spheres

class QualitativeViewer(Viewer):
    def __init__(self, **kwargs):
        super().__init__(title="Utility Viewer", **kwargs)

    def gui_load(self):
        imgui.set_next_window_position(1250, 100, imgui.FIRST_USE_EVER)
        imgui.set_next_window_size(
            self.window_size[0] * 0.4, self.window_size[1] * 0.5, imgui.FIRST_USE_EVER
        )
        imgui.end()



if __name__ == '__main__':
    v = QualitativeViewer()
    # take globs and add smple meshes to scene
    for pr, model in enumerate(C.models):
        # iterate over datasets
        model_root_trans = np.array([[(pr // 0.9), 0, 0]])
        for d_in, ds in enumerate(C.models[model]['datasets']):
            max_d_instances = C.max_instances
            npz_fls = sorted(glob.glob(os.path.join(C.models[model]['datasets'][ds]['npz'], "*.npz")))
            color = C.models[model]['datasets'][ds]['color']
            mr_fls = sorted(glob.glob(os.path.join(os.path.dirname(os.path.join(C.models[model]['datasets'][ds]['npz'])), "npz_evaluation","*.npz")))
            for i, (nppz_, nppz_m) in enumerate(zip(npz_fls, mr_fls)):
                # add model pose to scene
                instance_root_trans = np.array([[(model_root_trans[0][0]),0,  i // 0.9]])
                pose = MBSMPL.from_mb(
                    npz_data_path=nppz_,
                    z_up=False,
                    color=tuple(color),
                    name=f"{model}_{ds}_{i}",
                    trans = instance_root_trans
                )
                # get trans
                trans = np.load(nppz_)['trans']
                # get markers 
                m_fl = np.load(nppz_m)
                try:
                    markers = m_fl['markers']
                except:
                    markers = m_fl['markers3d']

                markers = Spheres(markers - trans + instance_root_trans, radius=0.014, color = tuple(color), name=f"{model}_{ds}_{i}_markers")

                v.scene.add(pose, markers)
                if (i + 1) >= max_d_instances:
                    break


    v.run()