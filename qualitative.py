from examples.load_MB import MBSMPL
from aitviewer.viewer import Viewer
import imgui
import glob
from aitviewer.configuration import CONFIG as C
import os
import numpy as np
from aitviewer.renderables.spheres import Spheres
from aitviewer.renderables.meshes import Meshes
from scipy.spatial.transform import Rotation as R

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
            # rotation
            r = R.from_euler("z", [180], degrees=True)
            max_d_instances = C.max_instances
            npz_fls = sorted(glob.glob(os.path.join(C.models[model]['datasets'][ds]['root'], "npz", "*.npz")))
            color = C.models[model]['datasets'][ds]['color']
            mr_fls = sorted(glob.glob(os.path.join(C.models[model]['datasets'][ds]['root'], "inferred", "*.npz")))
            for i, (nppz_, nppz_m) in enumerate(zip(npz_fls, mr_fls)):
                # add model pose to scene
                # instance_root_trans = np.array([[(model_root_trans[0][0]), 0,  i // 0.5]]) # 2 meters offset
                instance_root_trans = np.array([[(model_root_trans[0][0]), 0, (d_in * max_d_instances + i) // 0.5]]) # 2 meters offset
                # get trans
                trans = np.load(nppz_)['trans']
                # get global pose
                global_orient = np.load(nppz_)['root_orient']
                new_global_orientation = R.from_matrix(
                    (r.as_matrix() @ R.from_rotvec(global_orient).as_matrix())
                ).as_rotvec()
                pose = MBSMPL.from_mb(
                    npz_data_path=nppz_,
                    z_up=False,
                    color=tuple(color),
                    name=f"{model}_{ds}_{i}",
                    trans = instance_root_trans,
                    rotation = r.as_matrix().squeeze(0),
                )
                faces = pose.smpl_layer.faces
                # get markers 
                m_fl = np.load(nppz_m)
                try:
                    # markers = (r.as_matrix().squeeze() @ (m_fl['markers']).T).T
                    markers = m_fl['markers'] + instance_root_trans - trans
                    markers_transformed = (r.as_matrix().squeeze() @ (markers).T).T
                except:
                    # markers = (r.as_matrix().squeeze() @ m_fl['markers3d'].T).T
                    markers =  m_fl['markers3d'] + instance_root_trans - trans
                    markers_transformed = (r.as_matrix().squeeze() @ (markers).T).T

                markers_transformed = Spheres(markers_transformed, radius=0.014, color = tuple(color), name=f"{model}_{ds}_{i}_markers")

                # GT
                
                gt_mesh = Meshes(
                    (r.as_matrix().squeeze() @ (m_fl['vertices_gt'] + instance_root_trans - trans).T).T,
                    # m_fl['vertices_gt'],
                    faces.cpu().numpy(),
                    # position = instance_root_trans, # - trans,
                    color = (0.42,0.42,0.42,1.0),
                    name=f"gt_{ds}_{i}",
                )

                gt_markers = Spheres((r.as_matrix().squeeze() @ (m_fl['markers3d_gt'] + instance_root_trans - trans).T).T, radius=0.014, color = (0,1,0,0.5), name=f"gt_{ds}_{i}_markers")
                v.scene.add(pose, markers_transformed, gt_mesh, gt_markers)
                if (i + 1) >= max_d_instances:
                    break

    v.run()