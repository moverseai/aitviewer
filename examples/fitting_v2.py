import glob
import os
from aitviewer.viewer import Viewer
from aitviewer.renderables.spheres import Spheres
import numpy as np
from aitviewer.configuration import CONFIG as C
from examples.load_MB import MBSMPL
from scipy.spatial.transform import Rotation as R
from aitviewer.renderables.meshes import Meshes

if __name__ == "__main__":
    # load pkl file and markers
    root = C.root
    max_d_instances = C.max_instances
    # create viewer
    v = Viewer()
    # find methods to compare
    for m, mth in enumerate(C.fitting):
        fitting_root_trans = np.array([[0, 0, (m // 0.5)]])
        # load first method?
        npz_fls = sorted(
                    glob.glob(os.path.join(root, C.fitting[mth].method, C.fitting[mth].type, "npz", "*.npz"))
                )
        mr_fls = sorted(
            glob.glob(os.path.join(root, "npz", "*.npz"))
        )
        color = C.fitting[mth].color
        for i, (nppz_, nppz_m) in enumerate(zip(npz_fls, mr_fls)):
            r = R.from_euler("z", [180], degrees=True)
            instance_root_trans = np.array([[i // 0.5, 0, (fitting_root_trans[0][2])]]) # 2 meters offset
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
                    name=f"{C.fitting[mth].method}_{i}",
                    trans = instance_root_trans,
                    rotation = r.as_matrix().squeeze(0),
            )
            faces = pose.smpl_layer.faces
            # get markers 
            m_fl = np.load(nppz_m)
            markers =  m_fl['markers3d'] + instance_root_trans - trans
            markers_transformed = (r.as_matrix().squeeze() @ (markers).T).T
            # GT
            gt_mesh = Meshes(
                (r.as_matrix().squeeze() @ (m_fl['vertices_gt'] + instance_root_trans - trans).T).T,
                # m_fl['vertices_gt'],
                faces.cpu().numpy(),
                # position = instance_root_trans, # - trans,
                color = (0.42,0.42,0.42,1.0),
                name=f"gt_{i}",
            )
            gt_markers = Spheres((r.as_matrix().squeeze() @ (m_fl['markers3d_gt'] + instance_root_trans - trans).T).T, radius=C.markers_size, color = (0,1,0,0.5), name=f"gt_{i}_markers")
            markers_transformed = Spheres(markers_transformed, radius=C.markers_size, color = tuple(color), name=f"{C.fitting[mth].method}_{i}_markers")
            v.scene.add(pose, markers_transformed, gt_mesh, gt_markers)
            if (i + 1) >= max_d_instances:
                break
    v.run()
