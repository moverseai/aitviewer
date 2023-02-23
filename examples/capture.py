import glob
import os
from aitviewer.viewer import Viewer
from aitviewer.renderables.spheres import Spheres
import open3d as o3d
import numpy as np
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.models.smpl import SMPLLayer
from aitviewer.configuration import CONFIG as C
from aitviewer.scene.k4a import K4A
from aitviewer.renderables.billboard import Billboard


def merge_npz(npz_fls, num_betas=16):
    """Merge multiple npz files into one."""
    data = {}
    data["trans"] = []
    data["pose_body"] = []
    data["betas"] = []
    data["num_betas"] = num_betas
    data["root_orient"] = []
    data["pose_jaw"] = []
    data["poses"] = []
    for npz_fl in npz_fls:
        tmp = np.load(npz_fl, allow_pickle=False)
        data["trans"].append(tmp["trans"])
        data["pose_body"].append(tmp["pose_body"])
        data["betas"].append(tmp["betas"])
        data["root_orient"].append(tmp["root_orient"])
        data["pose_jaw"].append(tmp["pose_jaw"])
        data["poses"].append(tmp["poses"])
    # stack all the data
    for key in data.keys():
        if "betas" in key:
            continue
        else:
            data[key] = np.vstack(data[key])

    # take the mean of betas
    # TODO: this should be updated
    data["betas"] = np.median(data["betas"], axis=0)
    data["poses"] = np.hstack(
        [
            data["root_orient"],
            data["pose_body"],
            data["poses"],
        ]
    )
    return data


if __name__ == "__main__":
    # load pkl file and markers
    root = C.root
    npz_fls = sorted(
        glob.glob(os.path.join(root, "fit", C.fitting.method, C.fitting.type, "npz", "*.npz"))
    )
    markers_predicted = sorted(glob.glob(os.path.join(root, "markers", "*.ply")))

    markers_input = sorted(glob.glob(os.path.join(root, "input_plys", "*.ply")))

    markers_raw = sorted(glob.glob(os.path.join(root, "markers_raw", "*.ply")))

    data = merge_npz(npz_fls, C.body.num_betas)

    smpl_layer = SMPLLayer(
        model_type=C.body.type,
        gender=C.body.gender,
        device=C.device,
        num_betas=C.body.num_betas,
    )

    seq_mb = SMPLSequence(
        poses_body=data["pose_body"],
        smpl_layer=smpl_layer,
        poses_root=data["root_orient"],
        betas=data["betas"],
        trans=data["trans"],
    )

    # read point cloud
    markers_seq = []
    markers_input_seq = []
    clustered_markers_rendable = []
    markers_raw_seq = []
    clustered_markers = np.zeros((len(markers_predicted), 100, 3))
    raw_markers = np.zeros((len(markers_predicted), 200, 3))
    for markers, inp, raw in zip(markers_predicted, markers_input, markers_raw):
        markers = o3d.io.read_point_cloud(markers)
        markers_seq.append(np.asarray(markers.points))
        # markers input
        markers = o3d.io.read_point_cloud(inp)
        clustered_markers = np.asarray(markers.points)
        if clustered_markers.shape[0] < 100:
            clustered_markers = np.concatenate(
                (
                    clustered_markers,
                    np.zeros((100 - clustered_markers.shape[0], 3)),
                ),
                axis=0,
            )
        markers_input_seq.append(clustered_markers)
        # markers_raw
        markers = o3d.io.read_point_cloud(raw)
        raw_markers = np.asarray(markers.points)
        if raw_markers.shape[0] < 200:
            raw_markers = np.concatenate(
                (
                    raw_markers,
                    np.zeros((200 - raw_markers.shape[0], 3)),
                ),
                axis=0,
            )
        markers_raw_seq.append(raw_markers)

    markers = Spheres(np.array(markers_seq))
    markers_input = Spheres(np.array(markers_input_seq), color=(0, 1, 0, 1))  # green
    markers_raw = Spheres(np.array(markers_raw_seq), color=(1, 0, 0, 1))  # red

    # add cameras
    width = C.image_width
    height = C.image_height
    cameras = []

    # create viewer
    v = Viewer()
    for cfl in glob.glob(os.path.join(root, "*.npz")):
        camera_info = np.load(cfl)
        radial_dist = camera_info["radial"]
        tangential_dist = camera_info["tangential"]
        dist_vector = np.hstack([radial_dist[:2], tangential_dist, radial_dist[2:]])
        intrinsics = camera_info["intrinsics"].T  # [i]
        rot = camera_info["extrinsics"][:3, :3]
        trans = camera_info["extrinsics"][3, :3]
        extrinsics = np.vstack([rot, trans]).T
        dist_coeffs = dist_vector
        camera = K4A(
            intrinsics,
            extrinsics,
            width,
            height,
            dist_coeffs=dist_coeffs,
            viewer=v,
            is_selectable=False,
        )
        cameras.append(camera)
        v.scene.add(camera)
        # create a billobard
        camera_index = cfl.split("_")[-2]
        images_path = os.path.join(root, "cameras", camera_index)
        if not os.path.exists(images_path):
            print(f"no images for camera {camera_index}")
        else:
            billboard = Billboard.from_camera_and_distance(
                camera,
                0.0,
                width,
                height,
                [os.path.join(images_path, f) for f in sorted(os.listdir(images_path))],
            )
            v.scene.add(billboard)

    v.scene.add(seq_mb, markers, markers_input, markers_raw)
    v.run()
