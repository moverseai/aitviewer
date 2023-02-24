import numpy as np
from aitviewer.configuration import CONFIG as C
from aitviewer.viewer import Viewer
import glob
import os
import imgui
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.models.smpl import SMPLLayer
import toolz

class MBSMPL(SMPLSequence):
    def __init__(
        self,
        poses_body,
        smpl_layer,
        poses_root=None,
        betas=None,
        trans=None,
        poses_left_hand=None,
        poses_right_hand=None,
        device=C.device,
        dtype=C.f_precision,
        include_root=True,
        normalize_root=False,
        is_rigged=True,
        show_joint_angles=False,
        z_up=False,
        post_fk_func=None,
        icon="\u0093",
        **kwargs,
    ):
        super().__init__(
            poses_body,
            smpl_layer,
            poses_root,
            betas,
            trans,
            poses_left_hand,
            poses_right_hand,
            device,
            dtype,
            include_root,
            normalize_root,
            is_rigged,
            show_joint_angles,
            z_up,
            post_fk_func,
            icon,
            **kwargs,
        )

    @classmethod
    def from_mb(cls, npz_data_path, smpl_layer=None, log=True, z_up=True, **kwargs):
        """Load a pose from a npz file."""
        body_data = np.load(npz_data_path)
        if smpl_layer is None:
            smpl_layer = SMPLLayer(
                model_type=C.body.type,
                # gender=body_data["gender"].item(),
                device=C.device,
                num_betas=C.body.num_betas,
            )
        if log:
            print("Data keys available: {}".format(list(body_data.keys())))

        i_root_end = 3
        i_body_end = i_root_end + smpl_layer.bm.NUM_BODY_JOINTS * 3
        i_left_hand_end = i_body_end + smpl_layer.bm.NUM_HAND_JOINTS * 3
        i_right_hand_end = i_left_hand_end + smpl_layer.bm.NUM_HAND_JOINTS * 3
        trans = body_data['trans'] if not 'trans' in kwargs.keys() \
            else kwargs['trans']
        color = tuple(body_data["color"]) if 'color' in body_data.keys() \
            else  kwargs['color'] if 'color' in kwargs.keys() else \
                (0.5,0.5,0.5,1.0) # set a random color

        return cls(
            poses_body=body_data["poses"][:, i_root_end:i_body_end],
            poses_root=body_data["poses"][:, :i_root_end],
            poses_left_hand=body_data["poses"][:, i_body_end:i_left_hand_end],
            poses_right_hand=body_data["poses"][:, i_left_hand_end:i_right_hand_end],
            smpl_layer=smpl_layer,
            betas=body_data["betas"],
            # trans=body_data["trans"],
            trans = trans,
            z_up=z_up,
            color=color,
            **toolz.dissoc(kwargs,'trans','color')
            # **kwargs,
        )


class MBViewer(Viewer):
    def __init__(self, **kwargs):
        super().__init__(title="MB Viewer", **kwargs)
        self.gui_controls["load"] = self.gui_load
        self.filenames = glob.glob(
            os.path.join(C.datasets.amass, C.export.part, "**", "*.npz")
        )
        self.displaynames = list(
            map(
                lambda f: f"{os.path.basename(os.path.dirname(f))} - {os.path.basename(f)}",
                self.filenames,
            )
        )
        self.selected_pose = 0

    def gui_load(self):
        imgui.set_next_window_position(1250, 100, imgui.FIRST_USE_EVER)
        imgui.set_next_window_size(
            self.window_size[0] * 0.4, self.window_size[1] * 0.5, imgui.FIRST_USE_EVER
        )
        imgui.begin(C.export.part)
        imgui.push_item_width(imgui.get_window_width() * 0.95)
        _, self.selected_pose = imgui.listbox(
            "##Poses", self.selected_pose, self.displaynames, 15
        )
        imgui.pop_item_width()
        if imgui.button("Load Selected", width=100, height=50):
            filename = self.filenames[self.selected_pose]
            displayname = self.displaynames[self.selected_pose]

            mb_pose = MBSMPL.from_mb(
                npz_data_path=filename,
                z_up=False,
                # color=(239 / 255, 42 / 255, 239 / 255, 0.5),
            )

            self.scene.add(mb_pose)
        # imgui.push_item_width(imgui.get_window_width() * 0.95)
        # _, self.selected_sequence = imgui.listbox(
        #     "##Sequences", self.selected_sequence, self.displaynames, 15
        # )
        # imgui.pop_item_width()
        imgui.end()


if __name__ == "__main__":
    v = MBViewer()
    # v.run_animations = True
    # v.scene.camera.position = np.array([10.0, 2.5, 0.0])
    v.run()
