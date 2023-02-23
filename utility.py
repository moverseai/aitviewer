from examples.load_MB import MBSMPL
from aitviewer.viewer import Viewer
import imgui
import glob
from aitviewer.configuration import CONFIG as C
import os
import numpy as np

class UtilityViewer(Viewer):
    def __init__(self, **kwargs):
        super().__init__(title="Utility Viewer", **kwargs)

    def gui_load(self):
        imgui.set_next_window_position(1250, 100, imgui.FIRST_USE_EVER)
        imgui.set_next_window_size(
            self.window_size[0] * 0.4, self.window_size[1] * 0.5, imgui.FIRST_USE_EVER
        )
        # imgui.push_item_width(imgui.get_window_width() * 0.95)
        # _, self.selected_sequence = imgui.listbox(
        #     "##Sequences", self.selected_sequence, self.displaynames, 15
        # )
        # imgui.pop_item_width()
        imgui.end()



if __name__ == '__main__':
    v = UtilityViewer()
    # take globs and add smple meshes to scene
    for pr, pair in enumerate(C.pairs):
        # get the path to the npz file
        gt_fls = glob.glob(os.path.join(C.pairs[pair], "gt_*.npz"))
        pred_fls = glob.glob(os.path.join(C.pairs[pair], "pred_*.npz"))
        pair_root_trans = np.array([[(pr // 0.9), 0, 0]])
        for i, (gt, pred) in enumerate(zip(gt_fls, pred_fls)):
            # add gt_pose
            # instance root translation
            instance_root_trans = np.array([[(pair_root_trans[0][0]), i // 0.9, 0]])
            gt_pose = MBSMPL.from_mb(
                npz_data_path=gt,
                z_up=True,
                trans = instance_root_trans, # add an offset to the z axis
                name = f"{pair}_gt_{i}",
            )
            v.scene.add(gt_pose)
            # add pred_pose
            pred_pose = MBSMPL.from_mb(
                npz_data_path=pred,
                z_up=True,
                trans = instance_root_trans,
                name = f"{pair}_pred_{i}",
            )
            v.scene.add(pred_pose)
            if (i + 1) >= C.num_pairs:
                break
        
    # v.scene.add(SMPLSequence.t_pose())
    v.run()