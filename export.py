"""
Copyright (C) 2022  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import numpy as np
import os
import imgui
import torch
import toolz
import glob

from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer
from aitviewer.configuration import CONFIG as C
from aitviewer.utils import to_numpy

class ExportingViewer(Viewer):    
    def __init__(self, **kwargs):
        super().__init__(title='AMASS Exporter', **kwargs)
        self.exported = SMPLSequence.empty(gender=C.body.gender, name='Exported')
        self.exported.position = np.array([0.0, 0.0, 1.5])
        self.exported.skeleton_seq.enabled = False
        self.exported.rbs.enabled = False
        self.exported.genders = ['C.body.gender']
        self.exported.names = ['Empty T-pose']
        self.scene.add(self.exported)
        self.gui_controls['export']= self.gui_export
        self.gui_controls['load']= self.gui_load
        self.part, self.subject, self.action = 'Testset', 'One', 'Hard'
        self.filenames = glob.glob(os.path.join(C.datasets.amass, C.export.part, '**', '*_stageii.npz'))
        self.displaynames = list(map(lambda f: f"{os.path.basename(os.path.dirname(f))} - {os.path.basename(f)}", self.filenames))
        self.loaded_sequences = []
        self.selected_sequence = 0

    def gui_load(self):
        imgui.set_next_window_position(1250, 100, imgui.FIRST_USE_EVER)
        imgui.set_next_window_size(self.window_size[0] * 0.4, self.window_size[1] * 0.5, imgui.FIRST_USE_EVER)
        imgui.begin(C.export.part)
        imgui.push_item_width(imgui.get_window_width() * 0.95)
        _, self.selected_sequence = imgui.listbox(
            "##Sequences", self.selected_sequence, self.displaynames, 15
        )
        imgui.pop_item_width()
        if imgui.button('Load Selected', width=100, height=50):
            filename = self.filenames[self.selected_sequence]
            displayname = self.displaynames[self.selected_sequence]
            sequence = SMPLSequence.from_amass(npz_data_path=filename,                
                fps_out=60.0, color=(239/255, 42/255, 239/255, 0.5),
                name=displayname, show_joint_angles=True
            )
            self.scene.add(sequence)
            self.loaded_sequences.append(sequence.name)
            self.subject = f"{os.path.basename(os.path.dirname(filename))}_{os.path.splitext(os.path.basename(filename))[0]}"
        # imgui.same_line()
        # if imgui.button('Remove Selected', width=100, height=50):
        #     sequence = toolz.get(0, list(filter(
        #         lambda x: x.name == self.displaynames[self.selected_sequence], 
        #         self.scene.nodes
        #     )), None)
        #     self.loaded_sequences.remove(sequence.name)
        #     self.scene.remove(sequence)            
        imgui.end()

    def gui_export(self): 
        imgui.set_next_window_position(650, 50, imgui.FIRST_USE_EVER)
        imgui.set_next_window_size(self.window_size[0] * 0.2, self.window_size[1] * 0.3, imgui.FIRST_USE_EVER)
        expanded, _ = imgui.begin("Export", None)
        if expanded:
            imgui.push_style_color(imgui.COLOR_TEXT, 0.2, 0.75, 0.1)
            imgui.core.push_text_wrap_pos(300.)
            imgui.text(f"NOTE: exported frames are serially starting from the start (0). Meaning that to inspect the selected frames the playback needs to rewind. When paused ('space'), frame-by-frame playback is achieved through keys `,` and `.`.")
            imgui.core.pop_text_wrap_pos()
            imgui.pop_style_color(1)
            imgui.separator()
            imgui.begin_group()
            imgui.text(f"Exporting data to: ")
            imgui.same_line(spacing=1)
            imgui.push_style_color(imgui.COLOR_TEXT, 1.0, 0.0, 0.0)
            imgui.text(f"'{os.path.abspath(C.export_dir)}'")
            imgui.pop_style_color(1)
            imgui.push_item_width(imgui.get_window_width() * 0.2)
            imgui.label_text("##PartLabel", "Part:")
            imgui.pop_item_width()
            imgui.same_line(spacing=1)
            changed, self.part = imgui.input_text(
                '##Part', self.part, 256
            )
            imgui.push_item_width(imgui.get_window_width() * 0.2)
            imgui.label_text("##SubjectLabel", "Subject:")
            imgui.pop_item_width()
            imgui.same_line(spacing=1)
            changed, self.subject = imgui.input_text(
                '##Subject', self.subject, 128
            )
            imgui.push_item_width(imgui.get_window_width() * 0.2)
            imgui.label_text("##ActionLabel", "Action:")
            imgui.pop_item_width()
            imgui.same_line(spacing=1)
            changed, self.action = imgui.input_text(
                '##Action', self.action, 512,
            )            
            imgui.end_group()
            imgui.separator()
            imgui.begin_group()
            mode_clicked = imgui.button(f" Add Selected", width=125, height=50)
            if mode_clicked:
                n = toolz.get(0, list(filter(
                    lambda x: self.scene.is_selected(x) and isinstance(x, SMPLSequence) and x is not self.exported, 
                    self.scene.nodes
                )), None)
                if n is not None:
                    print(f"Adding frame #{n.current_frame_id} from {n.name}.")                    
                    self.exported.poses_body = torch.cat([self.exported.poses_body, n.poses_body[n.current_frame_id][np.newaxis, ...]], dim=0)
                    self.exported.poses_root = torch.cat([self.exported.poses_root, n.poses_root[n.current_frame_id][np.newaxis, ...]], dim=0)
                    self.exported.poses_left_hand = torch.cat([self.exported.poses_left_hand, n.poses_left_hand[n.current_frame_id][np.newaxis, ...]], dim=0)
                    self.exported.poses_right_hand = torch.cat([self.exported.poses_right_hand, n.poses_right_hand[n.current_frame_id][np.newaxis, ...]], dim=0)
                    self.exported.trans = torch.cat([self.exported.trans, n.trans[n.current_frame_id][np.newaxis, ...]], dim=0)
                    self.exported.vertices = np.concatenate([self.exported.vertices, n.vertices[n.current_frame_id][np.newaxis, ...]], axis=0)
                    self.exported.joints = np.concatenate([self.exported.joints, n.joints[n.current_frame_id][np.newaxis, ...]], axis=0)
                    self.exported.betas = torch.cat([self.exported.betas, n.betas], dim=0)
                    self.exported.n_frames = self.exported.trans.shape[0]                    
                    self.exported.current_frame_id = self.exported.n_frames - 1
                    for inner in self.exported.nodes:
                        inner.n_frames = self.exported.n_frames
                        inner.current_frame_id = self.exported.current_frame_id
                    self.exported.rbs.joints = self.exported.joints
                    self.exported.mesh_seq.vertices = self.exported.vertices                    
                    self.exported.genders.append(n.smpl_layer.bm.gender)
                    self.exported.names.append(n.name)
            imgui.same_line(spacing=15)
            mode_clicked = imgui.button(f"Remove Frame#{self.exported.current_frame_id}", width=125, height=50)
            if mode_clicked and len(self.exported.genders):
                removed_id = self.exported.current_frame_id
                print(f"Removing frame #{removed_id} from {self.exported.name}.")
                self.exported.poses_body = torch.cat([
                    self.exported.poses_body[:removed_id, :], self.exported.poses_body[removed_id+1:, :]
                ], dim=0)
                self.exported.poses_root = torch.cat([
                    self.exported.poses_root[:removed_id, :], self.exported.poses_root[removed_id+1:, :]
                ], dim=0)
                self.exported.poses_left_hand = torch.cat([
                    self.exported.poses_left_hand[:removed_id, :], self.exported.poses_left_hand[removed_id+1:, :]
                ], dim=0)
                self.exported.poses_right_hand = torch.cat([
                    self.exported.poses_right_hand[:removed_id, :], self.exported.poses_right_hand[removed_id+1:, :]
                ], dim=0)
                self.exported.trans = torch.cat([
                    self.exported.trans[:removed_id, :], self.exported.trans[removed_id+1:, :]
                ], dim=0)
                self.exported.vertices = np.concatenate([
                    self.exported.vertices[:removed_id, :, :], self.exported.vertices[removed_id+1:, :, :]
                ], axis=0)
                self.exported.joints = np.concatenate([
                    self.exported.joints[:removed_id, :, :], self.exported.joints[removed_id+1:, :, :]
                ], axis=0)
                self.exported.betas = torch.cat([
                    self.exported.betas[:removed_id, :], self.exported.betas[removed_id+1:, :]
                ], dim=0)
                self.exported.n_frames = self.exported.trans.shape[0]                    
                self.exported.current_frame_id = self.exported.n_frames - 1
                self.exported.mesh_seq.vertices = self.exported.vertices
                self.exported.mesh_seq.n_frames = self.exported.n_frames
                self.exported.mesh_seq.current_frame_id = self.exported.current_frame_id
                self.exported.skeleton_seq.joints = self.exported.joints
                self.exported.skeleton_seq.n_frames = self.exported.n_frames
                self.exported.skeleton_seq.current_frame_id = self.exported.current_frame_id
                self.exported.rbs.joints = self.exported.joints
                self.exported.genders.pop(removed_id-1)
                self.exported.names.pop(removed_id-1)
            imgui.same_line(spacing=15)
            mode_clicked = imgui.button(f"Export", width=125, height=50)
            if mode_clicked and len(self.exported.genders) > 1:
                print('export')
                filename = os.path.join(C.export_dir, self.part, self.subject, f"{self.action}.npz")
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                np.savez(filename,
                    betas=to_numpy(self.exported.betas[1:]),
                    trans=to_numpy(self.exported.trans[1:]),
                    root_orient=to_numpy(self.exported.poses_root[1:]),
                    pose_body=to_numpy(self.exported.poses_body[1:]),
                    pose_hand=to_numpy(torch.cat([
                            self.exported.poses_left_hand[1:],
                            self.exported.poses_right_hand[1:],
                        ], dim=1
                    )),
                    pose_jaw=np.zeros_like(to_numpy(self.exported.poses_root[1:])),
                    pose_eye=np.zeros_like(to_numpy(
                        torch.cat([
                            self.exported.poses_root[1:],
                            self.exported.poses_root[1:],
                        ], dim=1)
                    )),
                    gender=np.array(self.exported.genders[1], dtype='U6'),
                )
            imgui.text(f"Exported Frame #{self.exported.current_frame_id+1} / {self.exported.n_frames}")
            imgui.end_group()
        imgui.end()

if __name__ == '__main__':    
    v = ExportingViewer()
    v.run_animations = True
    v.scene.camera.position = np.array([10.0, 2.5, 0.0])    
    v.run()
