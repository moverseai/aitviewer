from aitviewer.scene.camera import OpenCVCamera
from aitviewer.configuration import CONFIG as C
import numpy as np

class K4A(OpenCVCamera):
    """
    Kinect for Azure camera wrapper.
    """
    def __init__(self, K, Rt, cols, rows, dist_coeffs=None, near=C.znear, far=C.zfar, viewer=None, **kwargs):
        super().__init__(K, Rt, cols, rows, dist_coeffs, near, far, viewer, **kwargs)

    def on_frame_update(self):
        self.position = self.current_Rt[:,3]
        self.rotation = self.current_Rt[:3,:3]
    
    @property
    def current_position(self):
        Rt = self.current_Rt
        pos = self.current_Rt[:,3] #-Rt[:, 0:3].T @ Rt[:, 3]
        return pos

    @property
    def current_rotation(self):
        Rt = self.current_Rt
        rot = np.copy(Rt[:, 0:3])
        rot[:, 1:] *= -1.0
        return rot