import numpy as np
import open3d.core as o3c
# deprecated now
class Frame():
    def __init__(self, frame_id, origin, points, normals, labels) -> None:
        
        self.frame_id = frame_id
        self.origin = origin
        self.points = points
        self.normals = normals
        self.labels = labels
        #self.ranges = torch.linalg.norm(self.points, dim=1, keepdim=True) # nx1
        
    def get_rays(self):
        return self.points - self.origin

    def get_points(self):
        return self.points

    def sample_data(self, N_rays):
        
        return 

    def get_sample_data(self, N_rays):

        return 

