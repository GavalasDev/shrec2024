import open3d as o3d

from data import Marker, DataPoint, MotionCapture, MotionClass, Data

class HandVisualizer:
    """Visualizes a single frame (datapoint) of a mocap using open3d"""
    def __init__(self, dp: DataPoint):
        self.vertices = list(dp)
        self.edges = Marker.connections(index_only=True)

        self.geometries = []
        
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(self.vertices)

        point_colors = [[1.0, 0.0, 0.0]] * 14 + [[0.0, 0.0, 1.0]] * 14 # left is red, right is blue
        self.pcd.colors = o3d.utility.Vector3dVector(point_colors)
        
        self.geometries.append(self.pcd)

        self.lines = o3d.geometry.LineSet()
        self.lines.points = self.pcd.points
        self.lines.lines = o3d.utility.Vector2iVector(self.edges)
        self.geometries.append(self.lines)

        #mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
        #self.geometries.append(mesh_frame)

    def update_vertices(self, vertices):
        self.vertices = vertices
        self.pcd.points = o3d.utility.Vector3dVector(self.vertices)
        self.lines.points = self.pcd.points
    
    def show(self):
        o3d.visualization.draw_geometries(self.geometries)


class MocapVisualizer:
    """Visualizes an entire mocap using open3d"""
    def __init__(self, mocap: MotionCapture):
        self.mocap = mocap
        self.hand = HandVisualizer(self.mocap[0])
        self.animation_callback = self.animation_callback_generator()

    def animation_callback_generator(self):
        # looping animation
        i = -1
        while True:
            i += 1
            yield i % len(self.mocap)

    def update_geometry(self, vis):
        current_frame = next(self.animation_callback)
        current_dp = self.mocap[current_frame]
        
        self.hand.update_vertices(list(current_dp))

        # Draw updated geometries
        for geo in self.hand.geometries:
            vis.update_geometry(geo)

    def show(self):
        o3d.visualization.draw_geometries_with_animation_callback(self.hand.geometries, callback_function=self.update_geometry)