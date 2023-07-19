import argparse
import numpy as np
import open3d
import time

class TimeIt:
    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        self.start = time.perf_counter()

    def __exit__(self, type, value, traceback):
        self.end = time.perf_counter()
        elapsed = self.end - self.start
        print(f'{self.msg} : {elapsed:0.3f} seconds')

def voxelize_point_cloud(data):
    return open3d.geometry.VoxelGrid.create_from_point_cloud(data, voxel_size=0.05)

def convert_voxel_to_np_array_rgb_point_cloud(voxel_data):
    def get_point_as_list(pt):
        pt3d = list(voxel_data.origin + pt.grid_index * voxel_data.voxel_size)
        ptcolor = list(pt.color)
        return pt3d + ptcolor

    return np.asarray([get_point_as_list(pt) for pt in voxel_data.get_voxels()])

def get_open3d_point_cloud_from_np_array(points):
    cloud = open3d.geometry.PointCloud()
    cloud.points = open3d.utility.Vector3dVector(points[:, 0:3])
    cloud.colors = open3d.utility.Vector3dVector(points[:, 3:])
    return cloud

def octree_operations(point_cloud_data):
    octree = open3d.geometry.Octree(max_depth=8)
    octree.convert_from_point_cloud(point_cloud_data)
    leaf_node, tree_info = octree.locate_leaf_node(point_cloud_data.points[0])
    print('Octree :', leaf_node, ' :: ', type(leaf_node))
    print('Indices of points', leaf_node.indices, '\nTotal points = ', len(leaf_node.indices))

def main():
    parser = argparse.ArgumentParser(description='Tool to visualize Open3d point clouds')
    parser.add_argument('--filename', type=str, help='Input .pts file', required=True)
    parser.add_argument('--voxelize', default=False, help='For downsampling', action='store_true')
    args = parser.parse_args()

    with TimeIt('Time to load : ') as time_it:
        data = open3d.io.read_point_cloud(args.filename)

    octree_operations(data)

    if args.voxelize:
        with TimeIt('Time for voxelization : ') as time_it:
            data = voxelize_point_cloud(data)

        print('After voxelization : ', data)

        new_data = convert_voxel_to_np_array_rgb_point_cloud(data)

    open3d.visualization.draw_geometries([get_open3d_point_cloud_from_np_array(new_data)], zoom=0.3412, 
                                      front=[0.4257, -0.2125, -0.8795], 
                                      lookat=[2.6172, 2.0475, 1.532], 
                                      up=[-0.0694, -0.9768, 0.2024])

if __name__ == "__main__":
    main()
