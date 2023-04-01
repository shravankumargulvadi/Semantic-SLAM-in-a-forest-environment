import open3d as o3d
import numpy as np
import shutil
import os


def crop(points, 
         x_min, 
         x_max, 
         y_min, 
         y_max, 
         z_min, 
         z_max):
    req_idx = (points[:, 0] >= x_min) & \
              (points[:, 0] <= x_max) & \
              (points[:, 1] >= y_min) & \
              (points[:, 1] <= y_max) & \
              (points[:, 2] >= z_min) & \
              (points[:, 2] <= z_max)
    return points[req_idx]

def main():
    folder_name = "./forest_refined/"
    shutil.rmtree(folder_name)
    os.mkdir(folder_name)
    

    pcd = o3d.io.read_point_cloud("./forest.pcd")
    points = np.asarray([pcd.points])[0]
    points[:, [1, 2]] = points[:, [2, 1]]
    points.shape
    min_z, max_z = points[:, -1].min(), points[:, -1].max()
    min_x, max_x = points[:, 0].min(), points[:, 0].max()
    min_y, max_y = points[:, 1].min(), points[:, 1].max()
    min_x, max_x, min_y, max_y, min_z, max_z
    normal_xs = np.linspace(0, max_x, 10)
    normal_ys = np.linspace(0, max_y, 13)
    xs, ys = np.meshgrid(normal_xs, normal_ys)
    xs = xs.flatten()
    ys = ys.flatten()
    radius = 500
    points_new = np.zeros((0, 3))
    for patch_no, (x, y) in enumerate(zip(xs, ys)):
        # print(x, y)
        if patch_no % 3 != 0:
            continue
        req_points = crop(points, x, x+2*radius, y, y+2*radius, 0, max_z)
        # plot_req_points(req_points)
        if req_points.shape[0] >= 100:
            np.savetxt(f'./{folder_name}/{patch_no}.csv', req_points)
            points_new = np.concatenate((points_new, req_points), axis=0)
        # print(req_points.shape)
    print(points_new.shape)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_new)
    o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud('./forest_new.pcd', pcd)

if __name__ == '__main__':
    main()