import numpy as np
import os
import argparse
import random
from scipy.spatial.transform import Rotation
import open3d as o3d
#import wandb
from utils import slam_objective, find_corres
from scipy.optimize import leastsq
from tqdm import tqdm


def find_tree_to_tree_association(trees, ps):
    '''
    trees: N*M*3 (N trees to M number per tree)
    ps: L*5 (for L previous trees)
    '''
    total_model = []
    req_ps = np.zeros((0, 5))
    for tree in tqdm(trees, total=len(trees)):
        min_distance = np.inf
        min_model_no = 0
        for model_no, p in enumerate(ps):
            curr_distance = find_association_trees(tree, p)
            if curr_distance < min_distance:
                min_model_no = model_no
                min_distance = curr_distance
        total_model.append(min_model_no)
        req_ps = np.concatenate((req_ps, ps[min_model_no].reshape((-1, 5))))
    print(len(total_model), sorted(total_model))
    return req_ps

def find_association_trees(tree, p):
    '''
    point: 3*1 x, y, z point
    p: N*5, array of parameters of N trees in previous frames
    '''
    x, y, z = tree[:, 0], tree[:, 1], tree[:, 2]
    obj_fun = (-np.cos(p[3])*(p[0]-x)-\
               z*np.cos(p[2])*np.sin(p[3])-\
               np.sin(p[2])*np.sin(p[3])*(p[1]-y))**2+\
               (z*np.sin(p[2]) - np.cos(p[2])*(p[1]-y))**2-\
               p[4]**2 # N*1
    # print(obj_fun.shape)
    obj_fun = np.abs(obj_fun)
    # print(np.argmin(obj_fun), p[np.argmin(obj_fun)])
    return np.sum(obj_fun**2)

def find_points_model_distance(tree, p):
    '''
    tree: M*3
    p: 1*5
    '''
    x, y, z = tree[:, 0], tree[:, 1], tree[:, 2]


def find_association(point, p):
    '''
    point: 3*1 x, y, z point
    p: N*5, array of parameters of N trees in previous frames
    '''
    x, y, z = point
    obj_fun = (-np.cos(p[:, 3])*(p[:, 0]-x)-\
               z*np.cos(p[:, 2])*np.sin(p[:, 3])-\
               np.sin(p[:, 2])*np.sin(p[:, 3])*(p[:, 1]-y))**2+\
               (z*np.sin(p[:, 2]) - np.cos(p[:, 2])*(p[:, 1]-y))**2-\
               p[:, 4]**2 # N*1
    obj_fun = np.abs(obj_fun)
    print(np.argmin(obj_fun), p[np.argmin(obj_fun)])

def load_tree_params(folder_name):
    '''
    load all the parameters and return an array and filename list
    '''
    p = np.zeros((0, 5))
    for file_no, filename in enumerate(sorted(os.listdir(folder_name))):
        filename = os.path.join(folder_name, filename)
        parameters = np.loadtxt(filename).reshape((1, -1))
        p = np.concatenate((p,  parameters), axis=0)
    return p

def plot_in_wandb(points):
    wandb.log({"Points Rotated and translated": wandb.Object3D(
                    {
                        "type": "lidar/beta",
                        "points": points
                    }
    )})

def generate_random_rotation_translation(min_x_angle=-10,
                                         max_x_angle=10,
                                         min_y_angle=-10,
                                         max_y_angle=10,
                                         min_z_angle=-10,
                                         max_z_angle=10,
                                         min_x_trans=0,
                                         max_x_trans=2,
                                         min_y_trans=0,
                                         max_y_trans=2,
                                         min_z_trans=0,
                                         max_z_trans=2):
    '''
    generates random 3 rotation euler angles and 3 translation along x, y, z axis
    returns
    3*4 (rotation + translation matrix)
    3*1 euler angles array
    3*1 translation array
    '''
    angle_x = random.uniform(min_x_angle, max_x_angle)
    angle_y = random.uniform(min_y_angle, max_y_angle)
    angle_z = random.uniform(min_z_angle, max_z_angle)
    trans_x = random.uniform(min_x_trans, max_x_trans)
    trans_y = random.uniform(min_y_trans, max_y_trans)
    trans_z = random.uniform(min_z_trans, max_z_trans)
    r = Rotation.from_euler('zyx', [angle_z, angle_y, angle_x], degrees=True)
    rotation_matrix = r.as_matrix()
    motion_matrix = np.concatenate((rotation_matrix, np.asarray([trans_x,
                                                                 trans_y,
                                                                 trans_z]).reshape((3, 1))), axis=1)
    return motion_matrix, np.asarray([angle_z, angle_y, angle_x]), np.asarray([trans_x, trans_y, trans_z])

def generate_transformation_matrix(euler_angles, translations):
    r = Rotation.from_euler('zyx', euler_angles.tolist(), degrees=True)
    rotation_matrix = r.as_matrix()
    motion_matrix = np.concatenate((rotation_matrix, translations.reshape((3, 1))), axis=1)
    return motion_matrix

def to_homogenous(points):
    '''
    converts to homogenous coordinates
    points: n*3 array
    output: n*4 array
    '''
    points = np.concatenate((points, np.ones((points.shape[0],1))), axis=1)
    return points

def add_noise(array, mean=0, std=2):
    '''
    adds gaussian noise to array with given mean and std    
    '''
    noise = np.random.normal(mean, std, size=array.shape)
    return array + noise

def transform_pcd(points, T, color=np.asarray([255, 0, 0])):
    colors = np.tile(color/255, (points.shape[0], 1))
    if T is None:
        points_transformed = points
    else:
        points_transformed = (T@to_homogenous(points).T).T
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_transformed)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def main():
    # wandb.init("optimization_project_vis")
    # random.seed(1001)
    # np.random.seed(1001)
    parser = argparse.ArgumentParser()
    parser.add_argument('--full_forest_filename', 
                        default="./forest_new.pcd")
    parser.add_argument('--tree_folder_name',
                        default="./forest/")
    parser.add_argument('--output_folder_name',
                        default="./forest_results/")
    args = parser.parse_args()

    T, euler_angles, translations = generate_random_rotation_translation()
    # print("req ", euler_angles, Rotation.from_matrix(T[:, :3]).as_euler('zyx', degrees=True))
    noisy_euler_angles = add_noise(euler_angles, 0, 1)
    noisy_translations = add_noise(translations, 0, 0.1)
    print(f"unoisy euler {euler_angles}\n, unoisy trans {translations}\n\
            noisy euler {noisy_euler_angles}\n, noisy trans {noisy_translations}\n")
    noisy_T = generate_transformation_matrix(noisy_euler_angles, noisy_translations)
    print(T)
    full_pcd = o3d.io.read_point_cloud(args.full_forest_filename)
    full_pcd_points = np.asarray(full_pcd.points)
    full_pcd_points /= 1000
    p = load_tree_params(args.output_folder_name)

    tree_segments = [np.loadtxt(os.path.join(args.tree_folder_name, tree_segment_filename))/1000 \
                     for tree_segment_filename in os.listdir(args.tree_folder_name)]
    trt = find_tree_to_tree_association(tree_segments, p)
    print(p.shape, len(tree_segments), " trt:", trt.shape , "\n")

    # inverse_noisy_R = np.linalg.inv(noisy_T[:, :3])
    # inverse_noisy_T = -np.linalg.inv(noisy_T[:, :3])@noisy_T[:, -1].reshape((3, 1))
    # noisy_euler_angles_inv = Rotation.from_matrix(inverse_noisy_R).as_euler('zyx', degrees=True)
    # s_0 = np.concatenate((noisy_euler_angles_inv.reshape(-1), inverse_noisy_T.reshape(-1)))
    # print(s_0)

    # pcd1 = transform_pcd(full_pcd_points, None, np.asarray([0, 255, 0]))
    # pcd2 = transform_pcd(full_pcd_points, T, np.asarray([0, 0, 255]))
    # pcd3 = transform_pcd(np.asarray(pcd2.points), 
    #                      generate_transformation_matrix(noisy_euler_angles_inv, inverse_noisy_T), 
    #                      np.asarray([0, 255, 255]))
    
    # corres_p = find_corres(np.asarray(pcd2.points), p)
    # print(corres_p.shape)
    # est_p , success = leastsq(slam_objective, 
    #                           s_0, 
    #                           args=(np.asarray(pcd2.points), corres_p), 
    #                           maxfev=100)
    # print(est_p)

    # ###
    # T_res = generate_transformation_matrix(np.asarray(est_p[:3]), np.asarray(est_p[3:]))
    # pcd4 = transform_pcd(np.asarray(pcd2.points), T_res, np.asarray([255, 0, 0]))
    # ###

    # o3d.visualization.draw_geometries([pcd1, pcd3, pcd4])

if __name__ == '__main__':
    main()
    