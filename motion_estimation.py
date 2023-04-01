import numpy as np
from scipy.optimize import leastsq
from scipy.spatial.transform import Rotation
import argparse
import os
from motion_estimation_2 import find_tree_to_tree_association, \
                                load_tree_params, \
                                generate_random_rotation_translation, \
                                add_noise, \
                                generate_transformation_matrix, \
                                to_homogenous
from tqdm import tqdm
import random
import open3d as o3d


def print_parameters_motion_est(s):
    for i in range(s.shape[0]):
        if i >2:
            print(f'{i}th parameter {np.round(s[i], 4)} degrees')
        else:
            print(f'{i}th parameter {np.round(s[i], 4)} meters')


def get_transf_mat(T):
    '''T=[translation_x,translation_y,translation_z,rotation_x,rotation_y,rotation_z]
    convert Tvec to homogenous transformation matrix'''
    r = Rotation.from_euler("zyx", T[3:], degrees=True)
    rotation_matrix = r.as_matrix()
    transf_mat=np.hstack((rotation_matrix, np.reshape(T[0:3],(3,1))))
    transf_mat= np.vstack((transf_mat, np.asarray([0,0,0,1])))
    return transf_mat

def errfunc(T, xyz_old, p):
    Transf_mat=get_transf_mat(T)
    Coords=np.hstack((xyz_old,np.ones((xyz_old.shape[0],1))))
    xyz_new=Transf_mat@np.transpose(Coords) #project using current transformation
    xyz_new=np.transpose(xyz_new)
  
    return cylinderdistance(xyz_new,p)

def cylinderdistance(xyz,p):
    x = xyz[:,0]
    y = xyz[:,1]
    z = xyz[:,2]
    fitfunc = lambda p, x, y, z: (- np.cos(p[3])*(p[0] - x) - \
                                    z*np.cos(p[2])*np.sin(p[3]) - \
                                    np.sin(p[2])*np.sin(p[3])*(p[1] - y))**2 +\
                                    (z*np.sin(p[2]) - np.cos(p[2])*(p[1] - y))**2 #fit function
    error = fitfunc(p, x, y, z) - p[4]**2 #error function  
    return error

def sum_cyl_dist_error(T, cyl_params, tree_segments):
    errors = np.zeros((0, 1))
    for file_no, tree in tqdm(enumerate(tree_segments)):
        param = cyl_params[int(file_no)]
        xyz = tree
        # xyz /= 1000
        error=errfunc(T, xyz, param)
        errors = np.concatenate((errors, 
                                 error.reshape((-1, 1))), axis=0)
    print(np.mean(errors**2))
    return errors.reshape(-1)
            

if __name__=="__main__":
    # random.seed(100)
    # np.random.seed(1001)
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_name', default='forest')
    parser.add_argument('--results_folder_name', default='forest_results')
    args = parser.parse_args()
    tree_folder_name = args.folder_name
    output_folder_name = args.results_folder_name
    p = load_tree_params(output_folder_name)
    tree_segments = [np.loadtxt(os.path.join(tree_folder_name, tree_segment_filename))/1000 \
                     for tree_segment_filename in os.listdir(tree_folder_name)]
    T, euler_angles, translations = generate_random_rotation_translation()
    
    transformed_tree_segments = [(T@to_homogenous(tree_segment).T).T for tree_segment in tree_segments]
    
    ###
    noisy_euler_angles = add_noise(euler_angles, 0, 2)
    noisy_translations = add_noise(translations, 0, 0.1)
    noisy_T = generate_transformation_matrix(noisy_euler_angles, noisy_translations)
    noisy_tree_segments = [(noisy_T@to_homogenous(tree_segment).T).T for tree_segment in tree_segments]
    
    inverse_noisy_R = np.linalg.inv(noisy_T[:, :3])
    inverse_noisy_t = -inverse_noisy_R@noisy_T[:, -1]
    noisy_euler_angles_inv = Rotation.from_matrix(
                                inverse_noisy_R).as_euler(
                                    'zyx', degrees=True)
    s_0 = np.concatenate((inverse_noisy_t.reshape(-1), 
                          noisy_euler_angles_inv.reshape(-1)))
    inverse_noisy_T = generate_transformation_matrix(noisy_euler_angles_inv, 
                                                     inverse_noisy_t)
    inv_transformed_segments = [(inverse_noisy_T@to_homogenous(tree_segment).T).T 
                                for tree_segment in transformed_tree_segments]
    print(s_0)
    cyl_params = find_tree_to_tree_association(inv_transformed_segments, p)
    # print(inv_transformed_segments)
    est_p , success = leastsq(sum_cyl_dist_error,
                              s_0, 
                              args=(cyl_params, transformed_tree_segments), 
                              maxfev=1000)
    resultant_T = generate_transformation_matrix(est_p[3:], est_p[:3])
    resultant_segments = [(resultant_T@to_homogenous(tree_segment).T).T 
                           for tree_segment in transformed_tree_segments]

    full_pcd_org = np.concatenate(tree_segments, 0).reshape(-1, 3)
    transformed_full_pcd = np.concatenate(transformed_tree_segments, 0).reshape(-1, 3)
    noisy_full_pcd = np.concatenate(noisy_tree_segments, 0).reshape(-1, 3)
    inv_full_pcd = np.concatenate(inv_transformed_segments, 0).reshape(-1, 3)
    result_full_pcd = np.concatenate(resultant_segments, 0).reshape(-1, 3)
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(full_pcd_org)
    pcd1.colors = o3d.utility.Vector3dVector(np.tile(np.asarray([255, 0, 0])/255, 
                                            (full_pcd_org.shape[0], 1)))
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(transformed_full_pcd)
    pcd2.colors = o3d.utility.Vector3dVector(np.tile(np.asarray([0, 255, 0])/255, 
                                            (transformed_full_pcd.shape[0], 1)))
    pcd3 = o3d.geometry.PointCloud()
    # pcd3.points = o3d.utility.Vector3dVector(noisy_full_pcd)
    # pcd3.colors = o3d.utility.Vector3dVector(np.tile(np.asarray([0, 0, 255])/255, 
    #                                         (noisy_full_pcd.shape[0], 1)))
    pcd4 = o3d.geometry.PointCloud()
    pcd4.points = o3d.utility.Vector3dVector(inv_full_pcd)
    pcd4.colors = o3d.utility.Vector3dVector(np.tile(np.asarray([0, 255, 255])/255, 
                                            (inv_full_pcd.shape[0], 1)))
    pcd5 = o3d.geometry.PointCloud()
    pcd5.points = o3d.utility.Vector3dVector(result_full_pcd)
    pcd5.colors = o3d.utility.Vector3dVector(np.tile(np.asarray([255, 255, 0])/255, 
                                            (result_full_pcd.shape[0], 1)))
    #o3d.visualization.draw_geometries([pcd1, pcd2, pcd4, pcd5])
    o3d.visualization.draw_geometries([pcd1,pcd2, pcd5])

    if success:
        print('success!!!')
    print("outer loop Done!")
    print("Estimated Parameters: ")
    print_parameters_motion_est(est_p)
        
      