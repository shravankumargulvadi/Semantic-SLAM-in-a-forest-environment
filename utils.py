import numpy as np
import os
import argparse
import random
from scipy.spatial.transform import Rotation
import open3d as o3d
from tqdm import tqdm

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
    # print(np.argmin(obj_fun), p[np.argmin(obj_fun)])
    return p[np.argmin(obj_fun)]

def to_homogenous(points):
    '''
    converts to homogenous coordinates
    points: n*3 array
    output: n*4 array
    '''
    points = np.concatenate((points, np.ones((points.shape[0],1))), axis=1)
    return points

def find_corres(points, p):
    corres_ps = []
    for point in tqdm(points, total=points.shape[0]):
        corres_p = find_association(point, p)
        corres_ps.append(corres_p)
    return np.asarray(corres_ps) 


def slam_objective(s, points, corres_ps):
    '''
    s = 6 elements with euler angles and translation
    points n*3
    '''
    r = Rotation.from_euler('zyx', s[:3].tolist(), degrees=True)
    rotation_matrix = r.as_matrix()
    motion_matrix = np.concatenate((rotation_matrix, s[3:].reshape((3, 1))), axis=1)
    points_trans = (motion_matrix@to_homogenous(points).T).T

if __name__ == '__main__':
    pass