
import numpy as np
import os
from scipy.optimize import minimize, least_squares
import open3d as o3d


def objective_fun_circle(s, p):
    '''
    s has 3 elements
        1. x coord of center
        2. y coord of center
        3. r radius of the circle
    '''
    obj = 0
    for point in p:
        obj += (s[2]**2 - (point[0] - s[0])**2 - (point[1] - s[1])**2)**2
    return obj

def direct_circle(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_ = x - x_mean
    y_ = y - y_mean
    r = np.sqrt(np.sum((x_**2 + y_**2)) / x_.shape[0])
    return r

def objective_fun_circle_least(s, p):
    '''
    s has 3 elements
        1. x coord of center
        2. y coord of center
        3. r radius of the circle
    '''
    obj = np.sqrt(np.abs(s[2]**2 - (p[:, 0] - s[0])**2 - (p[:, 1] - s[1])**2))
    return obj

def objective_fun(s, p):
    '''
    s has 5 elements
        1. rho
        2. phi
        3. v
        4. alpha
        5. K
    p has size N*3
    '''
    s[4] = s[4]**2
    s[0] = s[0]**2
    n = np.asarray([np.cos(s[1]) * np.sin(s[2]), 
                    np.sin(s[1]) * np.sin(s[2]), 
                    np.cos(s[2])])
    n_v = np.asarray([np.cos(s[1]) * np.cos(s[2]), 
                      np.sin(s[1]) * np.cos(s[2]), 
                      -np.sin(s[2])])
    n_phi = np.asarray([-np.sin(s[1]), 
                        np.cos(s[1]), 
                        0])
    a = n_v*np.cos(s[3]) + n_phi*np.sin(s[3])
    obj = 0
    for i in range(p.shape[0]):
        point = p[i].reshape(-1)
        curr = ((s[4]/2) * (np.linalg.norm(point)**2 - 
                            2*s[0]*np.dot(point.reshape(1, -1), n) - 
                            np.dot(point.reshape(1, -1), a)**2 + (s[0])**2) +  
                s[0] - np.dot(point.reshape(1, -1), n))
        obj += curr
        #obj = np.concatenate((obj, curr), axis=0)
    print(obj)
    return obj

def const1(s):
    return s[0]-0.01

def const2(s):
    return s[4]-0.01    

def const3(s):
    n = np.asarray([np.cos(s[1]) * np.sin(s[2]), 
                    np.sin(s[1]) * np.sin(s[2]), 
                    np.cos(s[2])])
    return n[0]**2 + n[1]**2 + n[2]**2 - 1

def const4(s):
    n = np.asarray([np.cos(s[1]) * np.sin(s[2]), 
                    np.sin(s[1]) * np.sin(s[2]), 
                    np.cos(s[2])])
    n_v = np.asarray([np.cos(s[1]) * np.cos(s[2]), 
                      np.sin(s[1]) * np.cos(s[2]), 
                      -np.sin(s[2])])
    n_phi = np.asarray([-np.sin(s[1]), 
                        np.cos(s[1]), 
                        0])
    a = n_v*np.cos(s[3]) + n_phi*np.sin(s[3])
    return n[0]*a[0] + n[1]*a[1] + n[2]*a[2]

def const5(s):
    n_v = np.asarray([np.cos(s[1]) * np.cos(s[2]), 
                      np.sin(s[1]) * np.cos(s[2]), 
                      -np.sin(s[2])])
    n_phi = np.asarray([-np.sin(s[1]), 
                        np.cos(s[1]), 
                        0])
    a = n_v*np.cos(s[3]) + n_phi*np.sin(s[3])
    return a[0]**2 + a[1]**2 + a[2]**2 - 1

def const6(s):
    return s[2]

def main():
    file_path = "./tree.npy"
    features = np.load(file_path)
    #features[:, [0, 1]] = features[:, [1, 0]]
    print(np.max(features[:, 0]), np.min(features[:, 0]),
          np.max(features[:, 1]), np.min(features[:, 1]),
          np.max(features[:, 2]), np.min(features[:, 2]))
    print(features.shape)
    features /= 1000
    r = direct_circle(features[:, 0], features[:, 1])
    print("radius ", r)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(features)
    # o3d.visualization.draw_geometries([pcd])

    # # decision variable s has 5 elements
    # s_0 = [0.1, 1, 1, 1, 1/0.3]
    # cons = [{'type': 'ineq', 'fun': const1},
    #         {'type': 'ineq', 'fun': const2},
    #         # {'type': 'eq', 'fun': const3},
    #         # {'type': 'eq', 'fun': const4},
    #         # {'type': 'eq', 'fun': const5}
    #         ]
    # res = minimize(objective_fun, 
    #                s_0, 
    #                args=features, 
    #                options={'disp': True,
    #                         'maxiter':10000},
    #                method='cobyla',
    #                constraints=cons
    #                )
    s_0 = [1, 1, 0.1]
    cons = [{'type': 'ineq', 'fun': const6}]
    # res = least_squares(objective_fun_circle_least, s_0, args=([features]))
    res = minimize(objective_fun_circle, 
                   s_0, 
                   args=features, 
                   options={'disp': True,
                            'maxiter':10000},
                   #method='Nelder-Mead',
                   constraints=cons
                   )

    print(res)

if __name__ == '__main__':
    main()