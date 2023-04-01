import numpy as np
from scipy.optimize import leastsq
import argparse
import os
from motion_estimation_2 import find_association_trees
import open3d as o3d
import shutil
from tqdm import tqdm

def print_parameters(s):
    for i in range(s.shape[0]):
        if i == 2 or i == 3:
            print(f'{i}th parameter {np.round(np.rad2deg(s[i]), 4)} degrees')
        else:
            print(f'{i}th parameter {np.round(s[i], 4)} meters')

def cylinderFitting(xyz,p):
    x = xyz[:,0]
    y = xyz[:,1]
    z = xyz[:,2]
    fitfunc = lambda p, x, y, z: (- np.cos(p[3])*(p[0] - x) - \
                                    z*np.cos(p[2])*np.sin(p[3]) - \
                                    np.sin(p[2])*np.sin(p[3])*(p[1] - y))**2 +\
                                    (z*np.sin(p[2]) - np.cos(p[2])*(p[1] - y))**2 #fit function
    errfunc = lambda p, x, y, z: fitfunc(p, x, y, z) - p[4]**2 #error function

    est_p , success = leastsq(errfunc, p, args=(x, y, z), maxfev=1000)
    return est_p


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_name', default='forest')
    parser.add_argument('--output_foldername', default='forest_results')
    args = parser.parse_args()
    if os.path.exists(args.output_foldername):
        shutil.rmtree(args.output_foldername)
    os.mkdir(args.output_foldername)
    foldername = f'./{args.folder_name}/'
    output_foldername = f'./{args.output_foldername}/'
    for file_no, filename in tqdm(enumerate(sorted(
                                       os.listdir(foldername))),
                                       total=len(os.listdir(foldername))):
        # print(filename)
            
        xyz = np.loadtxt(os.path.join(foldername, filename))
        xyz /= 1000
        p = np.array([-13.79,-8.45,0,0,100])
        #[X_c,Y_c,angle_x,angle_y,radius]

        # print(f"Performing Cylinder Fitting for {file_no} ... ")
        est_p =  cylinderFitting(xyz, p)
        # print("Fitting Done!")

        # print("Estimated Parameters: ")
        # print_parameters(est_p)
        cost = find_association_trees(xyz, est_p)
        print(f'Cost {cost}')
        
        if cost >= 1:
            print(cost, est_p)   
            os.remove(os.path.join(foldername, filename))      
        else:
            np.savetxt(os.path.join(output_foldername, 
                       filename.split('.')[0]+'.txt'), est_p)            