""" Demo to show prediction results.
    Author: chenxi-wang
"""

import os
import sys
from scipy.spatial.transform import Rotation 

root_path = os.path.dirname(os.path.abspath(__file__))
root_path_1 = '/'.join(root_path.split('/')[:-1])
sys.path.append(root_path)
sys.path.append(root_path_1)

import numpy as np
import open3d as o3d
import argparse
import importlib
import scipy.io as scio
from PIL import Image
import copy
import cv2
import torch
from det_pkg.yolov7.detect_old import detect_icra_new, load_model
from IPython import embed

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

# from graspnet import GraspNet, pred_decode
# from graspnet_dataset import GraspNetDataset
# from collision_detector import ModelFreeCollisionDetector
# from data_utils import CameraInfo, create_point_cloud_from_depth_image

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', default='/home/gongjt4/graspnegt_ws/src/graspnet_pkg/src/checkpoint-rs.tar', help='Model checkpoint path')
parser.add_argument('--num_point', type=int, default=18000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
parser.add_argument('--depth_path', type=str, help='path of depth image')
parser.add_argument('--image_path', type=str, help='path of rgb image')
cfgs = parser.parse_args()


# def get_net():
#     # Init the model
#     net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
#             cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     net.to(device)
#     # Load checkpoint
#     checkpoint = torch.load(cfgs.checkpoint_path)
#     net.load_state_dict(checkpoint['model_state_dict'])
#     start_epoch = checkpoint['epoch']
#     print("-> loaded checkpoint %s (epoch: %d)"%(cfgs.checkpoint_path, start_epoch))
#     # set model to eval mode
#     net.eval()
#     return net

def get_mask(color, p1, p2):
    mask = np.zeros((color.shape[0],color.shape[1]))
    # mask[int(p1[0]):int(p2[0]), int(p1[1]):int(p2[1])] = 1
    mask[int(p1[1]):int(p2[1]), int(p1[0]):int(p2[0])] = 1
    mask = mask > 0
    return mask


def get_and_process_data(label,cv_color, cv_depth, cfgs, model_info):
    # load data
    # color = np.array(Image.open(cfgs.image_path), dtype=np.float32) / 255.0
    # depth = np.array(Image.open(cfgs.depth_path)).clip(0,1000)
    # image_path = "~/det_ws/src/det_pkg/scripts/1680706715.566328.png"
    # color = cv2.imread(image_path)

    color = cv_color
    depth = cv_depth.clip(0,1000)
    file_folder = os.path.dirname(os.path.abspath(__file__))
    print(file_folder)
    # with open('./meta.json') as f:
    with open(file_folder + '/meta.json') as f:
        import json
        meta = json.load(f)
    intrinsic = np.array(meta['intrinsic_matrix'], dtype=float)
    extrinsic = np.array(meta['extrinsic_matrix'], dtype=float)
    factor_depth = np.array(meta['factor_depth'], dtype=float)

    source       = color
    conf_thres   = 0.25
    iou_thres    = 0.45
    classes      = None
    agnostic_nms = False
    augment      = False

    cv2.imwrite('~/det_ws/src/det_pkg/scripts/cv_color.png', cv_color)
    # points (n, 6)
    # exit(-1)
    points = detect_icra_new(model_info, source, conf_thres, iou_thres, 
                    classes, agnostic_nms, augment)
    

    # needed_index = 5 # tomato
    points_new = None
    for point in points:
        print(point)
        print(label)
        if point[-1] == float(label):
            print('detect success!')
            points_new = point
    if points_new == None:
        return None, None, None
    else:
        points = points_new

    color = color / 255.0
    # points = points[0]
    # embed()
    # generate mask
    # p2 = [210,235]
    # p1 = [120,145]
    p1 = points[0:2]
    p2 = points[2:4]
    print('p1: ', p1)
    print('p2: ', p2)
    workspace_mask = get_mask(color, p1, p2)

    # generate cloud
    camera = CameraInfo(640.0, 360.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True) # (360, 640, 3)
    # embed()
    # get center_pose 
    p_center = ((p1 + p2) / 2).cpu()
    print('p_center: ', p_center)
    center_pose = cloud[int(p_center[1]), int(p_center[0])]
    print('center_pose', center_pose)


    # get valid points
    mask = (workspace_mask & (depth > 0))
    cloud_masked = cloud[mask]
    color_masked = color[mask]

    # convert points from camera to robot
    # rota = np.array([[0,-1, 0],
    #                 [0, 0, -1],
    #                 [1, 0, 0]])
    # cloud_masked = rota.dot(cloud_masked.T).T
    # cloud_masked = np.concatenate((cloud_masked, np.ones(cloud_masked.shape[0])[:,None]),axis=1)
    # cloud_masked = np.linalg.inv(extrinsic).dot(cloud_masked.T)
    # cloud_masked = cloud_masked.T[:,:3]

    # sample points
    if len(cloud_masked) >= cfgs.num_point:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point-len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    # convert data
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
    end_points = dict()
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = cloud_sampled.to(device)
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled


    return end_points, cloud, center_pose

def get_grasps(net, end_points):
    # Forward pass
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    return gg

def collision_detection(gg, cloud):
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
    gg = gg[~collision_mask]
    return gg

def vis_grasps(gg, cloud):
    # gg.nms()
    # gg.sort_by_score()
    # gg = gg[:1]
    grippers = gg.to_open3d_geometry_list()
    print("gg=", gg)

    o3d.visualization.draw_geometries([cloud, *grippers])

def save_results(save_path, gg):
    scores = torch.from_numpy(gg.scores)
    score, index = scores.topk(5)
    with open(save_path, 'a') as f:
        for i in range(5):
            rotat = gg.rotation_matrices[index[i].item()].flatten()
            trans = gg.translations[index[i].item()].flatten()
            depth = gg.depths[index[i].item()].flatten()
            # rota = -rota
            # trans = -trans

            depth_vector = np.zeros_like(trans)
            depth_vector[0] = depth
            final_grasp_point = np.dot(gg.rotation_matrices[index[i].item()], depth_vector.reshape(1,-1).T).T.reshape(-1) + trans
            trans = final_grasp_point

            rotat = ",".join(str(x) for x in rotat.tolist())
            trans = ",".join(str(x) for x in trans.tolist())
            name = 'top_' + str("%02d"%(i+1)) + ': ' + 'score=' + str(score[i].item()) + '\n'
            nr = 'rotat:' + '\n'
            tr = 'trans:' + '\n'
            f.write(name)
            f.write(nr)
            f.write(rotat)
            f.write('\n')
            f.write(tr)
            f.write(trans)
            f.write('\n')
            f.write('\n')
    f.close()
    rotat = gg.rotation_matrices[index[0].item()].flatten()
    trans = gg.translations[index[0].item()].flatten()
    return rotat, trans

def demo(save_path='./top5.txt'):
    net = get_net()
    end_points, cloud  = get_and_process_data(cfgs)
    gg = get_grasps(net, end_points)
    if cfgs.collision_thresh > 0:
        gg = collision_detection(gg, np.array(cloud.points))
    rotat, trans = save_results(save_path, gg)
    vis_grasps(gg, cloud)
    return rotat, trans

def ros_demo(grasp_enable, label, cv_color, cv_depth, model_info, save_path='./top5.txt'):
    
    
    end_points, cloud, center_pose  = get_and_process_data(label, cv_color, cv_depth, cfgs, model_info)
    
    if grasp_enable:
        if end_points == None:
            return None, None
        net = get_net()
        gg = get_grasps(net, end_points)
        if cfgs.collision_thresh > 0:
            gg = collision_detection(gg, np.array(cloud.points))
        rotat, trans = save_results(save_path, gg)
        gg.nms()
        gg.sort_by_score()
        gg = gg[:1]
        rotat = gg.rotation_matrices.flatten()
        trans = gg.translations.flatten()
        vis_grasps(gg, cloud)
        return rotat, trans
    else:
        if end_points == None:
            return None
        center_rotat=np.array([[0,1,0],
                               [0,0,1],
                               [1,0,0]])
        return center_rotat.flatten(), center_pose.flatten()

if __name__=='__main__':
    save_path = './top5.txt'
    demo(save_path)
