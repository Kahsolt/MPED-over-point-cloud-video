# NOTE: This file is auto-translated by ChatGPT4 from `repo\MPED\HVS_task\SPED_score.m`
# then  manually checked & modified to kill runtime error :(

# 导入所需的库
import torch
import numpy as np
from scipy.spatial import KDTree

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 定义颜色转换矩阵
LMN_matrix = np.asarray([[0.06,0.63,0.27],[0.3,0.04,-0.35],[0.34,-0.60,0.17]]).T
YUV_matrix = np.asarray([[0.299, 0.587,0.114],[-0.1678,-0.3313,0.5],[0.5,-0.4187,-0.0813]]).T

# 定义函数
def SPED_score(pc_ori, pc_dis, pc_fast, k=10, distance_type='2-norm', color_type='RGB'):
    # 检查k的值
    if k<1: raise ValueError('the numebr of neighbors should be bigger than 1!')

    # 获取点云的数量
    count = pc_fast.shape[0]

    # 获取点云的坐标和颜色
    center_coordinate = pc_fast[:,:3]
    source_coordinate = pc_ori[:,:3]
    target_coordinate = pc_dis[:,:3]

    center_color = pc_fast[:,3:]
    source_color = pc_ori[:,3:]
    target_color = pc_dis[:,3:]

    # 根据颜色类型进行颜色转换
    if color_type == 'LMN':
        center_color = np.dot(center_color, LMN_matrix)
        source_color = np.dot(source_color, LMN_matrix)
        target_color = np.dot(target_color, LMN_matrix)
    elif color_type == 'YUV':
        center_color = np.dot(center_color, YUV_matrix)
        source_color = np.dot(source_color, YUV_matrix)
        target_color = np.dot(target_color, YUV_matrix)

    # 建立邻域
    tree_source = KDTree(source_coordinate)
    tree_target = KDTree(target_coordinate)
    _, idx_source = tree_source.query(center_coordinate, k, workers=8)
    _, idx_target = tree_target.query(center_coordinate, k, workers=8)

    # 计算每个邻域的势能
    center_mass                = torch.from_numpy(center_color)                 .to(device)
    neighbor_source_mass       = torch.from_numpy(source_color     [idx_source]).to(device)
    neighbor_target_mass       = torch.from_numpy(target_color     [idx_target]).to(device)
    neighbor_source_coordinate = torch.from_numpy(source_coordinate[idx_source]).to(device)
    neighbor_target_coordinate = torch.from_numpy(target_coordinate[idx_target]).to(device)

    center_mass_rep = center_mass.unsqueeze(1).expand(-1, k, -1)
    source_mass_dif = torch.abs(neighbor_source_mass-center_mass_rep)
    target_mass_dif = torch.abs(neighbor_target_mass-center_mass_rep)
    if color_type == 'RGB':
        source_mass_dif = torch.sqrt(1*source_mass_dif[:,0]+2*source_mass_dif[:,1]+1*source_mass_dif[:,2]+1)
        target_mass_dif = torch.sqrt(1*target_mass_dif[:,0]+2*target_mass_dif[:,1]+1*target_mass_dif[:,2]+1)
    else:
        source_mass_dif = torch.sqrt(6*source_mass_dif[:,0]+1*source_mass_dif[:,1]+1*source_mass_dif[:,2]+1)
        target_mass_dif = torch.sqrt(6*target_mass_dif[:,0]+1*target_mass_dif[:,1]+1*target_mass_dif[:,2]+1)

    center_coordinate = torch.from_numpy(center_coordinate).to(device)
    center_coordinate_rep = center_coordinate.unsqueeze(1).expand(-1, k, -1)
    source_coordinate_dif = neighbor_source_coordinate - center_coordinate_rep
    target_coordinate_dif = neighbor_target_coordinate - center_coordinate_rep
    if distance_type == '1-norm':
        source_distance_dif = torch.sum(torch.abs(source_coordinate_dif), dim=1)
        target_distance_dif = torch.sum(torch.abs(target_coordinate_dif), dim=1)
    elif distance_type == '2-norm':
        source_distance_dif = torch.sqrt(torch.sum(source_coordinate_dif**2, dim=1))
        target_distance_dif = torch.sqrt(torch.sum(target_coordinate_dif**2, dim=1))
    else:
        raise ValueError('Wrong distance type! Please use 1-norm or 2-norm!')

    if k==1:
        g_source = 1
        g_target = 1
    else:
        g_source = 1./torch.sqrt(source_distance_dif + 1)
        g_target = 1./torch.sqrt(target_distance_dif + 1)

    energy_source = (source_mass_dif * g_source * source_distance_dif)
    energy_target = (target_mass_dif * g_target * target_distance_dif)
    energy_source_sum = torch.sum(energy_source, dim=0)
    energy_target_sum = torch.sum(energy_target, dim=0)

    energy_diff = torch.sum(torch.abs(energy_source_sum - energy_target_sum))/(count*k)
    resolution = torch.sqrt(((torch.sum(source_distance_dif)/(count*k))))
    score = energy_diff/resolution

    return score
