#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/06/19 

import sys
sys.path.append('repo/fast_resampling_via_graph')
from graph_filter import compute_scores_from_points

from pathlib import Path
import numpy as np
import open3d as o3d
from open3d.cpu.pybind.geometry import PointCloud


def load_pcd(fp:Path) -> PointCloud:
  return o3d.io.read_point_cloud(fp)


def sample_points_topk_idx(points, scores, n_samples):
  top_k = np.argsort(scores)
  ids_sampled = top_k[::-1][:n_samples]
  return ids_sampled

def sample_points_prob_idx(points, scores, n_samples):
  scores = scores / np.sum(scores)
  ids_sampled = np.random.choice(points.shape[0], n_samples, replace=False, p=scores)
  return ids_sampled

def sample_points_idx(points, scores, n_samples, method):
  if method == 'prob':
    return sample_points_prob_idx(points, scores, n_samples)
  elif method == 'topk':
    return sample_points_topk_idx(points, scores, n_samples)


def sample_pcd_hijack(pcd, filter_type, n_samples, scale_min_dist, scale_max_dist, max_neighbor, var, method = "prob"):
  assert method in ['prob', 'topk']

  scores = compute_scores_from_points(pcd, filter_type, scale_min_dist, scale_max_dist, max_neighbor, var)

  xyz_np = np.asarray(pcd.points)
  rgb_np = np.asarray(pcd.colors)
  ids_sampled = sample_points_idx(xyz_np, scores, n_samples, method)
  xyz_sampled = xyz_np[ids_sampled]
  rgb_sampled = rgb_np[ids_sampled]
  xyz = o3d.utility.Vector3dVector(xyz_sampled)
  rgb = o3d.utility.Vector3dVector(rgb_sampled)
  pcd_sampled = o3d.geometry.PointCloud(xyz)
  pcd_sampled.colors = rgb

  return pcd_sampled
