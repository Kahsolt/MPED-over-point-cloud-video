#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/06/19 

import json
from time import time
from pathlib import Path
from argparse import ArgumentParser
from collections import defaultdict
from traceback import print_exc
from typing import *

import sys
sys.path.append('repo/MPED/CV_task')
try:
  from MPED import MPED_VALUE
  from Datalode import load_ply, farthest_point_sample, plot_3d_point_cloud
except:
  print_exc()
try:
  from graph_filter_hijack import load_pcd, sample_pcd_hijack
except:
  print_exc()

import torch
from torch import Tensor
import numpy as np
from tqdm import tqdm

from data import MOSs
from MPED_with_color import SPED_score

DATA_PATH = Path('database')
REF_PATH = DATA_PATH / 'reference'
DST_PATH = DATA_PATH / 'distort'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'       # uncomment this if CUDA is NOT supported


def load_pc(fp:Path, with_color:bool=False) -> Tensor:
  if with_color:
    xyz, rgb = load_ply(fp, with_color=True)
    data = np.concatenate([xyz.T, rgb], axis=-1)    # [N, D=6]
  else:
    data = np.asarray(load_ply(fp).T)   # [N, D=3]
  return torch.from_numpy(data).to(device, torch.float32)

def sample_pc(pc:Tensor, n:int) -> Tensor:
  return pc[farthest_point_sample(pc[..., :3].unsqueeze(0), n).squeeze(0), :]

def fast_resample_pc(fp:Path, m:int) -> np.ndarray:
  pcd = load_pcd(fp)
  pcd_sampled = sample_pcd_hijack(pcd, 'high', m, 5, 5, 25, 10, 'topk')
  xyz = pcd_sampled.points
  rgb = pcd_sampled.colors
  return np.concatenate([xyz, rgb], axis=-1)    # [N, D=6]

def get_MPED(src:Tensor, tgt:Tensor) -> float:
  return MPED_VALUE(src.unsqueeze(dim=0), tgt.unsqueeze(dim=0)).item()

def get_MPED_color(src:np.ndarray, tgt:np.ndarray, fast:np.ndarray) -> float:
  SPED_5  = SPED_score(src, tgt, fast, 5,  '2-norm', 'RGB').item()
  SPED_10 = SPED_score(src, tgt, fast, 10, '2-norm', 'RGB').item()
  return  (SPED_5 + SPED_10) / 2


def perf_count(fn):
  def wrapper(*args, **kwargs):
    t = time()
    r = fn(*args, **kwargs)
    print(f'>> [Timer] done in {time() - t:.3f}s')
    return r
  return wrapper


@perf_count
@torch.inference_mode()
def run(args):
  # load stats.json if exists
  fp = Path(args.out_dp) / f'stats_n={args.n}.json'
  if fp.is_file():
    with open(fp, 'r', encoding='utf-8') as fh:
      stats = json.load(fh)
    # sanity check
    for k, v in stats['args'].items():
      if hasattr(args, k) and getattr(args, k) != v:
        print(f'>> cmdline args mismatch with cached `stats.json`!!')
        print(f'>> manually delete file {fp} then try again :(')
        exit(-1)
  else:
    stats = None
  
  # calc MPED
  MPEDs = defaultdict(list)
  for prefix in MOSs:
    print('>> check *.ply files ...')
    ref_fps = sorted([str(fp) for fp in (REF_PATH / prefix).iterdir()])[args.s:args.t:args.r]
    dst_fps = {
      dp.name: sorted([str(fp) for fp in dp.iterdir()])[args.s:args.t:args.r] 
        for dp in sorted([dp for dp in DST_PATH.iterdir() if dp.name.startswith(prefix)])
    }
    for fps in dst_fps.values(): assert len(ref_fps) == len(fps), 'file count mismatch'

    print('>> calc MPEDs ...')
    for i, rfp in enumerate(tqdm(ref_fps)):
      pc = load_pc(rfp, args.with_color)
      rpc = sample_pc(pc, args.n)
      if args.with_color: rpc_fast = fast_resample_pc(rfp, args.m)
      for name, dfps in dst_fps.items():
        if stats is not None and name in stats['data']:
          print(f'>> ignore {name} due to already cached')
          continue
        dpc = sample_pc(load_pc(dfps[i], args.with_color), args.n)
        score = get_MPED_color(rpc.cpu().numpy(), dpc.cpu().numpy(), rpc_fast) if args.with_color else get_MPED(rpc, dpc)
        MPEDs[name].append(score)

  # write MPED stats
  print('>> calc MPEDs stats ...')
  if stats is None:
    stats = {
      'args': vars(args),
      'data': {},
    }
  for name, values in MPEDs.items():
    values = np.asarray(values)
    stats['data'][name] = {
      'value':  values.tolist(),
      'min':    np.min   (values).item(),
      'max':    np.max   (values).item(),
      'mean':   np.mean  (values).item(),
      'std':    np.std   (values).item(),
      'var':    np.var   (values).item(),
      'median': np.median(values).item(),
    }
  with open(fp, 'w', encoding='utf-8') as fh:
    json.dump(stats, fh, ensure_ascii=False, indent=2)
  print(f'>> save stats to {fp}')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-n', default=1024, type=int, help='downsampled point count')
  parser.add_argument('-m', default=512,  type=int, help='downsampled point count for rpc_fast, should be smaller than -n')
  parser.add_argument('-s', default=None, type=int, help='start from frame idx')
  parser.add_argument('-t', default=None, type=int, help='stop at frame idx')
  parser.add_argument('-r', default=None, type=int, help='step size for frame idx')
  parser.add_argument('-O', '--out_dp', default='out', type=str, help='output folder')
  parser.add_argument('--with_color', action='store_true')
  args = parser.parse_args()
  
  Path(args.out_dp).mkdir(exist_ok=True)

  run(args)
