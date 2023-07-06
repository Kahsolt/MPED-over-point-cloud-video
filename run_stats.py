#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/06/19 

import json
from time import time
from pathlib import Path
from argparse import ArgumentParser
from collections import defaultdict
from typing import *

import sys
sys.path.append('repo/MPED/CV_task')
from MPED import MPED_VALUE
from Datalode import load_ply, farthest_point_sample, plot_3d_point_cloud

import torch
from torch import Tensor
import numpy as np
from tqdm import tqdm

from data import MOSs

DATA_PATH = Path('database')
REF_PATH = DATA_PATH / 'reference'
DST_PATH = DATA_PATH / 'distort'
OUT_PATH = Path('out') ; OUT_PATH.mkdir(exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'       # uncomment this if NOT CUDA is not supported


def load_pc(fp:Path) -> Tensor:
  return torch.from_numpy(np.asarray(load_ply(fp).T)).to(device, torch.float32)

def sample_pc(pc:Tensor, n:int) -> Tensor:
  return pc[farthest_point_sample(pc.unsqueeze(0), n).squeeze(0), :]

def get_MPED(src:Tensor, tgt:Tensor) -> float:
  return MPED_VALUE(src.unsqueeze(dim=0), tgt.unsqueeze(dim=0)).item()


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
  fp = OUT_PATH / f'stats_n={args.n}.json'
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
    ref_fps = sorted([str(fp) for fp in (REF_PATH / prefix).iterdir()])[args.s:args.t]
    dst_fps = {
      dp.name: sorted([str(fp) for fp in dp.iterdir()])[args.s:args.t] 
        for dp in sorted([dp for dp in DST_PATH.iterdir() if dp.name.startswith(prefix)])
    }
    for fps in dst_fps.values(): assert len(ref_fps) == len(fps), 'file count mismatch'

    print('>> calc MPEDs ...')
    for i, rfp in enumerate(tqdm(ref_fps)):
      rpc = sample_pc(load_pc(rfp), args.n)
      for name, dfps in dst_fps.items():
        if stats is not None and name in stats['data']:
          print(f'>> ignore {name} due to already cached')
          continue
        dpc = sample_pc(load_pc(dfps[i]), args.n)
        MPEDs[name].append(get_MPED(rpc, dpc))

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
  parser.add_argument('-s', default=None, type=int, help='start from frame idx')
  parser.add_argument('-t', default=None, type=int, help='stop at frame idx')
  args = parser.parse_args()
  
  run(args)
