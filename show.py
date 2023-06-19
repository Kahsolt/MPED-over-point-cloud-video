#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/06/20 

from run_stats import *


def run(args):
  pc = load_pc(args.f)
  pc -= pc.mean(dim=0, keepdim=True)
  print('original:', pc.shape)

  if args.n:
    pc = sample_pc(pc, args.n)
    print('downsampled:', pc.shape)

  pc = pc.cpu().numpy()
  plot_3d_point_cloud(pc[:, 1], pc[:, 2], -pc[:, 0])   # rotate for better view


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-f', required=True, type=Path, help='path to *.ply file')
  parser.add_argument('-n', default=1024, type=int, help='downsampled point count')
  args = parser.parse_args()

  run(args)
