#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/06/20 

import json
from pathlib import Path

import matplotlib.pyplot as plt
from scipy.stats import spearmanr as SROCC, pearsonr as PLCC

from data import MOSs_flat

OUT_PATH = Path('out') ; assert OUT_PATH.is_dir(), 'should run `run_stats.py` first!'


def run(fp):
  with open(fp, 'r', encoding='utf-8') as fh:
    data = json.load(fh)
  print(f'>> load stats from {fp}')

  n = data['args']['n']
  data = data['data']

  MPEDs = {k: v['value'] for k, v in data.items()}

  plt.figure(figsize=(12, 8))

  plt.clf()
  for name, values in MPEDs.items():
    plt.plot(values, label=name)
  plt.suptitle(f'MPED-{n} per frame')
  plt.legend(loc='upper left')
  fp = OUT_PATH / f'MPED_timeseq_n={n}.png'
  plt.savefig(fp, dpi=600)
  print(f'>> savefig to {fp}')

  plt.clf()
  for i, (name, values) in enumerate(MPEDs.items()):
    plt.subplot(len(MPEDs), 1, i+1)
    plt.plot(values, label=name)
    plt.legend(loc='upper left')
  plt.suptitle(f'MPED-{n} per frame')
  fp = OUT_PATH / f'MPED_timeseq_n={n}_seprated.png'
  plt.savefig(fp, dpi=600)
  print(f'>> savefig to {fp}')

  names = list(MPEDs.keys())
  plt.clf()
  plt.plot([MOSs_flat[name] for name in names], label='MOS')
  for agg in ['min', 'max', 'mean', 'std', 'var', 'median']:
    plt.plot([data[name][agg] for name in names], label=agg)
  plt.xticks(ticks=list(range(len(MPEDs))), labels=MPEDs.keys())
  plt.suptitle(f'MPED-{n} agg')
  plt.legend(loc='right')
  fp = OUT_PATH / f'MPED_agg_n={n}.png'
  plt.savefig(fp, dpi=600)
  print(f'>> savefig to {fp}')

  AGG_NAMES = ['min', 'max', 'mean', 'std', 'var', 'median']
  plccs, sroccs = [], []
  fp = OUT_PATH / f'cc-{n}.txt'
  with open(fp, 'w', encoding='utf-8') as fh:
    moss = [MOSs_flat[name] for name in names]
    for agg in AGG_NAMES:
      vals = [data[name][agg] for name in names]
      fh.write(f'[{agg}]:\n')
      plcc  = PLCC (moss, vals) ; plccs .append(-plcc .statistic)   # NOTE: make it negative for better view
      srocc = SROCC(moss, vals) ; sroccs.append(-srocc.statistic)
      fh.write(f'  plcc:  {plcc }\n')
      fh.write(f'  srocc: {srocc}\n')
    fh.write('\n')
  print(f'>> save cc score to {fp}')

  plt.clf()
  ax = plt.bar(AGG_NAMES, height=plccs)
  plt.bar_label(ax, labels=[f'{x:.4f}' for x in plccs])
  plt.suptitle(f'PLCC-{n}')
  fp = OUT_PATH / f'plcc_n={n}.png'
  plt.savefig(fp, dpi=600)
  print(f'>> savefig to {fp}')

  plt.clf()
  ax = plt.bar(AGG_NAMES, height=sroccs)
  plt.bar_label(ax, labels=[f'{x:.4f}' for x in sroccs])
  plt.suptitle(f'SROCC-{n}')
  fp = OUT_PATH / f'srocc_n={n}.png'
  plt.savefig(fp, dpi=600)
  print(f'>> savefig to {fp}')


if __name__ == '__main__':
  for fp in OUT_PATH.iterdir():
    if fp.suffix != '.json': continue
    print(f'>> process: {fp}')
    run(fp)
