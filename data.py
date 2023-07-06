#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/07/06 

# NOTE: these data are copied from 'subjResults_dsis.xlsx' and hard-coded
# config this to determine what files are to run

MOSs = {
  'matis_ply_062K': {
    'gQP17-tQP20': 73.1,
    'gQP30-tQP35': 58.4,
    'gQP37-tQP43': 29.7,
    'gQP41-tQP48': 12.7,
  },
  'rafa_ply_062K': {
    'gQP17-tQP20': 73.6,
    'gQP30-tQP35': 61.1,
    'gQP37-tQP43': 30.3,
    'gQP41-tQP48': 13.0,
  }
}

MOSs_flat = { }
for prefix, entry in MOSs.items():
  for name, mos in entry.items():
    MOSs_flat[f'{prefix}_{name}'] = mos
