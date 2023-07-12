# MPED-over-point-cloud-video

​    Apply the MPED metric over point cloud videos to test whether it agrees with MOS, using PLCC & SROCC

----

### what's this

Apply the MPED metric over point cloud videos to test whether it agrees with MOS, using PLCC & SROCC

⚪ metrics & corelation-coefficient

metrics: 点云质量/保真度度量

  - MPED 提供了一种计算点云畸变的 **客观度量**
  - MOS 是一种通用 **主观评分**

corelation-coefficient: 衡量客观与主观的相关性

  - PLCC: Pearson’s linear correlation coefficient
  - SROCC: Spearman rank-order correlation coefficient

⚪ dataset

- `distort`: 4 个点云视频，都是 198 帧
- `reference`: 1 个点云视频，也是 198 帧
- `subjResults_dsis.xlsx`: 4 个点云视频的 MOS (主观评分，视作真值)


### quickstart

⚪ install once

- install [Build Tools for Visual Studio 2019](https://visualstudio.microsoft.com/downloads/)
- **manually** copy & run each cmdline in `install.cmd` to setup everything

⚪ run each time

- `conda activate pytorch3d`
- run `run.cmd` for MPED results & figures
- run `python show.py -f <*.ply> -n 4096` to visualize individual pc


### reference

- pytorch3d: [https://github.com/facebookresearch/pytorch3d](https://github.com/facebookresearch/pytorch3d)
- MPED: [https://github.com/Qi-Yangsjtu/MPED](https://github.com/Qi-Yangsjtu/MPED)
  - essay: [https://arxiv.org/abs/2103.02850](https://arxiv.org/abs/2103.02850)
- SROCC, KROCC, PLCC, RMSE: [https://blog.csdn.net/NewDay_/article/details/125255561](https://blog.csdn.net/NewDay_/article/details/125255561)

----
by Armit
2023/06/19 
