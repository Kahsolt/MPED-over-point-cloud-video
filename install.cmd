@ECHO OFF
REM install the fucking pytorch3d stuff

ECHO ^>^> DO NOT RUN THIS SCRIPT DIRECTLY!!
ECHO ^>^> manually copy and run the cmdlines below :)
ECHO.
FOR /F "skip=9 delims=" %%l IN (%~f0) DO ECHO %%l
EXIT /B


PUSHD repo

CALL init_repos.cmd

PUSHD pytorch3d
conda create -n pytorch3d python=3.9
conda activate pytorch3d
conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath

REM If you get any error runing cmdlines below, follow the official guide:
REM https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md

CALL "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
SET DISTUTILS_USE_SDK=1
SET FORCE_CUDA=1
python setup.py install
python -m unittest discover -v -s tests -t .
POPD

pip install plyfile
pip install matplotlib
pip install scipy
pip install open3d

POPD
