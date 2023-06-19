@ECHO OFF

IF NOT "%CONDA_DEFAULT_ENV%"=="pytorch3d" (
  ECHO ^>^> run ^"conda activate pytorch3d^" first!
  EXIT /B
)

python run_stats.py -n 32
python run_stats.py -n 256
python run_stats.py -n 512
python run_stats.py -n 1024
python run_stats.py -n 2048
python run_stats.py -n 3072

python run_plots.py
