@echo off
REM Get the directory where this script is located
setlocal enabledelayedexpansion
set UNITREE_RL_LAB_PATH=%~dp0
set UNITREE_RL_LAB_PATH=%UNITREE_RL_LAB_PATH:~0,-1%

REM Check if conda environment is activated
if "%CONDA_PREFIX%"=="" (
    echo [Error] No conda environment activated. Please activate the conda environment first.
    exit /b 1
)

set python_exe=%CONDA_PREFIX%\python.exe

REM Parse command line arguments
if "%1"=="" goto unknown
if "%1"=="-i" goto install
if "%1"=="--install" goto install
if "%1"=="-l" goto list
if "%1"=="--list" goto list
if "%1"=="-p" goto play
if "%1"=="--play" goto play
if "%1"=="-t" goto train
if "%1"=="--train" goto train
goto unknown

:install
echo Installing unitree_rl_lab...
git lfs install
pip install -e %UNITREE_RL_LAB_PATH%\source\unitree_rl_lab\
if not exist "%CONDA_PREFIX%\etc\conda\activate.d" mkdir "%CONDA_PREFIX%\etc\conda\activate.d"
(
    echo # for Isaac Lab
    echo set ISAACLAB_PATH=%ISAACLAB_PATH%
    echo # for unitree_rl_lab
    echo set UNITREE_RL_LAB_PATH=%UNITREE_RL_LAB_PATH%
) > "%CONDA_PREFIX%\etc\conda\activate.d\setenv.bat"
activate-global-python-argcomplete
goto end

:list
shift
"%python_exe%" "%UNITREE_RL_LAB_PATH%\scripts\list_envs.py" %*
goto end

:play
shift
"%python_exe%" "%UNITREE_RL_LAB_PATH%\scripts\rsl_rl\play.py" %*
goto end

:train
shift
"%python_exe%" "%UNITREE_RL_LAB_PATH%\scripts\rsl_rl\train.py" --headless %*
goto end

:unknown
echo Unknown option: %1
exit /b 1

:end
exit /b 0
