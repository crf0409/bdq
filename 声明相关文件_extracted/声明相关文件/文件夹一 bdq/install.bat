@echo off
chcp 65001 >nul
echo ============================================
echo IEEE33微电网控制系统 - 环境安装脚本
echo ============================================
echo.

:: 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到Python，请先安装Python 3.9或更高版本
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1/3] Python版本:
python --version
echo.

:: 升级pip
echo [2/3] 升级pip...
python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 (
    echo 使用默认源升级pip...
    python -m pip install --upgrade pip
)

:: 安装依赖
echo.
echo [3/3] 安装项目依赖...
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 (
    echo 使用默认PyPI源安装...
    pip install -r requirements.txt
)

:: 运行环境初始化
echo.
echo [4/4] 初始化环境...
python scripts/setup_environment.py

echo.
echo ============================================
echo 安装完成！
echo ============================================
echo.
echo 可用命令:
echo   python scripts/train_sac.py        - 训练SAC智能体
echo   python scripts/evaluate_sac.py     - 评估模型
echo   python scripts/task2_pv_impact.py  - 光伏影响分析
echo.
pause
