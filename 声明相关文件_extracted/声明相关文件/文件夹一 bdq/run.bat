@echo off
chcp 65001 >nul
echo ============================================
echo IEEE33微电网控制系统 - 运行菜单
echo ============================================
echo.
echo 请选择要运行的任务:
echo.
echo [1] 任务2: 光伏影响分析 (60秒仿真)
echo [2] 任务3: 下垂控制仿真
echo [3] 任务4: 训练SAC智能体
echo [4] 评估SAC模型
echo [5] 运行全部任务
echo [6] 环境检查
echo [0] 退出
echo.
set /p choice="请输入选项 [0-6]: "

if "%choice%"=="1" goto task2
if "%choice%"=="2" goto task3
if "%choice%"=="3" goto task4
if "%choice%"=="4" goto evaluate
if "%choice%"=="5" goto all
if "%choice%"=="6" goto check
if "%choice%"=="0" goto end

echo 无效选项！
pause
goto end

:task2
echo.
echo 运行任务2: 光伏影响分析...
python scripts/task2_pv_impact.py
pause
goto end

:task3
echo.
echo 运行任务3: 下垂控制仿真...
python scripts/task3_droop_control.py
pause
goto end

:task4
echo.
echo 运行任务4: 训练SAC智能体...
python scripts/train_sac.py
pause
goto end

:evaluate
echo.
echo 评估SAC模型...
python scripts/evaluate_sac.py
pause
goto end

:all
echo.
echo 运行所有任务...
echo.
echo [1/4] 光伏影响分析...
python scripts/task2_pv_impact.py
echo.
echo [2/4] 下垂控制仿真...
python scripts/task3_droop_control.py
echo.
echo [3/4] 训练SAC智能体...
python scripts/train_sac.py
echo.
echo [4/4] 评估模型...
python scripts/evaluate_sac.py
echo.
echo 所有任务完成！
pause
goto end

:check
echo.
echo 环境检查...
python scripts/setup_environment.py
pause
goto end

:end
