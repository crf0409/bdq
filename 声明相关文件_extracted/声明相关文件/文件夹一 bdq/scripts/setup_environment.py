"""
环境初始化脚本
- 创建必要的目录
- 检测并配置中文字体
- 检查依赖
"""
import os
import sys
import subprocess
import platform
from pathlib import Path

# 修复Windows控制台编码
if sys.platform == 'win32':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except:
        pass


def create_directories():
    """创建项目所需目录"""
    dirs = [
        "results",
        "results/models",
        "results/figures",
        "results/data",
        "logs",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"✓ 目录已创建/存在: {d}")


def check_chinese_fonts():
    """检测系统中的中文字体"""
    import matplotlib.font_manager as fm
    
    # 常见中文字体列表 (按优先级排序)
    chinese_fonts = [
        # Windows 字体
        'Microsoft YaHei', 'SimHei', 'SimSun', 'NSimSun', 'FangSong', 
        'KaiTi', 'SimKai', 'SimFang',
        # macOS 字体
        'PingFang SC', 'Heiti SC', 'STHeiti', 'STSong', 'STKaiti',
        # Linux 字体
        'Noto Sans CJK SC', 'Noto Sans CJK JP', 'WenQuanYi Micro Hei',
        'WenQuanYi Zen Hei', 'Droid Sans Fallback',
        # 其他
        'Source Han Sans CN', 'Source Han Serif CN',
        'DejaVu Sans', 'Arial Unicode MS',
    ]
    
    available_fonts = []
    for font in fm.findSystemFonts():
        try:
            prop = fm.FontProperties(fname=font)
            name = prop.get_name()
            available_fonts.append(name)
        except:
            pass
    
    # 查找第一个可用的中文字体
    for font in chinese_fonts:
        if font in available_fonts:
            print(f"✓ 找到中文字体: {font}")
            return font
    
    print("⚠ 未找到中文字体，将使用默认字体")
    return 'DejaVu Sans'


def configure_matplotlib_font(font_name):
    """配置matplotlib字体"""
    config_dir = Path.home() / '.matplotlib'
    config_dir.mkdir(exist_ok=True)
    
    config_content = f'''# 自动生成的matplotlib配置
font.family: sans-serif
font.sans-serif: {font_name}, DejaVu Sans, Arial, Helvetica
axes.unicode_minus: False
figure.dpi: 100
savefig.dpi: 150
'''
    
    config_file = config_dir / 'matplotlibrc'
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print(f"✓ matplotlib配置已保存: {config_file}")


def check_dependencies():
    """检查关键依赖是否已安装"""
    required = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'matplotlib': 'matplotlib',
        'yaml': 'pyyaml',
        'openpyxl': 'openpyxl',
        'gymnasium': 'gymnasium',
        'stable_baselines3': 'stable-baselines3',
        'torch': 'torch',
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
            print(f"✓ {package} 已安装")
        except ImportError:
            missing.append(package)
            print(f"✗ {package} 未安装")
    
    return missing


def install_dependencies(missing):
    """安装缺失的依赖"""
    if not missing:
        return
    
    print(f"\n正在安装缺失的依赖: {', '.join(missing)}")
    
    # 使用国内镜像加速
    mirrors = [
        'https://pypi.tuna.tsinghua.edu.cn/simple',
        'https://mirrors.aliyun.com/pypi/simple',
        'https://pypi.mirrors.ustc.edu.cn/simple',
    ]
    
    for mirror in mirrors:
        try:
            cmd = [sys.executable, '-m', 'pip', 'install', '--upgrade', 
                   '--index-url', mirror] + missing
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                print(f"✓ 依赖安装成功 (使用镜像: {mirror})")
                return
        except Exception as e:
            continue
    
    # 如果镜像都失败，使用默认源
    print("使用默认PyPI源安装...")
    cmd = [sys.executable, '-m', 'pip', 'install', '--upgrade'] + missing
    subprocess.check_call(cmd)


def print_system_info():
    """打印系统信息"""
    print("=" * 60)
    print("系统信息")
    print("=" * 60)
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"Python版本: {platform.python_version()}")
    print(f"Python路径: {sys.executable}")
    print(f"项目路径: {Path.cwd()}")
    print("=" * 60)


def main():
    print("=" * 60)
    print("IEEE33微电网控制系统 - 环境初始化")
    print("=" * 60)
    
    print_system_info()
    
    # 1. 创建目录
    print("\n[1/4] 创建项目目录...")
    create_directories()
    
    # 2. 检查依赖
    print("\n[2/4] 检查依赖...")
    missing = check_dependencies()
    
    # 3. 安装缺失依赖
    if missing:
        print("\n[3/4] 安装缺失依赖...")
        install_dependencies(missing)
    else:
        print("\n[3/4] 所有依赖已安装，跳过")
    
    # 4. 配置字体
    print("\n[4/4] 配置中文字体...")
    font = check_chinese_fonts()
    configure_matplotlib_font(font)
    
    print("\n" + "=" * 60)
    print("环境初始化完成！")
    print("=" * 60)
    print("\n可用命令:")
    print("  python scripts/train_sac.py        # 训练SAC智能体")
    print("  python scripts/evaluate_sac.py     # 评估模型")
    print("  python scripts/task2_pv_impact.py  # 运行任务2仿真")
    print("=" * 60)


if __name__ == "__main__":
    main()
