"""
中文字体配置模块
自动检测系统并配置合适的中文字体
"""
import platform
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings


def get_system_chinese_font():
    """
    根据操作系统获取推荐的中文字体
    
    Returns:
        str: 字体名称
    """
    system = platform.system()
    
    if system == "Windows":
        # Windows 系统推荐字体（按优先级）
        windows_fonts = [
            'Microsoft YaHei',  # 微软雅黑（最常用）
            'SimHei',           # 黑体
            'SimSun',           # 宋体
            'NSimSun',          # 新宋体
            'FangSong',         # 仿宋
            'KaiTi',            # 楷体
        ]
        return windows_fonts
    
    elif system == "Darwin":  # macOS
        mac_fonts = [
            'PingFang SC',      # 苹方
            'Heiti SC',         # 黑体
            'STHeiti',          # 华文黑体
            'STSong',           # 华文宋体
            'Arial Unicode MS',
        ]
        return mac_fonts
    
    else:  # Linux 和其他系统
        linux_fonts = [
            'Noto Sans CJK SC',
            'Noto Sans CJK JP',
            'WenQuanYi Micro Hei',
            'WenQuanYi Zen Hei',
            'Source Han Sans CN',
            'Droid Sans Fallback',
        ]
        return linux_fonts


def check_font_available(font_name):
    """
    检查字体是否可用
    
    Args:
        font_name: 字体名称
        
    Returns:
        bool: 是否可用
    """
    from matplotlib import font_manager
    
    try:
        available_fonts = [f.name for f in font_manager.fontManager.ttflist]
        return font_name in available_fonts
    except:
        return False


def configure_chinese_font(font_name=None):
    """
    配置matplotlib中文字体
    
    Args:
        font_name: 指定字体名称，None则自动检测
        
    Returns:
        str: 实际配置的字体名称
    """
    if font_name is None:
        # 自动检测可用字体
        candidates = get_system_chinese_font()
        for font in candidates:
            if check_font_available(font):
                font_name = font
                break
    
    # 如果没有找到中文字体，使用默认字体
    if font_name is None:
        font_name = 'DejaVu Sans'
        warnings.warn("未找到中文字体，将使用默认字体，中文可能显示为方框")
    
    # 配置字体
    rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans', 'Arial', 'Helvetica']
    rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    return font_name


def get_font_info():
    """
    获取当前字体配置信息
    
    Returns:
        dict: 字体信息
    """
    return {
        'sans_serif': rcParams['font.sans-serif'],
        'unicode_minus': rcParams['axes.unicode_minus'],
        'system': platform.system(),
    }


# 自动配置（导入时执行）
_configured_font = None

def setup_font(font_name=None):
    """
    设置中文字体（全局入口）
    
    Args:
        font_name: 指定字体名称，None则自动检测
        
    Returns:
        str: 配置的字体名称
    """
    global _configured_font
    _configured_font = configure_chinese_font(font_name)
    return _configured_font


# 模块导入时自动配置
setup_font()


if __name__ == "__main__":
    # 测试字体配置
    print("系统:", platform.system())
    print("候选字体:", get_system_chinese_font())
    print("当前配置:", get_font_info())
    
    # 绘制测试图
    import numpy as np
    
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.linspace(0, 10, 100)
    ax.plot(x, np.sin(x), label='正弦波')
    ax.set_xlabel('X轴 (时间)')
    ax.set_ylabel('Y轴 (幅值)')
    ax.set_title(f'中文显示测试 (字体: {_configured_font})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/font_test.png', dpi=150)
    print("测试图已保存: results/font_test.png")
    plt.close()
