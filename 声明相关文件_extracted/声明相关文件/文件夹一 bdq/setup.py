"""
IEEE33微电网电压控制系统 - 安装配置
"""
from setuptools import setup, find_packages

setup(
    name="ieee33-microgrid-control",
    version="1.0.0",
    description="IEEE33节点微电网深度强化学习电压控制系统",
    author="Research Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "pyyaml>=6.0",
        "openpyxl>=3.1.0",
        "gymnasium>=0.29.0",
        "stable-baselines3>=2.2.0",
        "torch>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
)
