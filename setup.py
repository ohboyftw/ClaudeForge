#!/usr/bin/env python3
"""
Setup script for Claude Configuration Manager
"""

from setuptools import setup, find_packages
import os

# Read README content
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="claude-config-manager",
    version="1.0.0",
    author="Claude Configuration Manager Team",
    author_email="dev@example.com",
    description="AI-powered Claude configuration management with DSPy + Ollama orchestration",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/example/claude-config-manager",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "pytest-mock>=3.12.0",
            "black>=23.11.0",
            "flake8>=6.1.0",
            "mypy>=1.7.1",
        ],
        "web": [
            "fastapi>=0.104.1",
            "uvicorn>=0.24.0",
            "jinja2>=3.1.2",
        ],
        "monitoring": [
            "prometheus-client>=0.19.0",
            "grafana-client>=3.5.0",
        ],
        "ml": [
            "numpy>=1.24.4",
            "pandas>=2.1.3",
            "scikit-learn>=1.3.2",
        ]
    },
    entry_points={
        "console_scripts": [
            "claude-config=claude_config_manager:main",
            "ccm=claude_config_manager:main",
        ],
    },
    include_package_data=True,
    package_data={
        "claude_config_manager": [
            "templates/*.md",
            "templates/*.json",
            "config/*.yaml",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/example/claude-config-manager/issues",
        "Source": "https://github.com/example/claude-config-manager",
        "Documentation": "https://claude-config-manager.readthedocs.io/",
    },
)