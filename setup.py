"""
Setup script for GRANITE framework
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="granite-svi",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@university.edu",
    description="Graph-Refined Accessibility Network for Integrated Transit Equity",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/granite",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "granite=scripts.run_granite:main",
        ],
    },
    include_package_data=True,
    package_data={
        "granite": ["config/*.yaml"],
    },
)