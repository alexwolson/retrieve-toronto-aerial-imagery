#!/usr/bin/env python3
"""Setup script for Toronto Aerial Imagery Fetcher."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="toronto-aerial-imagery-fetcher",
    version="1.0.0",
    description="CLI tool to fetch City of Toronto orthorectified aerial imagery from OGC WMTS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Toronto Aerial Imagery Fetcher Contributors",
    license="MIT",
    py_modules=["fetch_imagery"],
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.31.0",
        "pillow>=10.0.0",
        "rasterio>=1.3.0",
        "numpy>=1.24.0",
        "pyproj>=3.6.0",
    ],
    entry_points={
        "console_scripts": [
            "fetch-toronto-imagery=fetch_imagery:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: GIS",
    ],
    keywords="toronto aerial imagery wmts geotiff gis mapping",
)
