import re
from pathlib import Path
from setuptools import setup, find_packages

# Single source of the version: src/gadget_analysis/__init__.py
_init = (Path(__file__).parent / 'src' / 'gadget_analysis' / '__init__.py').read_text()
VERSION = re.search(r"^__version__\s*=\s*'([^']+)'", _init, re.M).group(1)

setup(
    name="gadget_analysis",
    version=VERSION,
    package_dir={"": "src"},
    packages=find_packages("src"),
    # data files the modules read from their own folders (processing configs, channel maps, pad geometry,
    # SRIM tables, GUI artwork, ...); only matters for a non-editable install
    package_data={"gadget_analysis": ["**/*.csv", "**/*.txt", "**/*.npy", "**/*.PNG", "**/*.xbm", "**/*.pdf",
                                      "**/*.gui_ini", "**/*.ini", "**/*.json"]},
    python_requires=">=3.11",
)
