import codecs
import os
import re
from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

here = os.path.abspath(os.path.dirname(__file__))

def read(*parts):
    with codecs.open(os.path.join(here, *parts), "r") as fp:
        return fp.read()

def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

setup(
    name="konkey",
    version=find_version("konkey", "__init__.py"),
    author="Yang",
    author_email="yangliuav@gmail.com",
    url="https://github.com/yangliuav",
    description="Pytorch based Echo cancellation toolkit, modified from asteroid",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    python_requires=">=3.6",
    install_requires=[
        # From requirements/torchhub.txt
        "numpy>=1.16.4",
        "scipy>=1.1.0",
        "torch>=1.3.0",
        "SoundFile>=0.10.2",
        "huggingface_hub>=0.0.2",
        # From requirements/install.txt
        "PyYAML>=5.0",
        "pandas>=0.23.4",
        "pytorch-lightning>=1.0.1",
        "torchaudio>=0.5.0",
        "pb_bss_eval>=0.0.2",
        "torch_stoi>=0.1.2",
        "torch_optimizer>=0.0.1a12",
        "julius",
    ],
    entry_points={
        "console_scripts": [
            "konkey-upload=konkey.scripts.konkey_cli:upload",
            "konkey-infer=konkey.scripts.konkey_cli:infer",
            "konkey-register-sr=konkey.scripts.konkey_cli:register_sample_rate",
            "konkey-versions=konkey.scripts.konkey_versions:print_versions",
        ]
    },
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Development Status :: 1 - Alpha",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)