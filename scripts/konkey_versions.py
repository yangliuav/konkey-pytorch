import sys
import pathlib
import subprocess
import torch
import pytorch_lightning as pl

def print_versions():
    """CLI function to get info about the konkey and dependency versions."""
    for k, v in konkey_versions().items():
        print(f"{k:20s}{v}")

def konkey_versions():
    return {
        "konkey": konkey_version(),
        "PyTorch": pytorch_version(),
        "PyTorch-Lightning": pytorch_lightning_version(),
    }


def pytorch_version():
    return torch.__version__


def pytorch_lightning_version():
    return pl.__version__


def konkey_version():
    konkey_root = pathlib.Path(__file__).parent.parent.parent
    if konkey_root.joinpath(".git").exists():
        return f"{konkey.__version__}, Git checkout {get_git_version(konkey_root)}"
    else:
        return konkey.__version__


def get_git_version(root):
    def _git(*cmd):
        return subprocess.check_output(["git", *cmd], cwd=root).strip().decode("ascii", "ignore")

    try:
        commit = _git("rev-parse", "HEAD")
        branch = _git("rev-parse", "--symbolic-full-name", "--abbrev-ref", "HEAD")
        dirty = _git("status", "--porcelain")
    except Exception as err:
        print(f"Failed to get Git checkout info: {err}", file=sys.stderr)
        return ""
    s = commit[:12]
    if branch:
        s += f" ({branch})"
    if dirty:
        s += f", dirty tree"
    return s
