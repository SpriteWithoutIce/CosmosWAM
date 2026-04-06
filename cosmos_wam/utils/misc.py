"""Miscellaneous utilities."""
import os


def get_work_dir() -> str:
    """Get the working directory for saving outputs."""
    work_dir = os.environ.get('WORK_DIR', './outputs')
    os.makedirs(work_dir, exist_ok=True)
    return work_dir
