import os
from typing import Tuple


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def raw_pdg_path(output_root: str, repository: str, commit: str) -> str:
    p = os.path.normpath(os.path.join(output_root, "raw_pdg", repository, commit))
    ensure_dir(p)
    return p


def file_level_path(output_root: str, repository: str, commit: str) -> str:
    p = os.path.normpath(os.path.join(output_root, "file_level", repository, commit))
    ensure_dir(p)
    return p


def manifests_path(output_root: str) -> str:
    p = os.path.normpath(os.path.join(output_root, "manifests"))
    ensure_dir(p)
    return p


def splits_path(output_root: str) -> str:
    p = os.path.normpath(os.path.join(output_root, "splits", "walk_forward"))
    ensure_dir(p)
    return p


def models_path(output_root: str) -> str:
    p = os.path.normpath(os.path.join(output_root, "models"))
    ensure_dir(p)
    return p


def reports_path(output_root: str) -> str:
    p = os.path.normpath(os.path.join(output_root, "reports"))
    ensure_dir(p)
    return p
