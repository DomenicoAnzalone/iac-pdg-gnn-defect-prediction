import json
import os
from pathlib import PurePosixPath


def normalize_location_file(location_file, repo_root=None):
    if location_file is None:
        return None
    if isinstance(location_file, bytes):
        location_file = location_file.decode("utf-8", errors="ignore")
    location_file = str(location_file).replace("\\", "/")
    if repo_root is not None:
        repo_root_norm = os.path.abspath(repo_root).replace("\\", "/")
        if location_file.startswith(repo_root_norm):
            location_file = location_file[len(repo_root_norm) :].lstrip("/")
    location_file = os.path.normpath(location_file).replace("\\", "/")
    return location_file


def parse_location(location_value):
    if location_value is None:
        return None
    if isinstance(location_value, str):
        try:
            location_value = json.loads(location_value)
        except (ValueError, TypeError):
            pass
    if isinstance(location_value, dict):
        return location_value.get("file")
    return None


def path_matches_file(location_file, target_file):
    if location_file is None or target_file is None:
        return False

    location_file_norm = normalize_location_file(location_file)
    target_file_norm = normalize_location_file(target_file)
    if location_file_norm == target_file_norm:
        return True

    location_parts = PurePosixPath(location_file_norm).parts
    target_parts = PurePosixPath(target_file_norm).parts
    if len(target_parts) == 0:
        return False
    if len(location_parts) >= len(target_parts) and tuple(location_parts[-len(target_parts) :]) == tuple(target_parts):
        return True
    return False


def get_task_nodes_for_file(G, target_file, repo_root=None):
    task_nodes = []
    for node, data in G.nodes(data=True):
        if str(data.get("node_type", "")) != "Task":
            continue
        location_value = data.get("location")
        location_file = parse_location(location_value)
        if location_file is None:
            continue
        if path_matches_file(location_file, target_file):
            task_nodes.append(node)
        else:
            location_file_norm = normalize_location_file(location_file, repo_root)
            if path_matches_file(location_file_norm, target_file):
                task_nodes.append(node)
    return task_nodes
