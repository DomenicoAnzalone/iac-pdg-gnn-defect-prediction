import os
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

from .paths import file_level_path, manifests_path

try:
    import dictionary_file_tasknode as dft
    import project_pdg_info as ppi
except Exception:
    dft = None
    ppi = None


def slice_repository_file_level(repository: str, commit: str, output_root: str) -> Dict[str, Any]:
    """Use existing helpers to create file-level PDGs for files present in ansible.csv rows for this repo/commit."""
    outdir = file_level_path(output_root, repository, commit)
    manifest_dir = manifests_path(output_root)
    failed_files = os.path.join(manifest_dir, "failed_files.json")

    results = {"saved": [], "failed": []}

    if dft is None or ppi is None:
        logger.warning("Slicing helpers not available (dictionary_file_tasknode/project_pdg_info)")
        return results

    try:
        # use provided helper to get mapping file -> list of PDGs
        mapping = dft.getDict__file_taskPDG(repository)
        for filepath, graphs in mapping.items():
            safe_name = sanitize(filepath)
            for i, G in enumerate(graphs):
                try:
                    filename = f"{safe_name}_{i}.graphml"
                    path = os.path.join(outdir, filename)
                    import networkx as nx

                    nx.write_graphml(G, path)
                    meta = {
                        "repository": repository,
                        "commit": commit,
                        "filepath": filepath,
                        "filename": filename,
                        "nodes": G.number_of_nodes(),
                        "edges": G.number_of_edges(),
                    }
                    with open(path + ".meta.json", "w", encoding="utf-8") as f:
                        json.dump(meta, f, indent=2)
                    results["saved"].append(meta)
                except Exception as e:
                    logger.exception("Failed to save file-level PDG for %s: %s", filepath, e)
                    results["failed"].append({"filepath": filepath, "reason": str(e)})
                    _append_json(failed_files, {"repository": repository, "commit": commit, "filepath": filepath, "reason": str(e)})

    except Exception:
        logger.exception("Error while slicing repository %s", repository)

    return results


def sanitize(path: str) -> str:
    return (
        str(path)
        .replace("/", "_")
        .replace("\\\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


def _append_json(path: str, obj):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = []
        data.append(obj)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        logger.exception("Failed to append to %s", path)
