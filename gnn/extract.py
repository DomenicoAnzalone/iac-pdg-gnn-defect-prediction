import os
import logging
import json
from typing import List

logger = logging.getLogger(__name__)

from .paths import raw_pdg_path, manifests_path

try:
    import extractPdgRepositoryLevel as repo_extractor
    import change_commit as change_commit
except Exception:
    repo_extractor = None
    change_commit = None


def extract_for_commit(repository: str, commit: str, output_root: str) -> bool:
    """Checkout commit, run repository-level PDG extractor and save outputs under output_root/raw_pdg"""
    outdir = raw_pdg_path(output_root, repository, commit)

    # record metadata
    manifest_dir = manifests_path(output_root)
    failed_repos = os.path.join(manifest_dir, "failed_repositories.json")
    failed_commits = os.path.join(manifest_dir, "failed_commits.json")

    try:
        # checkout
        if change_commit is None:
            logger.warning("change_commit module not available; skipping checkout")
        else:
            ok = change_commit.checkout_repository(repositoryName=repository, commit=commit)
            if not ok:
                logger.error("Checkout failed for %s @ %s", repository, commit)
                _append_json(failed_commits, {"repository": repository, "commit": commit, "reason": "checkout_failed"})
                return False

        # extract repository-level PDG
        if repo_extractor is None:
            logger.warning("repo extractor not available; skipping PDG generation")
            return False

        ok = repo_extractor.extract_pdg_repository_level(repository=repository)
        # copy result from output/repositories/<repo>/PDG/pdg.dot into outdir/repository_level/pdg.dot
        src = os.path.normpath(os.path.join("output", "repositories", repository, "PDG", "pdg.dot"))
        dest_dir = os.path.join(outdir, "repository_level")
        os.makedirs(dest_dir, exist_ok=True)
        if os.path.exists(src):
            import shutil

            shutil.copy(src, os.path.join(dest_dir, "pdg.dot"))

        if not ok:
            logger.error("PDG extraction returned non-zero for %s@%s", repository, commit)
            _append_json(failed_commits, {"repository": repository, "commit": commit, "reason": "pdg_nonzero"})
            return False

        # move task-level outputs if present
        task_src = os.path.normpath(os.path.join("output", "repositories", repository, "PDG_task_level"))
        if os.path.isdir(task_src):
            import shutil

            task_dest = os.path.join(outdir, "task_level")
            if os.path.isdir(task_dest):
                shutil.rmtree(task_dest)
            shutil.copytree(task_src, task_dest)

        return True
    except Exception as e:
        logger.exception("Exception while extracting PDG for %s@%s: %s", repository, commit, e)
        _append_json(failed_repos, {"repository": repository, "commit": commit, "reason": str(e)})
        return False


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
