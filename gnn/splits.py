import os
import logging
import pandas as pd
from .paths import splits_path

logger = logging.getLogger(__name__)


def walk_forward_splits(df: pd.DataFrame, output_root: str):
    base = splits_path(output_root)
    # group by repository
    for repo, g in df.groupby("repository"):
        repo_dir = os.path.join(base, repo)
        os.makedirs(repo_dir, exist_ok=True)
        g = g.sort_values(["committed_at", "commit"]).reset_index(drop=True)
        n = len(g)
        if n < 2:
            logger.info("Repository %s has too few rows (%d) for walk-forward", repo, n)
            continue
        # for each t from 1..n-1, train=0..t-1, test=t
        for t in range(1, n):
            train = g.iloc[:t]
            test = g.iloc[[t]]
            fold = f"fold_{t:03d}"
            train.to_csv(os.path.join(repo_dir, f"{fold}_train.csv"), index=False)
            test.to_csv(os.path.join(repo_dir, f"{fold}_test.csv"), index=False)
