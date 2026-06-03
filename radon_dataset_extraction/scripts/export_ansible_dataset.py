import argparse
import os
import shutil
import tempfile
import traceback
from pathlib import Path

import nltk

from repominer.mining.ansible import AnsibleMiner
from repominer.metrics.ansible import AnsibleMetricsExtractor


PROJECT_ROOT = Path(__file__).resolve().parents[1]


nltk.download("punkt", quiet=False)
nltk.download("punkt_tab", quiet=False)
nltk.download("stopwords", quiet=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-url", required=True)
    parser.add_argument("--default-branch", default="master")
    parser.add_argument(
        "--clone-dir",
        default=str(Path(tempfile.gettempdir()) / "radon_dataset_repo"),
    )
    parser.add_argument("--output", default=str(PROJECT_ROOT / "output" / "dataset.csv"))
    parser.add_argument("--product", action="store_true")
    parser.add_argument("--process", action="store_true")
    parser.add_argument("--delta", action="store_true")
    args = parser.parse_args()

    clone_dir = args.clone_dir
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if os.path.exists(clone_dir):
    	shutil.rmtree(clone_dir)

    os.makedirs(clone_dir, exist_ok=True)

    try:
        print(f"[1/6] Repo URL: {args.repo_url}")
        print(f"[2/6] Branch: {args.default_branch}")
        print(f"[3/6] Clone dir: {clone_dir}")

        print("[4/6] Creating AnsibleMiner...")
        miner = AnsibleMiner(args.repo_url, args.default_branch, clone_dir)

        print("[5/6] Mining fixing commits...")
        fixing_commits = miner.get_fixing_commits()
        print(f"Fixing commits found: {len(fixing_commits)}")

        print("Mining fixed files / bug-inducing commits...")
        fixed_files = miner.get_fixed_files()
        print(f"Fixed files found: {len(fixed_files)}")

        print("Labeling file snapshots...")
        labeled_files = list(miner.label())
        print(f"Labeled file snapshots: {len(labeled_files)}")

        if len(labeled_files) == 0:
            print("No labeled files found. No CSV will be generated for this repository.")
            return

        product = args.product
        process = args.process
        delta = args.delta

        if not any([product, process, delta]):
            product = True
            process = True
            delta = True

        print("[6/6] Extracting metrics...")
        print(f"Metrics enabled: product={product}, process={process}, delta={delta}")

        metrics_extractor = AnsibleMetricsExtractor(args.repo_url, "release", clone_dir)

        metrics_extractor.extract(
            labeled_files,
            product=product,
            process=process,
            delta=delta
        )

        dataset = metrics_extractor.dataset

        print("Dataset shape:", dataset.shape)
        print("Dataset columns:")
        print(list(dataset.columns))

        for candidate_label in ["failure_prone", "defective", "label", "buggy"]:
            if candidate_label in dataset.columns:
                print(f"Label distribution for {candidate_label}:")
                print(dataset[candidate_label].value_counts(dropna=False))

        dataset.to_csv(output_path, index=False)
        print(f"Saved CSV to: {output_path}")

    except Exception:
        print("ERROR during dataset export")
        traceback.print_exc()
        raise

    finally:
        if os.path.exists(clone_dir):
            shutil.rmtree(clone_dir)


if __name__ == "__main__":
    main()
