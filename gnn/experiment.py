from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json


class ExperimentManager:
    def __init__(self, output_root: Path):
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

    def create_run(self, name: str, config: dict) -> Path:
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        folder = self.output_root / f"{name}_{ts}"
        folder.mkdir(parents=True, exist_ok=False)
        with open(folder / "config.json", "w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2)
        return folder
