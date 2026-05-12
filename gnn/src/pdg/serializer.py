import json
from pathlib import Path


class GraphSerializer:

    def __init__(self, output_dir):

        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def save(self, graph_data, filename):

        output_path = self.output_dir / filename

        with open(output_path, "w") as f:

            json.dump(
                graph_data,
                f,
                indent=4
            )

        print(
            f"\nGraph saved to: {output_path}"
        )