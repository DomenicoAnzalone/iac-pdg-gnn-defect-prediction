import os
import subprocess
import traceback

def extract_pdg_repository_level(repository: str):

    repositories_path = os.path.normpath(os.path.join("input", "repositories", repository))
    output_path = os.path.normpath(os.path.join("output", "repositories", repository, "PDG"))
    os.makedirs(output_path, exist_ok=True)

    try:

        command = (f"scansible build-pdg -f graphviz {repositories_path}")

        output_file = os.path.join(output_path, "pdg.dot")

        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True
        )

        print("STDERR:")
        print(result.stderr)

        print("RETURN CODE:")
        print(result.returncode)

        with open(output_file, "w") as file:
            file.write(result.stdout)

        return result.returncode == 0

    except:
        traceback.print_exc()
        return False