# Ansible-2 Dataset Artifacts

This folder contains the tracked dataset artifacts selected from the second PDG extraction run.

## Contents

- `radon/ansible-2_rows_successful_extracted.csv`: RADON-derived Ansible-2 dataset rows successfully extracted during the run.
- `pdg_extraction/extraction_status.csv`: PDG extraction status, including successfully extracted PDGs and their related paths.
- `pdg_extraction/run_metadata.json`: metadata for the extraction run that produced these artifacts.

The original run outputs are stored under `output/second_pdg_extraction/`, which is ignored by Git. This folder keeps only the curated files that should be versioned.
