# Infrastructure-as-Code Defect Prediction using Task-Level Program Dependence Graphs and Graph Neural Networks

This repository contains the implementation and experimental pipeline for a research project focused on defect prediction for Infrastructure-as-Code (IaC) systems, with a focus on Ansible repositories.

The project extends previous work based on Program Dependence Graph (PDG) metrics by introducing a graph-based deep learning approach using Graph Neural Networks (GNNs).

Main pipeline folders:

- `radon_dataset_extraction/`: Ansible repository discovery and RADON dataset extraction.
- `pdg_file_level_extraction/`: self-contained batch pipeline for direct file-level PDG extraction with Scansible, temporary repository clones and per-run output folders.
- `gnn/`: preprocessing, training and evaluation of graph-based models.
