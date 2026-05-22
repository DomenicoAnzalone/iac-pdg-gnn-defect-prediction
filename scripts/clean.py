from importlib.resources import path
import os
from pathlib import Path
import shutil
import pandas as pd

def clean_output_dir():
    directory = os.path.normpath(os.path.join(os.getcwd(), "output", "repositories"))

    # Rimuovi tutto il contenuto della directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            shutil.rmtree(path = dir_path)

def clean_repository(repository : str):
    path = os.path.normpath(os.path.join(os.getcwd(), "input", "repositories", repository, "PDG"))
    if(os.path.exists(path = path)):
        shutil.rmtree(path = path)

def clean_metrics_cols(dataset_path: str):

    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
        df = df[
            [
                "commit",
                "repository",
                "committed_at",
                "filepath",
                "failure_prone"
            ]
        ]

        cleaned_file_path = "../input/ansible_core_features.csv"
        df.to_csv(cleaned_file_path, index=False)
        print("Saved:", cleaned_file_path)
    else:
        print("File not found:", dataset_path)

def clean_rows_without_pdg_metrics(dataset_path: str):
    
    input_csv = Path(dataset_path)

    output_csv = input_csv.parent / "ansible_with_pdg_metrics.csv"

    metric_columns = [
        "maxPdgVertices",
        "verticesCount",
        "edgesToVerticesRatio",
        "edgesCount",
        "globalInput",
        "lackOfCohesion",
        "indirectFanOut",
        "indirectFanIn",
        "directFanOut",
        "directFanIn",
        "globalOutput"
    ]

    df = pd.read_csv(input_csv)
    print(f"Righe originali: {len(df)}")

    df_cleaned = df.dropna(subset=metric_columns)
    print(f"Righe dopo pulizia: {len(df_cleaned)}")
    print(f"Righe rimosse: {len(df) - len(df_cleaned)}")

    df_cleaned.to_csv(output_csv, index=False)

    print(f"CSV pulito salvato in: {output_csv}")