from importlib.resources import path
import os
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

def clean_dataset(dataset_path: str):

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

        cleaned_file_path = "../input/ansible_cleaned.csv"
        df.to_csv(cleaned_file_path, index=False)
        print("Saved:", cleaned_file_path)
    else:
        print("File not found:", dataset_path)
        