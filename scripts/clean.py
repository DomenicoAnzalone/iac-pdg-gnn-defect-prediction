import os
import shutil

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
        