import legacy.task_metrics as mt
import legacy.file_metrics as mf
import writer_reader as wr
import networkx as nx
import get_file_path as fp
import dictionary_file_tasknode as df
import project_pdg_info as pi
import parse_pdg as pp
import extract_task_subgraph as es
import csv
import json
import os

OUTPUT_DIR = ".\\output\\"
REPO_DICTIONARY = pp.getRepoDictionary()
rows = []

#nome del file CSV
nome_file_csv = "task_metrics.csv"
csv_file_path = OUTPUT_DIR+nome_file_csv

# Apri il file CSV in modalità scrittura
with open(csv_file_path, mode="w", newline="") as file_csv:
    # Definisci gli header delle colonne
    intestazioni = ["gitRepository", "repository", "file", "taskId", "verticesCount", "edgesCount", "edgesToVerticesRatio", "globalInput", "globalOutput", "directFanIn", "indirectFanIn", "directFanOut", "indirectFanOut"]
    writer = csv.DictWriter(file_csv, fieldnames=intestazioni)
    writer.writeheader()

    _, GOOD_REPOS_NAME  = pi.repo_with_graph("graphml")
    for i in range(0,len(GOOD_REPOS_NAME)):
        print("\n")
        print("Repository:",GOOD_REPOS_NAME[i])
        dictionary = df.getDict__file_taskPDG(GOOD_REPOS_NAME[i])
        for playbook in dictionary.keys():
            setPDG = dictionary[playbook]
            for pdg in setPDG:
                role = json.loads(pdg.graph["role_name"])
                row = {}
                row["gitRepository"] = REPO_DICTIONARY[GOOD_REPOS_NAME[i]]
                row["repository"] = GOOD_REPOS_NAME[i]
                row["file"] = playbook
                row["taskId"] = role["id"]

                row["verticesCount"]        = round(mt.verticesCount(pdg), 2)
                row["edgesCount"]           = round(mt.edgesCount(pdg), 2)
                row["edgesToVerticesRatio"] = round(mt.edgesToVerticesRatio(pdg), 2)
                row["globalInput"]          = round(mt.globalInput(pdg), 2)
                row["globalOutput"]         = round(mt.globalOutput(pdg), 2)
                row["directFanIn"]          = round(mt.directFanIn(pdg,dictionary), 2)
                row["indirectFanIn"]        = round(mt.indirectFanIn(pdg,dictionary), 2)
                row["directFanOut"]         = round(mt.directFanOut(pdg,dictionary), 2)
                row["indirectFanOut"]       = round(mt.indirectFanOut(pdg,dictionary), 2)
                print(row)
                writer.writerow(row)





