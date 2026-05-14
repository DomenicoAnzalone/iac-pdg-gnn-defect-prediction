import pandas as pd
import csv
import os


def find_index_row(data, metrics : dict):
    data = data[data['repository'] == metrics["gitRepository"]]
    data = data[data['commit'] == metrics["commit"]]
    row = data[data['filepath'] == metrics["filepath"]] 

    if(row.empty):
        print("Row in cui aggiungere pdg metrics non trovata. Non tutti i file dello studio precedente contengono task o vengono estratti dal tool.")
        print("Repository", metrics["gitRepository"], "Commit", metrics["commit"], "File", metrics["filepath"])
        metrics["index_row"] = None
    else:
        indice_riga = int(row.index.values)
        #aumento di 1 l'indice della riga in quanto non è compresa l'intestazione
        indice_riga = indice_riga + 1
        # Resto nel dizionario solo le metriche
        del metrics["gitRepository"]
        del metrics["repository"]
        del metrics["commit"]
        del metrics["filepath"]
        metrics["index_row"] = indice_riga

def add_metrics_to_row(list_dict_metric: list, righe):
    for metrics in list_dict_metric:
        if not(metrics["index_row"] is None):
            indice_riga = metrics["index_row"]
            del metrics["index_row"]
            
            if indice_riga >= len(righe):
                print("La riga specificata non esiste nel file CSV.")
                return
        
            riga_originale = righe[indice_riga]
            intestazioni = righe[0]  # Assume che la prima riga contenga gli intestazioni delle colonne
        
            for colonna_nome, nuovo_valore in metrics.items():
                if colonna_nome not in intestazioni:
                    print(f"La colonna '{colonna_nome}' non esiste nel file CSV.")
                    return
                colonna_indice = intestazioni.index(colonna_nome)
            
                riga_originale[colonna_indice] = nuovo_valore
    
            righe[indice_riga] = riga_originale
    return righe

def save(list_dict_metric: list):
    data = pd.read_csv(os.path.normpath(os.path.join(os.getcwd(), "output", "ansible.csv")))
    
    for metrics in list_dict_metric:
        find_index_row(data = data, metrics = metrics)
        
    # Recupero la riga originale e aggiungo le metriche calcolate nelle apposite colonne
    with open(os.path.normpath(os.path.join(os.getcwd(), "output", "ansible.csv")), 'r') as file_originale:
        reader = csv.reader(file_originale)
        righe = list(reader)
        
        righe = add_metrics_to_row(list_dict_metric=list_dict_metric, righe=righe)
        
    with open(os.path.normpath(os.path.join(os.getcwd(), "output", "ansible.csv")), 'w', newline='\n') as file_modificato:
        writer = csv.writer(file_modificato)
        writer.writerows(righe)
        

def save1(metrics: dict):
    data = pd.read_csv(os.path.normpath(os.path.join(os.getcwd(), "output", "ansible.csv")))
    
    data = data[data['repository'] == metrics["gitRepository"]]
    data = data[data['commit'] == metrics["commit"]]
    row = data[data['filepath'] == metrics["filepath"]] 

    if(row.empty):
        print("Row in cui aggiungere pdg metrics non trovata. Non tutti i file dello studio precedente contengono task o vengono estratti dal tool.")
        print("Repository", metrics["gitRepository"], "Commit", metrics["commit"], "File", metrics["filepath"])

    else:
        indice_riga = int(row.index.values)
        #aumento di 1 l'indice della riga in quanto non è compresa l'intestazione
        indice_riga = indice_riga + 1
        # Resto nel dizionario solo le metriche
        del metrics["gitRepository"]
        del metrics["repository"]
        del metrics["commit"]
        del metrics["filepath"]

        # Recupero la riga originale e aggiungo le metriche calcolate nelle apposite colonne
        with open(os.path.normpath(os.path.join(os.getcwd(), "output", "ansible.csv")), 'r') as file_originale:
            reader = csv.reader(file_originale)
            righe = list(reader)
            
            if indice_riga >= len(righe):
                print("La riga specificata non esiste nel file CSV.")
                return
            
            riga_originale = righe[indice_riga]
            intestazioni = righe[0]  # Assume che la prima riga contenga gli intestazioni delle colonne
            
            for colonna_nome, nuovo_valore in metrics.items():
                if colonna_nome not in intestazioni:
                    print(f"La colonna '{colonna_nome}' non esiste nel file CSV.")
                    return
                
                colonna_indice = intestazioni.index(colonna_nome)
                
                riga_originale[colonna_indice] = nuovo_valore
            
            righe[indice_riga] = riga_originale
        
        with open(os.path.normpath(os.path.join(os.getcwd(), "output", "ansible.csv")), 'w', newline='\n') as file_modificato:
            writer = csv.writer(file_modificato)
            writer.writerows(righe)