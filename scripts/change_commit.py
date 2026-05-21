import git
import project_pdg_info as info
import pandas as pd
import parse_pdg as pp
import os
import traceback

def checkout_repository(repositoryName : str, commit : str):
    repository = git.Repo(os.path.normpath(os.path.join(os.getcwd(), "input", "repositories", repositoryName)))
    try:
        commit = repository.commit(commit)
        repository.git.execute(['git', 'checkout', '--', '.'])
        repository.git.checkout(commit)         
        return True
    except:
        traceback.print_exc()
        return False