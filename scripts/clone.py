import os
import git
import parse_pdg as p
import traceback

def clone_repositories():
    dizionario = p.getRepoDictionary()
    print(dizionario)
    GITHUB_PREFIX = "https://github.com/"
    try:
        for nome_repository, username_github in dizionario.items():
            print("Cloning...",nome_repository)
            if os.path.exists(os.path.normpath(os.path.join(os.getcwd(), "input", "repositories", nome_repository))):
                print(f"{nome_repository} already exists, skipping...")
                continue
            git.Repo.clone_from(GITHUB_PREFIX+username_github+".git", os.path.normpath(os.path.join(os.getcwd(), "input", "repositories", nome_repository)))
    except:
        traceback.print_exc()

clone_repositories()