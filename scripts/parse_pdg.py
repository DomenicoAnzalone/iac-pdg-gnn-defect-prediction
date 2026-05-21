import pandas as pd
import re
import os

PRIVATE_REPO = {'openstack-ops', 'ansible-rabbitmq', 'PGBlitz'}

def getRepoDictionary():
    data = pd.read_csv(os.path.normpath(os.path.join(os.getcwd(), "input", "ansible.csv")))
    repositories = data.repository
    repositories = set(repositories)
    githubName_repoName = repositories
    # remove github username from file path
    pattern = re.compile(r'^.*?/')
    for repo in PRIVATE_REPO:
        if repo in githubName_repoName:
            githubName_repoName.remove(repo)
    repoDic = {}
    for string in githubName_repoName:
        repoDic[pattern.sub('', string)] = string
    return repoDic

