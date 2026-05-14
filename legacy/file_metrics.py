import legacy.task_metrics as tm
import json
import os
import yaml
import re

""" 
Input: all PDGs task level of a playbook
Output: The number of shared vertices between playbook's tasks
"""
def lackOfCohesion(PDG_list):
    if(len(PDG_list)==1):
        return 0;
    pdg = PDG_list[0]
    setNode = set(pdg.nodes)
    for i in range(1,len(PDG_list)):
        setNode = setNode.intersection(PDG_list[i].nodes)
    return len(setNode)

""" 
Input: all PDGs task level of a playbook
Output: Number of tasks a playbook contains
"""
def count_ansible_tasks(file_path):
    with open(file_path, 'r') as file:
        ansible_content = yaml.safe_load(file)
        if not isinstance(ansible_content, list):
            return 0

        task_count = 0
        for item in ansible_content:
            if isinstance(item, dict) and 'name' in item:
                task_count += 1
        return task_count
    
def taskCount(PDG_list):
    pdg = PDG_list[0]
    role = pdg.graph["role_name"]
    role = json.loads(role)
    filePath = role["path"]
    path = os.path.normpath(os.path.join(os.getcwd(), "input", "repositories", filePath)) 
    return count_ansible_tasks(path)

""" 
Input: all PDGs task level of a playbook
Output:  Average number of distinct occurrences of programmer-defined labels within a task
"""
def taskIdentifier(PDG_list):
    k = taskCount(PDG_list)
    taskId = 0
    if(k == 0):
        return 0
    for i in range(0,len(PDG_list)):
        taskId = taskId + tm.numberOfIdentifiers(PDG_list[i])/k
    return taskId

""" 
Input: all PDGs task level of a playbook
Output:  Number of vertices of the task's PDGs with the maximum number of vertices in all task's PDGs of a playbook.
"""
def maxPdgVertices(PDG_list):
    max = 0
    for i in range(0,len(PDG_list)):
        if(len(PDG_list[i].nodes) > max):
            max = len(PDG_list[i].nodes)
    return max

""" 
Input: all PDGs task level of a playbook
Output: Average number of lines of code (LOC) in a playbook's tasks
"""
def taskSize(PDG_list):
    pdg = PDG_list[0]
    role = pdg.graph["role_name"]
    role = json.loads(role)
    filePath = role["path"]
    path = os.path.normpath(os.path.join(os.getcwd(), "input", "repositories", filePath)) 

    k = taskCount(PDG_list)
    if(k == 0):
        return 0
    """ esclude le linee di codice vuote. Esempio newline """
    SI = count_loc_ansible_playbook(path) 
    return SI/k

def count_loc_ansible_playbook(playbook_path):
    try:
        with open(playbook_path, 'r') as file:
            loc_count = sum(1 for line in file if line.strip() and not line.strip().startswith('#'))
        return loc_count
    except FileNotFoundError:
        return 0
    
""" 
Input: all PDGs task level of a playbook
Output: Average ration between the task's sizes and the playbook's LOC
"""
def taskCoverage(PDG_list):
    pdg = PDG_list[0]
    role = pdg.graph["role_name"]
    role = json.loads(role)
    filePath = role["path"]
    path = os.path.normpath(os.path.join(os.getcwd(), "input", "repositories", filePath)) 

    taskSizes = taskSize(PDG_list)
    playbookLOC = count_loc_ansible_playbook(path)
    return taskSizes/playbookLOC