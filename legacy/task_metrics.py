import json

""" Input: PDG task level
    Output: Number of non-local variables modified in a task.
"""
def globalOutput(G):
    non_local_var_modified = 0

    nodes = G.nodes
    """ Non local variables """
    try:
        for node in nodes:
            if(G.nodes[node]["node_type"]=="Variable"):
                if(G.nodes[node]["scope_level"] < "16"):
                    if(G.nodes[node]["version"] > "0"):
                        non_local_var_modified+=1
    except Exception as e:
        print("Error: globalOutput.\n ", e.__traceback__)

    return non_local_var_modified


""" Input: PDG task level
    Output: The number of task parameters and non local variables used in a task
"""
def globalInput(G):
    non_local_var = 0
    param = 0
    nodes = G.nodes
    try:
        """ Non local variables """
        for node in nodes:
            if(G.nodes[node]["node_type"]=="Variable"):
                if(G.nodes[node]["scope_level"] < "16"):
                    non_local_var+=1

        """ Parameters """
        u = G.edges(data=True)
        source, target, keys = zip(*u)
        for i in range(0, len(source)):
            if(keys[i]["type"]=="KEYWORD"):
                param+=1
    except Exception as e:
        print("Error: globalInput.\n ", e.__traceback__)
    
    return non_local_var+param


""" Input: PDG task level
    Output: The number of vertices in a task's PDG
"""
def verticesCount(G):
    return len(G.nodes)


""" Input: PDG task level
    Output: The number of edges in a task's PDG
"""
def edgesCount(G):
    return len(G.edges)


""" Input: PDG task level
    Output: Ratio between the number of dependence edges and the number of vertices PDG for a given task's PDG
"""
def edgesToVerticesRatio(G):
    return len(G.edges)/len(G.nodes)


""" Input: PDG task level
    Output: Number of identifiers in task
"""
def numberOfIdentifiers(G):
    identifiers = 0
    nodes = G.nodes
    try:
        for node in nodes:
            if(G.nodes[node]["node_type"]=="Variable"):
               identifiers+=1
    except Exception as e:
        print("Error: numberOfIdentifiers.\n ", e.__traceback__)
    return identifiers


""" DirectFanIn
Input: PDG task level
Output: Sum of the number of tasks in other playbooks that use the output variables directly modified in a task.
"""
def directFanIn(task, playbook):
    """ Cercare le variabili modificate nel task in modo diretto, ovvero che il task non sia invocato da altri task, quindi che abbia includer_location = None"""
    role = task.graph["role_name"]
    role = json.loads(role)
    taskId = role["id"]
    taskLocation = task.nodes[int(taskId)]["location"]
    taskLocation = json.loads(taskLocation)
    if(taskLocation["includer_location"]==None):
        count = 0
        nodes = task.nodes
        playbookTarget = role["path"]
        varSet = set()
        try:
            for node in nodes:
                if(task.nodes[node]["node_type"]=="Variable" and int(task.nodes[node]["version"])>0):
                    varSet.add(node)
            for play in playbook.keys():
                if not(play in playbookTarget):
                    for task in playbook[play]:
                        if(len(set(task.nodes).intersection(varSet))>0):
                            count+=1
            return count
        except Exception as e:
            print("Error: directFanIn.\n ", e.with_traceback)
    return 0
    

""" IndirectFanIn
Input: PDG task level
Output: Sum of the number of tasks in other playbooks that use the output variables indirectly modified in a task.
"""
def indirectFanIn(task, playbook):
    """ Cercare le variabili modificate nel task in modo indiretto, ovvero il task deve essere invocato da altri task, quindi che abbia includer_location != None"""
    role = task.graph["role_name"]
    role = json.loads(role)
    taskId = role["id"]
    taskLocation = task.nodes[int(taskId)]["location"]
    taskLocation = json.loads(taskLocation)
    if(taskLocation["includer_location"]!=None):
        count = 0
        nodes = task.nodes
        playbookTarget = role["path"]
        varSet = set()
        try:
            for node in nodes:
                if(task.nodes[node]["node_type"]=="Variable" and int(task.nodes[node]["version"])>0):
                    varSet.add(node)
            for play in playbook.keys():
                if not(play in playbookTarget):
                    for task in playbook[play]:
                        if(len(set(task.nodes).intersection(varSet))>0):
                            count+=1
            return count
        except Exception as e:
            print("Error: indirectFanIn.\n ", e.with_traceback)
    return 0
   

""" DirectFanOut
Input: PDG task level
Output: Sum of the number of tasks in other playbooks whose output variables are directly modified and used in a task.
"""
def directFanOut(task, playbook):
    u = task.edges(data=True)
    s, t, k = zip(*u)
    role = task.graph["role_name"]
    role = json.loads(role)
    taskId = role["id"]
    taskLocation = task.nodes[int(taskId)]["location"]
    taskLocation = json.loads(taskLocation)
    if(taskLocation["includer_location"]==None):
        count = 0
        nodes = task.nodes
        playbookTarget = role["path"]
        varDict = {}
        try:
            for node in nodes:
                if(task.nodes[node]["node_type"]=="Variable" and int(task.nodes[node]["version"])>0):
                    for i in range(0,len(s)):
                        if(s[i]==node and k[i]["type"]=="USE"):
                            varDict[task.nodes[node]["name"]] = task.nodes[node]["version"]
                           
            for play in playbook.keys():
                if not(play in playbookTarget):
                    for t in playbook[play]:
                        for node in t.nodes:
                            if(t.nodes[node]["node_type"]=="Variable" and (t.nodes[node]["name"] in varDict.keys()) ):
                                if(t.nodes[node]["version"] < varDict[t.nodes[node]["name"]]):
                                    count+=1
            return count
        except Exception as e:
            print("Error: directFanOut.\n ", e.with_traceback)
    return 0


""" IndirectFanOut
Input: PDG task level
Output: Sum of the number of tasks in other playbooks whose output variables are indirectly modified and used in a task.
"""
def indirectFanOut(task, playbook):
    u = task.edges(data=True)
    s, t, k = zip(*u)
    role = task.graph["role_name"]
    role = json.loads(role)
    taskId = role["id"]
    taskLocation = task.nodes[int(taskId)]["location"]
    taskLocation = json.loads(taskLocation)
    if(taskLocation["includer_location"] != None):
        count = 0
        nodes = task.nodes
        playbookTarget = role["path"]
        varDict = {}
        try:
            for node in nodes:
                if(task.nodes[node]["node_type"]=="Variable" and int(task.nodes[node]["version"])>0):
                    for i in range(0,len(s)):
                        if(s[i]==node and k[i]["type"]=="USE"):
                            varDict[task.nodes[node]["name"]] = task.nodes[node]["version"]
                           
            for play in playbook.keys():
                if not(play in playbookTarget):
                    for t in playbook[play]:
                        for node in t.nodes:
                            if(t.nodes[node]["node_type"]=="Variable" and (t.nodes[node]["name"] in varDict.keys()) ):
                                if(t.nodes[node]["version"] < varDict[t.nodes[node]["name"]]):
                                    count+=1
            return count
        except Exception as e:
            print("Error: indirectFanOut.\n ", e.with_traceback)
    return 0