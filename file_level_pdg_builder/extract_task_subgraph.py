# -*- coding: utf-8 -*-

# Considera archi in entrata e uscita a tutte le distanze. A distanza 1 tiene conto dell’eventuale presenza di un nodo Conditional.

def markVerticesofGraphDist3(G, S):
    #declare
    V = G.nodes
    u = G.edges(data=True)
    source, target, keys = zip(*u)
    workList = set()
    markList = set()
    #begin
    workList = S
    #----------DISTANZA = 1-----------
    w = workList.pop()
    markList.add(w)
    for i in range(0, len(source)):
        # verifica se il nodo w ha un arco in entrata. (v,w)
        if target[i] == w:
            if not (markList.__contains__(source[i])):
                # se v non è stato già marcato lo aggiungo alla workList ma solo se non è un control-flow-edge
                if keys[i]["type"] in "ORDER-ORDER_TRANS-ORDER_BACK":
                    if G.nodes[source[i]]["node_type"] == "Conditional":
                        workList.add(source[i])
                    else:
                        markList.add(source[i])
                else:
                    workList.add(source[i])  # altrimenti lo uso per esplorare
        # verifica se il nodo w ha un arco in uscita. (w,v)
        if source[i] == w:
            if not (markList.__contains__(target[i])):
                # se v non è stato già marcato lo aggiungo alla workList ma solo se non è un control-flow-edge
                if keys[i]["type"] in "ORDER-ORDER_TRANS-ORDER_BACK":
                    markList.add(target[i])
                else:
                    workList.add(target[i])  # altrimenti lo uso per esplorare
    # ---------------------------------
    while len(workList) != 0:
        w = workList.pop()
        markList.add(w)

        for i in range(0, len(source)):
            # verifica se il nodo w ha un arco in entrata. (v,w)
            if target[i] == w:
                if not (markList.__contains__(source[i])):
                    # se v non è stato già marcato lo aggiungo alla workList ma solo se non è un control-flow-edge
                    if keys[i]["type"] in "ORDER-ORDER_TRANS-ORDER_BACK":
                        markList.add(source[i])
                    else:
                        workList.add(source[i])
            # verifica se il nodo w ha un arco in uscita. (w,v)
            if source[i] == w:
                if not (markList.__contains__(target[i])):
                    # se v non è stato già marcato lo aggiungo alla workList ma solo se non è un control-flow-edge
                    if keys[i]["type"] in "ORDER-ORDER_TRANS-ORDER_BACK":
                        markList.add(target[i])
                    else:
                        workList.add(target[i])
    return markList


# Considera archi in entrata e uscita solo a distanza 1, poi considera solo archi in entrata. A distanza 1 tiene conto dell’eventuale presenza di un nodo Conditional.
def markVerticesofGraphDist2(G, S):
    #declare
    V = G.nodes
    u = G.edges(data=True)
    source, target, keys = zip(*u)
    workList = set()
    markList = set()
    #begin
    workList = S
    #----------DISTANZA = 1-----------
    w = workList.pop()
    markList.add(w)
    for i in range(0, len(source)):
        # verifica se il nodo w ha un arco in entrata. (v,w)
        if target[i] == w:
            if not (markList.__contains__(source[i])):
                # se v non è stato già marcato lo aggiungo alla workList ma solo se non è un control-flow-edge
                if keys[i]["type"] in "ORDER-ORDER_TRANS-ORDER_BACK":
                    if G.nodes[source[i]]["node_type"] == "Conditional":
                        workList.add(source[i])
                    else:
                        markList.add(source[i])
                else:
                    workList.add(source[i])  # altrimenti lo uso per esplorare
        # verifica se il nodo w ha un arco in uscita. (w,v)
        if source[i] == w:
            if not (markList.__contains__(target[i])):
                # se v non è stato già marcato lo aggiungo alla workList ma solo se non è un control-flow-edge
                if keys[i]["type"] in "ORDER-ORDER_TRANS-ORDER_BACK":
                    markList.add(target[i])
                else:
                    workList.add(target[i])  # altrimenti lo uso per esplorare
    # ---------------------------------
    while len(workList) != 0:
        w = workList.pop()
        markList.add(w)

        for i in range(0, len(source)):
            # verifica se il nodo w ha un arco in entrata. (v,w)
            if target[i] == w:
                if not (markList.__contains__(source[i])):
                    # se v non è stato già marcato lo aggiungo alla workList ma solo se non è un control-flow-edge
                    if keys[i]["type"] in "ORDER-ORDER_TRANS-ORDER_BACK":
                        markList.add(source[i])
                    else:
                        workList.add(source[i])
    return markList


# Considera solo archi in entrata, a distanza 1 tiene conto dell’eventuale presenza di un nodo Conditional.
def markVerticesofGraphDist1(G, S):
    #declare
    V = G.nodes
    u = G.edges(data=True)
    source, target, keys = zip(*u)
    workList = set()
    markList = set()
    #begin
    workList = S
    #----------DISTANZA = 1-----------
    w = workList.pop()
    markList.add(w)
    for i in range(0, len(source)):
        # verifica se il nodo w ha un arco in entrata. (v,w)
        if target[i] == w:
            if not (markList.__contains__(source[i])):
                # se v non è stato già marcato lo aggiungo alla workList ma solo se non è un control-flow-edge
                if keys[i]["type"] in "ORDER-ORDER_TRANS-ORDER_BACK":
                    if G.nodes[source[i]]["node_type"] == "Conditional":
                        workList.add(source[i])
                    else:
                        markList.add(source[i])
                else:
                    workList.add(source[i])  # altrimenti lo uso per esplorare
    # ---------------------------------
    while len(workList) != 0:
        w = workList.pop()
        markList.add(w)

        for i in range(0, len(source)):
            # verifica se il nodo w ha un arco in entrata. (v,w)
            if target[i] == w:
                if not (markList.__contains__(source[i])):
                    # se v non è stato già marcato lo aggiungo alla workList ma solo se non è un control-flow-edge
                    if keys[i]["type"] in "ORDER-ORDER_TRANS-ORDER_BACK":
                        markList.add(source[i])
                    else:
                        workList.add(source[i])
    return markList


def getPDG_tasklevel(G, task_node_id):
    return G.subgraph(markVerticesofGraphDist2(G, {task_node_id}))
