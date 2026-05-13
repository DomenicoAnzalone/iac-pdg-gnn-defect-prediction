import legacy.task_metrics as mt
import legacy.file_metrics as mf
import dictionary_file_tasknode as df
import parse_pdg as pp
import project_pdg_info as info


def extract_file_metrics_from_repo(repository : str):
    REPO_DICTIONARY = pp.getRepoDictionary()
    list_file_metrics = list()

    dictionary = df.getDict__file_taskPDG(repository)
    for playbook in dictionary.keys():
        setPDG = dictionary[playbook]
        metric = {}
        metric["gitRepository"] = REPO_DICTIONARY[repository]
        metric["repository"] = repository
        metric["filepath"] = playbook
        metric["commit"] = info.getCommit(repository).hexsha

        metric["maxPdgVertices"]   = round(mf.maxPdgVertices(setPDG)   , 2)
        metric["lackOfCohesion"]   = round(mf.lackOfCohesion(setPDG)   , 2)
        for pdg in setPDG:
            verticesCount        = round(mt.verticesCount(pdg), 2)
            edgesCount           = round(mt.edgesCount(pdg), 2)
            edgesToVerticesRatio = round(mt.edgesToVerticesRatio(pdg), 2)
            globalInput          = round(mt.globalInput(pdg), 2)
            globalOutput         = round(mt.globalOutput(pdg), 2)
            directFanIn          = round(mt.directFanIn(pdg,dictionary), 2)
            indirectFanIn        = round(mt.indirectFanIn(pdg,dictionary), 2)
            directFanOut         = round(mt.directFanOut(pdg,dictionary), 2)
            indirectFanOut       = round(mt.indirectFanOut(pdg,dictionary), 2) 
        metric["verticesCount"]        = verticesCount
        metric["edgesCount"]           = edgesCount
        metric["edgesToVerticesRatio"] = edgesToVerticesRatio
        metric["globalInput"]          = globalInput
        metric["globalOutput"]         = globalOutput
        metric["directFanIn"]          = directFanIn
        metric["indirectFanIn"]        = indirectFanIn
        metric["directFanOut"]         = directFanOut
        metric["indirectFanOut"]       = indirectFanOut

        list_file_metrics.append(metric)

    return list_file_metrics
        



