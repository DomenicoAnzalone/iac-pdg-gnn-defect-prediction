from src.pdg.schema import NodeType
from src.pdg.schema import EdgeType

def infer_node_type(attrs):

    shape = attrs.get("shape", "")
    style = attrs.get("style", "")
    label = attrs.get("label", "")

    label = label.replace('"', '')

    if shape == "ellipse":
        return NodeType.TASK

    if shape == "circle":
        return NodeType.INTERMEDIATE

    if "dashed" in style:
        return NodeType.EXPRESSION

    if (
        "str:" in label or
        "bool:" in label or
        "int:" in label or
        "float:" in label
    ):
        return NodeType.LITERAL

    if label in ["list", "dict"]:
        return NodeType.COLLECTION

    if shape == "box" and "dotted" in style:
        return NodeType.VARIABLE

    return NodeType.UNKNOWN

def infer_edge_type(attrs):

    label = attrs.get("label", "")

    label = label.replace('"', '')

    # Core semantic relations

    if label == "DEF":
        return EdgeType.DEF

    if label == "USE":
        return EdgeType.USE

    if label == "ORDER":
        return EdgeType.ORDER

    if label == "WHEN":
        return EdgeType.WHEN

    if label == "LOOP":
        return EdgeType.LOOP

    if label == "NOTIFIES":
        return EdgeType.NOTIFIES

    if label == "KEYWORD":
        return EdgeType.KEYWORD

    # Parameter bindings

    if label.startswith("args."):
        return EdgeType.PARAMETER

    if label.startswith("_"):
        return EdgeType.PARAMETER
    
    if label.startswith("DEFLOOPITEM"):
        return EdgeType.LOOP_SOURCE

    if label == "0":
        return EdgeType.LOOP_SOURCE

    return EdgeType.UNKNOWN