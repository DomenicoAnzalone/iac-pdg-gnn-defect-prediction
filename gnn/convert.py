import os
import logging
import json
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

try:
    import networkx as nx
    import torch
    from torch_geometric.utils import from_networkx
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:
    nx = None
    torch = None
    TfidfVectorizer = None

from .paths import manifests_path
from .config import DEFAULT_CONFIG


def _stable_hash_vector(s: str, dim: int):
    # simple deterministic hashing into a small dense vector
    vec = torch.zeros(dim, dtype=torch.float32)
    if s is None:
        return vec
    for i, ch in enumerate(str(s)):
        idx = (ord(ch) + i) % dim
        vec[idx] += 1.0
    # normalize
    if vec.sum() > 0:
        vec = vec / vec.norm(p=2)
    return vec


def _build_tfidf_vocab(graph_list: List) -> Dict[str, int]:
    """Build TF-IDF vocabulary from all node labels in a list of graphs."""
    if TfidfVectorizer is None:
        return {}
    corpus = []
    for G in graph_list:
        labels = []
        for _, attrs in G.nodes(data=True):
            lab = str(attrs.get("label", ""))
            if lab:
                labels.append(lab)
        if labels:
            corpus.append(" ".join(labels))
    if not corpus:
        return {}
    vectorizer = TfidfVectorizer(max_features=32, lowercase=True, token_pattern=r"\w+")
    try:
        vectorizer.fit(corpus)
        vocab = {word: idx for idx, word in enumerate(vectorizer.get_feature_names_out())}
        return vocab
    except Exception:
        logger.exception("Failed to build TF-IDF vocab")
        return {}


def _load_vocab(path: str) -> Dict[str, int]:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        logger.exception("Failed to load node type vocab %s", path)
    return {}


def _save_vocab(path: str, vocab: Dict[str, int]):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, indent=2)
    except Exception:
        logger.exception("Failed to save node type vocab %s", path)


def _save_feature_schema(output_root: str, schema: Dict[str, Any]):
    """Save feature schema metadata to manifests."""
    manifest_dir = manifests_path(output_root)
    schema_path = os.path.join(manifest_dir, "feature_schema.json")
    try:
        with open(schema_path, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2)
        logger.info("Saved feature schema to %s", schema_path)
    except Exception:
        logger.exception("Failed to save feature schema")


def graphml_to_pyg(graphml_path: str, out_path: str) -> Optional[str]:
    """Convert a GraphML file to a PyG Data object with engineered node features.

    Features: [degree_norm (1), node_type_onehot (len(vocab)), label_hash(dim=config)]
    A global node_type vocab is maintained under `manifests`.
    """
    if nx is None or torch is None:
        logger.warning("networkx or torch not available: skipping conversion %s", graphml_path)
        return None

    try:
        G = nx.read_graphml(graphml_path)

        # load or init vocab
        manifest_dir = manifests_path(DEFAULT_CONFIG.output_root)
        vocab_path = DEFAULT_CONFIG.node_type_vocab_file
        vocab = _load_vocab(vocab_path)

        # collect node types present in this graph
        node_types = set()
        for _, attrs in G.nodes(data=True):
            t = attrs.get("node_type", "")
            node_types.add(str(t))

        # extend vocab
        changed = False
        for t in sorted(node_types):
            if t not in vocab:
                vocab[t] = len(vocab)
                changed = True
        if changed:
            _save_vocab(vocab_path, vocab)

        dim_hash = DEFAULT_CONFIG.node_label_hash_dim

        # build node feature matrix
        n = G.number_of_nodes()
        feat_dim = 1 + max(1, len(vocab)) + dim_hash
        x = torch.zeros((n, feat_dim), dtype=torch.float32)

        node_list = list(G.nodes())
        node_index = {node_list[i]: i for i in range(len(node_list))}

        for node, attrs in G.nodes(data=True):
            i = node_index[node]
            # degree (normalized by max degree)
            deg = float(G.degree(node))
            x[i, 0] = deg

            # node type one-hot
            t = str(attrs.get("node_type", ""))
            idx = vocab.get(t, None)
            if idx is not None and idx < len(vocab):
                # place 1.0 at position 1 + idx (and cap to vocab length)
                pos = 1 + (idx % max(1, len(vocab)))
                x[i, pos] = 1.0

            # label/text hashing
            lab = attrs.get("label", "")
            vec = _stable_hash_vector(lab, dim_hash)
            x[i, 1 + max(1, len(vocab)) : 1 + max(1, len(vocab)) + dim_hash] = vec

        # normalize degree column
        if x[:, 0].max() > 0:
            x[:, 0] = x[:, 0] / (x[:, 0].max())

        # convert using from_networkx and replace x
        data = from_networkx(G)
        data.x = x

        # if label exists as graph attr, map to data.y
        graph_label = None
        if "label" in G.graph:
            try:
                graph_label = float(G.graph.get("label"))
            except Exception:
                graph_label = None
        if graph_label is not None:
            data.y = torch.tensor([graph_label], dtype=torch.float32)

        # save schema on first conversion
        schema = {
            "node_features": {
                "degree_norm": {"dim": 1, "description": "Normalized node degree"},
                "node_type_onehot": {"dim": len(vocab), "vocab": vocab},
                "label_hash": {"dim": dim_hash, "description": "Hash of node label"},
            },
            "total_dim": feat_dim,
            "timestamp": str(__import__("datetime").datetime.now()),
        }
        _save_feature_schema(DEFAULT_CONFIG.output_root, schema)

        torch.save(data, out_path)
        return out_path
    except Exception:
        logger.exception("Failed to convert %s to pyg", graphml_path)
        return None
