import random
from typing import List, Any


def undersample(items: List[Any], labels: List[int], seed: int = 42):
    random.seed(seed)
    # reduce majority class to minority size
    from collections import defaultdict

    by_label = defaultdict(list)
    for it, lab in zip(items, labels):
        by_label[lab].append(it)

    sizes = {k: len(v) for k, v in by_label.items()}
    if not sizes:
        return items, labels
    min_size = min(sizes.values())
    new_items = []
    new_labels = []
    for k, v in by_label.items():
        sampled = random.sample(v, min(min_size, len(v)))
        new_items.extend(sampled)
        new_labels.extend([k] * len(sampled))

    return new_items, new_labels


def oversample(items: List[Any], labels: List[int], seed: int = 42):
    random.seed(seed)
    from collections import defaultdict

    by_label = defaultdict(list)
    for it, lab in zip(items, labels):
        by_label[lab].append(it)

    sizes = {k: len(v) for k, v in by_label.items()}
    if not sizes:
        return items, labels
    max_size = max(sizes.values())
    new_items = []
    new_labels = []
    for k, v in by_label.items():
        rep = []
        while len(rep) < max_size:
            rep.extend(v)
        rep = rep[:max_size]
        new_items.extend(rep)
        new_labels.extend([k] * len(rep))

    return new_items, new_labels
