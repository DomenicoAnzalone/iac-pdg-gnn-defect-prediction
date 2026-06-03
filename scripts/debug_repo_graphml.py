from pathlib import Path
import csv
import networkx as nx

status = Path('output/extraction_combined_status.csv')
if not status.exists():
    raise FileNotFoundError(f'status not found: {status}')

with status.open(newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    failures = [row for row in reader if row['status'] == 'FILE_EXTRACTION_FAILURE']

print('file extraction failures:', len(failures))
for idx, row in enumerate(failures[:3], start=1):
    print('--- sample', idx)
    print('repository:', row['repository'])
    print('commit    :', row['commit'])
    print('filepath  :', row['filepath'])
    print('repo_graphml_path:', row['repo_graphml_path'])
    path = Path(row['repo_graphml_path'])
    print('exists:', path.exists())
    if not path.exists():
        continue
    G = nx.read_graphml(path)
    print('nodes', G.number_of_nodes(), 'edges', G.number_of_edges())
    keys = set()
    sample = []
    for node, data in G.nodes(data=True):
        keys.update(data.keys())
        if len(sample) < 10:
            sample.append((node, data))
    print('node attr keys:', sorted(keys))
    print('sample nodes:')
    for node,data in sample:
        print(' ', node, data)
    print()
