from pathlib import Path
import networkx as nx

path = Path('output/pdg_repo_level/naftulikay/ansible-role-degoss/61c9bfaf749c8a4564d527a1eda151dc984f1d5a/PDG_REPO_LEVEL/pdg.graphml')
print('path', path.exists())
G = nx.read_graphml(path)
print('nodes', G.number_of_nodes(), 'edges', G.number_of_edges())

all_keys = set()
for node, data in G.nodes(data=True):
    all_keys.update(data.keys())
print('node attr keys:', sorted(all_keys))

print('\n--- first 20 nodes with attrs ---')
for i, (node, data) in enumerate(G.nodes(data=True)):
    if i >= 20:
        break
    print(node, data)

print('\n--- nodes containing file/task/location-like values ---')
count = 0
for node, data in G.nodes(data=True):
    for key, value in data.items():
        if isinstance(value, str) and any(substr in value.lower() for substr in ['task', 'file', 'location', 'play', 'role', 'path']):
            print(node, key, value)
            count += 1
            break
print('matches:', count)
