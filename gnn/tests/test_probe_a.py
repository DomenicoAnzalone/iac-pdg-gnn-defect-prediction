#!/usr/bin/env python3
"""Probe A: ingest and normalize CSV"""
from gnn.ingest import normalize_and_index

df = normalize_and_index('input/ansible.csv')
print(f'Ingested rows: {len(df)}')
print(f'Columns: {list(df.columns)}')
print(f'Label distribution:')
print(df['label'].value_counts())
print(f'\nSample rows:')
print(df.head(2)[['repository', 'commit', 'filepath', 'label', 'committed_at']])
print('\n✓ Probe A passed')
