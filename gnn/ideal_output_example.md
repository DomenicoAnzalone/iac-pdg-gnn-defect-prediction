# Esempio di output ideale

Questo file mostra un esempio di output generato da `python -m gnn.run_pipeline` usando la pipeline creata.

```
---
repository: ansible-nginx
test_commit: 3a4f2d1c
train_size: 120
validation_size: 30
test_size: 15
balanced_train_size: 120
train_failure_rate: 0.500
validation_failure_rate: 0.467
test_failure_rate: 0.533
---
repository: ansible-postgresql
test_commit: bf12d01e
train_size: 95
validation_size: 24
test_size: 12
balanced_train_size: 96
train_failure_rate: 0.438
validation_failure_rate: 0.417
test_failure_rate: 0.500
```

In un caso reale, il runner visualizzerà una serie di blocchi come questo per ciascun split walk-forward elaborato. Le metriche principali sono:

- `train_size`: numero di esempi usati per il training prima del bilanciamento
- `validation_size`: numero di esempi nel validation set
- `test_size`: numero di esempi nel test set
- `balanced_train_size`: numero finale di esempi nel training set dopo oversampling/undersampling
- `*_failure_rate`: percentuale di grafi con `failure_prone=1` in ciascuna porzione

Questo esempio non rappresenta dati reali del dataset ma illustra il formato dell'output.
