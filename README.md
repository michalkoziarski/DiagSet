# Python container for DiagSet-A dataset 

Python container for DiagSet-A, a dataset for prostate cancer histopathological image classification available at https://ai-econsilio.diag.pl/. More detailed description of the dataset can be found at https://arxiv.org/abs/2105.04014.

## Sample usage

To create train and test containers, and sample batches for one epoch:

```python
from container import TrainingDiagSetDataset, EvaluationDiagSetDataset

train_set = TrainingDiagSetDataset(
    root_path='./DiagSet-A',
    partitions=['train', 'validation'],
    magnification=40
)
test_set = EvaluationDiagSetDataset(
    root_path='./DiagSet-A',
    partitions=['test'],
    magnification=40
)

for _ in range(train_set.length()):
    images, labels = train_set.batch()
```

To create a container for binary classification:

```python
train_set = TrainingDiagSetDataset(
    root_path='./DiagSet-A',
    partitions=['train', 'validation'],
    magnification=40,
    label_dictionary={'BG': 0, 'T': 0, 'N': 0, 'A': 0, 'R1': 1, 'R2': 1, 'R3': 1, 'R4': 1, 'R5': 1}
)
```

To create a container that will sample images from both classes with equal probability:

```python
train_set = TrainingDiagSetDataset(
    root_path='./DiagSet-A',
    partitions=['train', 'validation'],
    magnification=40,
    label_dictionary={'BG': 0, 'T': 0, 'N': 0, 'A': 0, 'R1': 1, 'R2': 1, 'R3': 1, 'R4': 1, 'R5': 1},
    class_ratios={0: 0.5, 1: 0.5}
)
```
