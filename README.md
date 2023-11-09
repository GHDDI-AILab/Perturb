# Perturb
Predicting cellular responses to gene and compound perturbations.

## Directory Structure

```
Perturb/
├── README.md
├── examples
│   └── pert
│       ├── atom_dict.gpkl
│       ├── conf.drug.toml
│       ├── conf.gene.toml
│       ├── finetune_drug_perturb.py
│       ├── finetune_gene_perturb.py
│       └── finetune_drug_ad_siamese_perturb.py
└── perturb
    ├── __init__.py
    ├── data
    │   ├── __init__.py
    │   ├── data.py
    │   └── preprocess.py
    ├── loss
    │   ├── __init__.py
    │   └── masked_loss.py
    ├── metrics
    │   ├── __init__.py
    │   ├── evaluation.py
    │   └── plotting.py
    ├── model
    │   ├── __init__.py
    │   ├── compound_model.py
    │   ├── compound_processor.py
    │   ├── dsbn.py
    │   ├── generation_model.py
    │   ├── grad_reverse.py
    │   └── model.py
    ├── tokenizer
    │   ├── __init__.py
    │   └── tokenizer.py
    ├── train
    │   ├── __init__.py
    │   └── trainer.py
    └── utils
        ├── __init__.py
        ├── attr_dict.py
        └── utils.py

```

## Basic Usage

Download the repo:
```bash
$ git clone https://github.com/GHDDI-AILab/Perturb.git
$ cd Perturb/
```

Import the package in Python:
```python
from perturb.data import PertData
from perturb.model import TransformerGenerator
from perturb.train import Trainer
from perturb.utils import read_pickle, write_pickle
from perturb.utils.attr_dict import AttrDict
```

## Examples

```bash
$ cd examples/pert/
$ python finetune_drug_perturb.py -c conf.drug.toml 1> run.log 2> run.err
```
