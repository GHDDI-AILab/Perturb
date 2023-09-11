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
│       └── finetune_gene_perturb.py
└── perturb
    ├── __init__.py
    ├── data
    │   ├── __init__.py
    │   └── data.py
    ├── loss
    │   ├── __init__.py
    │   └── masked_loss.py
    ├── metrics
    │   ├── __init__.py
    │   └── evaluation.py
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
        ├── preprocess.py
        └── utils.py

```
