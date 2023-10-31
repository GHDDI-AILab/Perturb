# %%
import copy
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import List, Tuple, Dict, Union, Optional
import warnings
import pandas as pd 

import torch
from anndata import AnnData
import scanpy as sc
import scvi
import numpy as np
# Set the environment variable
os.environ["WANDB_MODE"] = "dryrun"
import wandb
import umap
from scipy.sparse import issparse
import scipy.stats
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.decomposition import PCA
from sklearn.metrics import jaccard_score
from sklearn.manifold import TSNE
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)

from scgpt.tokenizer.gene_tokenizer import GeneVocab

sys.path.insert(0, '../..')
import scgpt as scg
from perturb.model.compound_model import Transformer4Cmpd, AdversarialDiscriminator, SiameseNetwork
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from perturb.loss.masked_loss import (
    masked_mse_loss,
    weighted_mse_loss,
    contrastive_loss_multilayer,
)
from scgpt.preprocess import Preprocessor
from scgpt import SubsetsBatchSampler
from scgpt.utils import set_seed, category_str2int, eval_scib_metrics

sc.set_figure_params(figsize=(4, 4))
# os.environ["KMP_WARNINGS"] = "off"
os.environ["WANDB_MODE"] = "offline"

hyperparameter_defaults = dict(
    seed=42,
    dataset_name="K562_compoud21",
    do_train=True,
    load_model="save/scGPT_bc",
    mask_ratio=0.0,
    epochs=15,
    n_bins=False,
    GEPC=False,  # Masked value prediction for cell embedding
    ecs_thres=0.0,  # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
    dab_weight=1.0,
    lr=1e-4,
    batch_size=32,
    layer_size=128,
    nlayers=4,
    nhead=4,
    # if load model, batch_size, layer_size, nlayers, nhead will be ignored
    dropout=0.25,
    schedule_ratio=0.92,  # ratio of epochs for learning rate schedule
    save_eval_interval=5,
    log_interval=100,
    fast_transformer=True,
    pre_norm=False,
    amp=True,  # Automatic Mixed Precision
)
run = wandb.init(
    config=hyperparameter_defaults,
    project="scGPT",
    reinit=True,
    settings=wandb.Settings(start_method="fork"),
)
config = wandb.config

set_seed(config.seed)

# %%
# settings for input and preprocessing
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
mask_ratio = config.mask_ratio
mask_value = -1
pad_value = -2
n_input_bins = config.n_bins

n_hvg = 1200  # number of highly variable genes
max_seq_len = n_hvg + 1
per_seq_batch_sample = False
DSBN = False  # Domain-spec batchnorm
explicit_zero_prob = False  # whether explicit bernoulli for zeros
use_gene_val_corr = False
# %%
dataset_name = config.dataset_name
save_dir = Path(f"./save/Test_NonzeroTrans_ad_siamease_cmpds_shuffled_mlp_features_control_mean_embed_log1p_target_without_mean_finetune_{dataset_name}-{time.strftime('%b%d-%H-%M')}/")
save_dir.mkdir(parents=True, exist_ok=True)
print(f"save to {save_dir}")
# save the whole script to the dir
os.system(f"cp {__file__} {save_dir}")

logger = scg.logger
scg.utils.add_file_handler(logger, save_dir / "run.log")


# %% [markdown]
# ## Loading and preparing data
if dataset_name == "PBMC_10K":
    adata = scvi.data.pbmc_dataset()  # 11990 × 3346
    ori_batch_col = "batch"
    adata.obs["celltype"] = adata.obs["str_labels"].astype("category")
    adata.var = adata.var.set_index("gene_symbols")
    data_is_raw = True
elif dataset_name == "CMAP_3CL_10uM":
    adata = sc.read_h5ad("/home/jguo02/GHDDI_workspace/PertGPT/scGPT/data/cmap_3CL_10um_24h.h5ad")
    ori_smiles_col = "canonical_smiles"
    adata.var = adata.var.set_index("features")
    data_is_raw = True
elif dataset_name == "K562_compoud21":
    adata = sc.read_h5ad("/home/jguo02/GHDDI_workspace/PertGPT/scGPT/data/K562_compoud21.h5ad")
    control_mask = adata.obs['treatment'].str.contains('S0000') #split adata into control and perturbed
    adata_control = adata[control_mask]
    adata_cmpd_perturbed = adata[~control_mask]
    ori_smiles_col = "canonical_smiles"
    adata_cmpd_perturbed.var = adata_cmpd_perturbed.var.set_index("symbol")
    adata_control.var = adata_control.var.set_index("symbol")
    data_is_raw = True
elif dataset_name == "K562_compoud188":
    adata = sc.read_h5ad("/home/jguo02/GHDDI_workspace/PertGPT/scGPT/data/K562_compoud188.h5ad")
    control_mask = adata.obs['treatment'].str.contains('S0000') #split adata into control and perturbed
    adata_control = adata[control_mask]
    adata_cmpd_perturbed = adata[~control_mask]
    ori_smiles_col = "canonical_smiles"
    adata_cmpd_perturbed.var = adata_cmpd_perturbed.var.set_index("symbol")
    adata_control.var = adata_control.var.set_index("symbol")
    data_is_raw = True

# %%

# make the smiles column
adata_cmpd_perturbed.obs["SMILES"] = adata_cmpd_perturbed.obs[ori_smiles_col].astype(str)

adata_cmpd_perturbed.var["gene_name"] = adata_cmpd_perturbed.var.index.tolist()
nan_rows = adata_cmpd_perturbed.var["gene_name"].isna()
adata_cmpd_perturbed = adata_cmpd_perturbed[:, ~nan_rows]

adata_control.var["gene_name"] = adata_control.var.index.tolist()
nan_rows_control = adata_control.var["gene_name"].isna()
adata_control = adata_control[:, ~nan_rows_control]


if config.load_model is not None:
    model_dir = Path(config.load_model)
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    vocab_file = model_dir / "vocab.json"

    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)
    
    ###-----process perturbed adata gene ids---####
    adata_cmpd_perturbed.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in adata_cmpd_perturbed.var["gene_name"]
    ]
    gene_ids_in_vocab = np.array(adata_cmpd_perturbed.var["id_in_vocab"])
    logger.info(
        f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
        f"in vocabulary of size {len(vocab)}."
    )
    adata_cmpd_perturbed = adata_cmpd_perturbed[:, adata_cmpd_perturbed.var["id_in_vocab"] >= 0]
    
    ###-----process contrl adata gene ids---####
    adata_control.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in adata_control.var["gene_name"]
    ]
    gene_ids_in_vocab = np.array(adata_control.var["id_in_vocab"])
    logger.info(
        f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
        f"in vocabulary of size {len(vocab)}."
    )
    adata_control = adata_control[:, adata_control.var["id_in_vocab"] >= 0]
    # model
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
    logger.info(
        f"Resume model from {model_file}, the model args will be overriden by the "
        f"config {model_config_file}."
    )
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs["n_layers_cls"]
else:
    vocab_file = "/home/jguo02/GHDDI_workspace/PertGPT/scGPT/save/scGPT_bc/vocab.json"
    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)
    
    ###-----process perturbed adata gene ids---####
    adata_cmpd_perturbed.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in adata_cmpd_perturbed.var["gene_name"]
    ]
    gene_ids_in_vocab = np.array(adata_cmpd_perturbed.var["id_in_vocab"])
    logger.info(
        f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
        f"in vocabulary of size {len(vocab)}."
    )
    adata_cmpd_perturbed = adata_cmpd_perturbed[:, adata_cmpd_perturbed.var["id_in_vocab"] >= 0]
    
    ###-----process contrl adata gene ids---####
    adata_control.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in adata_control.var["gene_name"]
    ]
    gene_ids_in_vocab = np.array(adata_control.var["id_in_vocab"])
    logger.info(
        f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
        f"in vocabulary of size {len(vocab)}."
    )
    adata_control = adata_control[:, adata_control.var["id_in_vocab"] >= 0]
    # model
    embsize = config.layer_size 
    nhead = config.nhead
    nlayers = config.nlayers  
    d_hid = config.layer_size


# %%
# set up the preprocessor, use the args to config the workflow
preprocessor_perturb = Preprocessor(
    use_key="X",  # the key in adata.layers to use as raw data
    filter_gene_by_counts=3,  # step 1
    filter_cell_by_counts=False,  # step 2
    normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
    result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
    log1p=data_is_raw,  # 4. whether to log1p the normalized data
    result_log1p_key="X_log1p",
    subset_hvg=n_hvg,  # 5. whether to subset the raw data to highly variable genes
    hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
    binning=config.n_bins,  # 6. whether to bin the raw data and to what number of bins
    # result_binned_key="X_binned",  # the key in adata.layers to store the binned data
)
preprocessor_perturb(adata_cmpd_perturbed)

preprocessor_control = Preprocessor(
    use_key="X",  # the key in adata.layers to use as raw data
    filter_gene_by_counts=0,  # step 1
    filter_cell_by_counts=False,  # step 2
    normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
    result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
    log1p=data_is_raw,  # 4. whether to log1p the normalized data
    result_log1p_key="X_log1p",
    subset_hvg=False,  # 5. whether to subset the raw data to highly variable genes
    hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
    binning=config.n_bins,  # 6. whether to bin the raw data and to what number of bins
    # result_binned_key="X_binned",  # the key in adata.layers to store the binned data
)

if adata_control.var.index.duplicated().any():
    adata_control = adata_control[:, ~adata_control.var.index.duplicated(keep='first')] # check remove duplicate gene ids and keep the first
adata_control = adata_control[:, adata_cmpd_perturbed.var.index] # get adata_control for 1200 hvg adata gene ids adata_cmpd_perturbed

preprocessor_control(adata_control)


# %% [markdown]
# ## Tokenize input

# %%

# input_layer_key = "X_binned"
input_layer_key = "X_log1p"
all_counts = (
    adata_cmpd_perturbed.layers[input_layer_key].A
    if issparse(adata_cmpd_perturbed.layers[input_layer_key])
    else adata_cmpd_perturbed.layers[input_layer_key]
)
genes = adata_cmpd_perturbed.var["gene_name"].tolist()
smiles_list =adata_cmpd_perturbed.obs["SMILES"].tolist()

all_counts_control = (
    adata_control.layers[input_layer_key].A
    if issparse(adata_control.layers[input_layer_key])
    else adata_control.layers[input_layer_key]
)

(
    train_data, 
    valid_data,
    train_smiles_list,
    valid_smiles_list,
) = train_test_split(
    all_counts, smiles_list, test_size=0.1, shuffle=True
)

def synchronized_shuffle(adata, smiles):
    """
    Shuffle the given adata object and smiles list in a synchronized manner.
    
    Parameters:
    - adata: An AnnData object that needs to be shuffled.
    - smiles: A list of SMILES strings associated with the adata object.
    
    Returns:
    - shuffled_adata: The shuffled AnnData object.
    - shuffled_smiles: The shuffled list of SMILES strings.
    """
    permutation = np.random.permutation(len(adata))
    shuffled_adata = adata[permutation]
    shuffled_smiles = np.array(smiles)[permutation].tolist()
    return shuffled_adata, shuffled_smiles

# Shuffle train_data and train_smiles_list together
train_data, train_smiles_list = synchronized_shuffle(train_data, train_smiles_list)

# Shuffle valid_data and valid_smiles_list together
valid_data, valid_smiles_list = synchronized_shuffle(valid_data, valid_smiles_list)

# %%
vocab.set_default_index(vocab["<pad>"])
gene_ids = np.array(vocab(genes), dtype=int)

# %%
tokenized_control= tokenize_and_pad_batch(
    all_counts_control,
    gene_ids,
    max_len=max_seq_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=True,  # append <cls> token at the beginning
    include_zero_gene=True,
)

tokenized_train = tokenize_and_pad_batch(
    train_data,
    gene_ids,
    max_len=max_seq_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=True,  # append <cls> token at the beginning
    include_zero_gene=True,
)
tokenized_valid = tokenize_and_pad_batch(
    valid_data,
    gene_ids,
    max_len=max_seq_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=True,
    include_zero_gene=True,
)
logger.info(
    f"control set number of samples: {tokenized_control['genes'].shape[0]}, "
    f"\n\t feature length: {tokenized_control['genes'].shape[1]}"
)
logger.info(
    f"train set number of samples: {tokenized_train['genes'].shape[0]}, "
    f"\n\t feature length: {tokenized_train['genes'].shape[1]}"
)
logger.info(
    f"valid set number of samples: {tokenized_valid['genes'].shape[0]}, "
    f"\n\t feature length: {tokenized_valid['genes'].shape[1]}"
)

# %%
def prepare_data(sort_seq_batch=False) -> Tuple[Dict[str, torch.Tensor]]:
    
    tokenized_control["values"] = torch.mean(tokenized_control["values"].float(), dim=0, keepdim=True) #take mean value for all control cells
    # tokenized_control["values"] = torch.zeros_like(tokenized_control["values"])
    control_values_train = resample(tokenized_control["values"], replace=True, n_samples=tokenized_train["values"].shape[0])

    masked_values_train = random_mask_value(
        control_values_train,
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    
    control_values_valid = resample(tokenized_control["values"], replace=True, n_samples=tokenized_valid["values"].shape[0])

    masked_values_valid = random_mask_value(
        control_values_valid,
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    print(
        f"random masking at epoch {epoch:3d}, ratio of masked values in train: ",
        f"{(masked_values_train == mask_value).sum() / (masked_values_train - pad_value).count_nonzero():.4f}",
    )

    input_gene_ids_train, input_gene_ids_valid = (
        tokenized_train["genes"],
        tokenized_valid["genes"],
    )
    input_values_train, input_values_valid = masked_values_train, masked_values_valid
    target_values_train, target_values_valid = (
        tokenized_train["values"],
        tokenized_valid["values"],
    )
    

    train_data_pt = {
        "gene_ids": input_gene_ids_train,
        "values": input_values_train,
        "target_values": target_values_train,
        "smiles_list": train_smiles_list,
    }
    valid_data_pt = {
        "gene_ids": input_gene_ids_valid,
        "values": input_values_valid,
        "target_values": target_values_valid,
        "smiles_list": valid_smiles_list,
    }

    return train_data_pt, valid_data_pt


# dataset
class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


# data_loader
def prepare_dataloader(
    data_pt: Dict[str, torch.Tensor],
    batch_size: int,
    shuffle: bool = False,
    intra_domain_shuffle: bool = False,
    drop_last: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    dataset = SeqDataset(data_pt)

    if per_seq_batch_sample:
        # find the indices of samples in each seq batch
        subsets = []
        batch_labels_array = data_pt["batch_labels"].numpy()
        for batch_label in np.unique(batch_labels_array):
            batch_indices = np.where(batch_labels_array == batch_label)[0].tolist()
            subsets.append(batch_indices)
        data_loader = DataLoader(
            dataset=dataset,
            batch_sampler=SubsetsBatchSampler(
                subsets,
                batch_size,
                intra_subset_shuffle=intra_domain_shuffle,
                inter_subset_shuffle=shuffle,
                drop_last=drop_last,
            ),
            num_workers=num_workers,
            pin_memory=True,
        )
        return data_loader

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
    )
    return data_loader


# %% [markdown]
# # Create and finetune scGPT

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ntokens = len(vocab)  # size of vocabulary
model = SiameseNetwork(Transformer4Cmpd(
    ntokens,
    embsize,
    nhead,
    d_hid,
    nlayers,
    vocab=vocab,
    dropout=config.dropout,
    pad_token=pad_token,
    pad_value=pad_value,
    do_mvc=config.GEPC,
    do_dab=False,
    use_batch_labels=False,
    domain_spec_batchnorm=None,
    n_input_bins=n_input_bins,
    ecs_threshold=config.ecs_thres,
    explicit_zero_prob=explicit_zero_prob,
    use_fast_transformer=config.fast_transformer,
    pre_norm=config.pre_norm,
))
if config.load_model is not None:
    try:
        model.load_state_dict(torch.load(model_file))
        logger.info(f"Loading all model params from {model_file}")
    except:
        # only load params that are in the model and match the size
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_file)
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        for k, v in pretrained_dict.items():
            logger.info(f"Loading params {k} with shape {v.shape}")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

model.to(device)
wandb.watch(model)


criterion = weighted_mse_loss
criterion_dab = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=config.lr, eps=1e-4 if config.amp else 1e-8
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=config.schedule_ratio)

scaler = torch.cuda.amp.GradScaler(enabled=config.amp)

def log_heatmap(corr_matrix, run, tag):
    fig, ax = plt.subplots()
    cax = ax.matshow(corr_matrix, cmap='coolwarm')
    fig.colorbar(cax)
    run.log({tag: wandb.Image(plt)})
    plt.close(fig)

def split_arrays_by_smiles(smiles_list, *arrays):
    """
    Split multiple numpy arrays based on the unique SMILES in the list.

    Parameters:
    - smiles_list (List[str]): List of SMILES strings.
    - *arrays (np.ndarrays): Multiple numpy arrays to be split.

    Returns:
    - Tuple containing:
        * Split arrays based on unique SMILES
        * unique_smiles (List[str]): List of unique SMILES strings.
    """
    
    unique_smiles = list(set(smiles_list))
    
    # Collect indices for each SMILES
    smiles_to_indices = {smiles: [] for smiles in unique_smiles}
    for idx, smiles in enumerate(smiles_list):
        smiles_to_indices[smiles].append(idx)

    # Split arrays based on the SMILES indices
    split_arrays = []
    for array in arrays:
        split_array = [array[smiles_to_indices[smiles]] for smiles in unique_smiles]
        split_arrays.append(split_array)

    return (*split_arrays, unique_smiles)

def gene_expression_analysis(total_control_values, total_output_values, total_target_values, smiles):
    """
    Analyze and visualize gene expression data.

    Parameters:
    - total_control_values (torch.Tensor): The control values tensor.
    - total_output_values (torch.Tensor): The predicted output values tensor.
    - total_target_values (torch.Tensor): The real target values tensor.
    - smiles (str): A string representing the SMILES notation of the compound.

    Returns:
    - corr_mean (float): The Pearson correlation between real mean and predicted mean.
    - delta_corr_mean (float): The Pearson correlation between the delta of real mean and predicted mean.
    - sp_corr_mean (float): The Spearman correlation between real mean and predicted mean.
    - delta_sp_corr_mean (float): The Spearman correlation between the delta of real mean and predicted mean.
    """
    
    prefix = smiles[:10]
    prefix = prefix.replace("/", "")

    # Compute statistics
    num_genes = total_output_values.shape[1]

    control_mean = np.mean(total_control_values, axis=0)
    real_mean = np.mean(total_target_values, axis=0)
    predicted_mean = np.mean(total_output_values, axis=0)

    corr_mean, _ = scipy.stats.pearsonr(real_mean, predicted_mean)
    delta_corr_mean, _ = scipy.stats.pearsonr(real_mean - control_mean, predicted_mean - control_mean)
    
    sp_corr_mean, _ = scipy.stats.spearmanr(real_mean, predicted_mean)
    delta_sp_corr_mean, _ = scipy.stats.spearmanr(real_mean - control_mean, predicted_mean - control_mean)

    # Create visualizations
    if epoch == 5 or epoch ==config.epochs:
        q2_values1 = np.percentile(total_target_values, 20, axis=0)
        q8_values1 = np.percentile(total_target_values, 80, axis=0)

        q2_values2 = np.percentile(total_output_values, 20, axis=0)
        q8_values2 = np.percentile(total_output_values, 80, axis=0)

        q2_values3 = np.percentile(total_control_values, 20, axis=0)
        q8_values3 = np.percentile(total_control_values, 80, axis=0)
        
        fig, axs = plt.subplots(4, 1, figsize=(15, 25))
        bar_width = 3
        gap = 3

        for i in range(4):
            start_idx = i * 300
            
            end_idx = (i + 1) * 300 if i != 3 else num_genes 

            genes = np.arange(start_idx, start_idx + (300 if i != 3 else 301), gap)

            sub_mean1 = real_mean[start_idx:end_idx][::gap]
            sub_mean2 = predicted_mean[start_idx:end_idx][::gap]
            sub_mean3 = control_mean[start_idx:end_idx][::gap]
            sub_q2_1 = q2_values1[start_idx:end_idx][::gap]
            sub_q8_1 = q8_values1[start_idx:end_idx][::gap]
            sub_q2_2 = q2_values2[start_idx:end_idx][::gap]
            sub_q8_2 = q8_values2[start_idx:end_idx][::gap]
            sub_q2_3 = q2_values3[start_idx:end_idx][::gap]
            sub_q8_3 = q8_values3[start_idx:end_idx][::gap]

            axs[i].bar(genes + bar_width/6, height=(sub_q8_1-sub_q2_1), bottom=sub_q2_1, alpha=0.4, width=bar_width, label='Range Perturbed', color='blue')
            axs[i].bar(genes, height=(sub_q8_2-sub_q2_2), bottom=sub_q2_2, alpha=0.4, width=bar_width, label='Range Predicted', color='orange')
            axs[i].bar(genes - bar_width/6, height=(sub_q8_3-sub_q2_3), bottom=sub_q2_3, alpha=0.4, width=bar_width, label='Range Control', color='green')

            axs[i].scatter(genes + bar_width/6, sub_mean1,label='Mean Perturbed', color='darkblue', s=10)
            axs[i].scatter(genes, sub_mean2, label='Mean Predicited',  color='red', s=10)
            axs[i].scatter(genes - bar_width/6, sub_mean3, label='Mean Control',  color='darkgreen', s=10)
            

            axs[i].set_xlabel('Gene No.')
            axs[i].set_ylabel('Gene Expression')
            axs[i].set_title(f'Genes from {start_idx} to {end_idx}')
            axs[i].legend()
            axs[i].grid(True, which='both', linestyle='--', linewidth=0.5)

        plt.tight_layout()# Using the prefix for naming the saved images
        plt_path = f"save/{prefix}_bar_visualization.png"
        plt.savefig(plt_path)
        plt.close()
        
        #plot delta value here
        delta_total_target_values = total_target_values - total_control_values
        delta_total_output_values = total_output_values - total_control_values
        delta_real_mean = np.mean(delta_total_target_values, axis=0)
        delta_predict_mean = np.mean(delta_total_output_values, axis=0)

        q2_values1 = np.percentile(delta_total_target_values, 20, axis=0)
        q8_values1 = np.percentile(delta_total_target_values, 80, axis=0)

        q2_values2 = np.percentile(delta_total_output_values, 20, axis=0)
        q8_values2 = np.percentile(delta_total_output_values, 80, axis=0)
        
        fig, axs = plt.subplots(4, 1, figsize=(15, 25))
        bar_width = 3
        gap = 3

        for i in range(4):
            start_idx = i * 300
            
            end_idx = (i + 1) * 300 if i != 3 else num_genes 

            genes = np.arange(start_idx, start_idx + (300 if i != 3 else 301), gap)

            sub_mean1 = delta_real_mean[start_idx:end_idx][::gap]
            sub_mean2 = delta_predict_mean[start_idx:end_idx][::gap]
            sub_q2_1 = q2_values1[start_idx:end_idx][::gap]
            sub_q8_1 = q8_values1[start_idx:end_idx][::gap]
            sub_q2_2 = q2_values2[start_idx:end_idx][::gap]
            sub_q8_2 = q8_values2[start_idx:end_idx][::gap]

            axs[i].bar(genes + bar_width/6, height=(sub_q8_1-sub_q2_1), bottom=sub_q2_1, alpha=0.4, width=bar_width, label='Range Perturbed-Ctrl', color='blue')
            axs[i].bar(genes, height=(sub_q8_2-sub_q2_2), bottom=sub_q2_2, alpha=0.4, width=bar_width, label='Range Predicted-Ctrl', color='orange')

            axs[i].scatter(genes + bar_width/6, sub_mean1,label='Mean Perturbed-Ctrl', color='darkblue', s=10)
            axs[i].scatter(genes, sub_mean2, label='Mean Predicted-Ctrl',  color='red', s=10)

            axs[i].set_xlabel('Gene No.')
            axs[i].set_ylabel('Gene Expression')
            axs[i].set_title(f'Genes from {start_idx} to {end_idx}')
            axs[i].legend()
            axs[i].grid(True, which='both', linestyle='--', linewidth=0.5)

        plt.tight_layout()
        plt_path2 = f"save/{prefix}_delta_bar_visualization.png"
        plt.savefig(plt_path2)
        plt.close()

        # Using wandb to log images
        wandb.log({f"{prefix}_bar_visualization": wandb.Image(plt_path)})
        wandb.log({f"{prefix}_delta_bar_visualization": wandb.Image(plt_path2)})

    return corr_mean, delta_corr_mean, sp_corr_mean, delta_sp_corr_mean

def kl_divergence(p, q):
    epsilon = 1e-10 
    p = p + epsilon
    q = q + epsilon
    return np.sum(p * np.log(p / q))

def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

def intra_corr_matrix_js_divergence(split_output, split_target, unique_smiles):
    """
    Compute and visualize the Jensen-Shannon divergence between target and output correlation matrices.
    
    Parameters:
    - split_control: List of control data splits.
    - split_output: List of model output data splits.
    - split_target: List of target data splits.
    - unique_smiles: List of unique SMILES strings representing compounds.
    
    Returns:
    - JS divergence value between the target and output correlation matrices.
    """
    targets_corr_matrix = np.zeros((len(split_target), len(split_target)))
    for m, split_target_m in enumerate(split_target):
        split_target_m_mean = np.mean(split_target_m, axis=0)
        for n, split_target_n in enumerate(split_target):
            split_target_n_mean = np.mean(split_target_n, axis=0)
            corr_target_mn, _ = scipy.stats.pearsonr(split_target_m_mean, split_target_n_mean)
            targets_corr_matrix[m, n] =  corr_target_mn

    # Compute prediction correlations between different compounds 
    outputs_corr_matrix = np.zeros((len(split_output), len(split_output)))
    for m, split_output_m in enumerate(split_output):
        split_output_m_mean = np.mean(split_output_m, axis=0)
        for n, split_output_n in enumerate(split_output):
            split_output_n_mean = np.mean(split_output_n, axis=0)
            corr_output_mn, _ = scipy.stats.pearsonr(split_output_m_mean, split_output_n_mean)
            outputs_corr_matrix[m, n] =  corr_output_mn

    # Plot heat map for two correlation matrix 
    targets_corr_matrix_min = np.min(targets_corr_matrix)
    outputs_corr_matrix_min = np.min(outputs_corr_matrix)
    # normalize_min = min(targets_corr_matrix_min, outputs_corr_matrix_min)
    normalized_targets_corr_matrix = (targets_corr_matrix - targets_corr_matrix_min)/(1.0 - targets_corr_matrix_min)
    normalized_outputs_corr_matrix = (outputs_corr_matrix - outputs_corr_matrix_min)/(1.0 - outputs_corr_matrix_min)
    # normalized_targets_corr_matrix = targets_corr_matrix
    # normalized_outputs_corr_matrix = outputs_corr_matrix
    fig, axarr = plt.subplots(1, 2, figsize=(12, 5))

    cax1 = axarr[0].matshow(normalized_targets_corr_matrix, cmap='RdBu_r', vmin=0, vmax=1)
    axarr[0].set_title('Intra-Perturbation Corre. Matrix')
    axarr[0].set_ylabel('Test Comps')
    plt.colorbar(cax1, ax=axarr[0])

    cax2 = axarr[1].matshow(normalized_outputs_corr_matrix, cmap='RdBu_r', vmin=0, vmax=1)
    axarr[1].set_title('Intra-Prediction Corre. Matrix')
    plt.colorbar(cax2, ax=axarr[1])

    plt.tight_layout()

    plt_path = f"save/Perturb&PredictHeatmaps.png"
    plt.savefig(plt_path)
    plt.close()

    wandb.log({"Heatmaps": [wandb.Image(fig, caption="Perturb & Predict Heatmaps")]})
    return js_divergence(targets_corr_matrix, outputs_corr_matrix)

def average_distance(points):
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    distances = np.sqrt((diff ** 2).sum(axis=2))
    upper_triangle_indices = np.triu_indices_from(distances, k=1)
    average_dist = distances[upper_triangle_indices].mean()
    return average_dist

def features_scatter(features, smiles_list, feature_name=None):
    """
    features: PyTorch tensor (batch_size x 1201 x 512)
    smiles_list: list of SMILES strings, length = batch_size
    """
    print(f'Drawing {feature_name} UMAP, 1/10 data points randomly sampled')
    
    if feature_name is None:
        feature_name = 'features'
    
    features_np = features

    indices = np.random.choice(features_np.shape[0], int(features_np.shape[0] * 0.1), replace=False)
    features_np_sampled = features_np[indices]
    smiles_list_sampled = [smiles_list[i] for i in indices]

    # Map each unique SMILES to a unique ID
    unique_smiles = list(set(smiles_list_sampled))
    smiles_to_id = {smiles: i for i, smiles in enumerate(unique_smiles)}
    ids_list_sampled = [smiles_to_id[smiles] for smiles in smiles_list_sampled]

    reducer = umap.UMAP()
    features_2d = reducer.fit_transform(features_np_sampled.reshape(features_np_sampled.shape[0], -1))
    avg_dist = average_distance(features_2d)  # Make sure you've defined average_distance somewhere
    logger.info("-" * 89)
    logger.info(f"Average distance between points: {avg_dist}")

    unique_ids = list(set(ids_list_sampled))
    colors = {id_: [np.random.rand(), np.random.rand(), np.random.rand()] for id_ in unique_ids}
    sample_colors = [colors[id_] for id_ in ids_list_sampled]

    scatter = {
        'x': features_2d[:, 0].tolist(),
        'y': features_2d[:, 1].tolist(),
        'labels': ids_list_sampled,
        'color': sample_colors
    }

    # Create a scatter plot
    plt.figure(figsize=(10, 7))
    for id_, c in colors.items():
        idx = [i for i, x in enumerate(ids_list_sampled) if x == id_]
        plt.scatter(features_2d[idx, 0], features_2d[idx, 1], c=[c for _ in idx], label=str(id_), s=5)

    # Estimate number of columns for the legend
    num_unique_ids = len(unique_ids)
    estimated_rows_per_column = 60 / 2.5  # Adjust this as needed
    ncols = max(1, round(num_unique_ids / estimated_rows_per_column))

    plt.legend(markerscale=2, loc='upper left', bbox_to_anchor=(1, 1), ncol=ncols)
    plt.xlim([-30, 30])
    plt.ylim([-30, 30])
    plt.title(f'UMAP plot of {feature_name} Average Dist: {avg_dist}')
    
    # Save the plot as an image
    image_path = f'{feature_name}_umap_plot.png'
    plt.savefig(image_path, bbox_inches='tight')
    plt.close()
    
    # Log the scatter plot as a wandb image
    wandb.log({f"{feature_name}_scatter_plot": wandb.Image(image_path)})
    
    return avg_dist

def train(model: nn.Module, loader: DataLoader) -> None:
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss, total_contrastive, total_mse1, total_mse2, total_corr_loss1, total_corr_loss2, total_gepc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    total_error = 0.0
    log_interval = config.log_interval
    start_time = time.time()

    num_batches = len(loader)
    train_control_values = []
    train_target_values = []
    train_smiles_comps = []
    for batch, batch_data in enumerate(loader):
        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        smiles_comps = batch_data["smiles_list"]
        target_values = batch_data["target_values"].to(device) - input_values.clone() ### set target to delta value
        # target_values = batch_data["target_values"].to(device)
        train_control_values.append(input_values)
        train_target_values.append(target_values)
        train_smiles_comps.extend(smiles_comps)
        
        mean_values = torch.mean(input_values, dim=0, keepdim=True)
        cotrastive_mask = (mean_values != 0.).float()
        cotrastive_mask = None
        if len(smiles_comps) % 2 == 1:
            # If odd, duplicate the last sample and append to each tensor or list
            input_gene_ids = torch.cat((input_gene_ids, input_gene_ids[-1].unsqueeze(0)))
            input_values = torch.cat((input_values, input_values[-1].unsqueeze(0)))
            target_values = torch.cat((target_values, target_values[-1].unsqueeze(0)))
            smiles_comps.append(smiles_comps[-1])
        half_batch_size = len(smiles_comps) // 2

        # Split the data into two halves
        input_gene_ids1 = input_gene_ids[:half_batch_size]
        input_gene_ids2 = input_gene_ids[half_batch_size:]

        input_values1 = input_values[:half_batch_size]
        input_values2 = input_values[half_batch_size:]

        target_values1 = target_values[:half_batch_size]
        target_values2 = target_values[half_batch_size:]

        smiles_comps1 = smiles_comps[:half_batch_size]
        smiles_comps2 = smiles_comps[half_batch_size:]

        src_key_padding_mask1 = input_gene_ids1.eq(vocab[pad_token])
        src_key_padding_mask2 = input_gene_ids2.eq(vocab[pad_token])
        is_same_smiles = [1 if smiles1 == smiles2 else 0 for smiles1, smiles2 in zip(smiles_comps1, smiles_comps2)]
        is_same_smiles = torch.tensor(is_same_smiles).to(device)
        with torch.cuda.amp.autocast(enabled=config.amp):
            input1 = (input_gene_ids1, input_values1, smiles_comps1, src_key_padding_mask1)
            input2 = (input_gene_ids2, input_values2, smiles_comps2, src_key_padding_mask2)
            features_gat1, features_transformer1, features_mlp1,  features_gat2, features_transformer2, features_mlp2, output_dict1, output_dict2 = model(input1, input2)
            
            loss1 = loss_mse1 = criterion(
                input_values1, output_dict1["mlm_output"], target_values1
            ) # weighted mse loss
            loss2 = loss_mse2 = criterion(
                input_values2, output_dict2["mlm_output"], target_values2
            ) # weighted mse loss
            loss = loss1 + loss2
            metrics_to_log = {"train/mse1": loss_mse1.item()}
            metrics_to_log = {"train/mse2": loss_mse2.item()}
           
            if use_gene_val_corr:
                loss_gene_val_corr1 = delta_cosine_loss(
                    input_values1, output_dict1["mlm_output"], target_values1)
                loss_gene_val_corr2 = delta_cosine_loss(
                    input_values2, output_dict2["mlm_output"], target_values2)
                loss = loss + loss_gene_val_corr1 + loss_gene_val_corr2
                metrics_to_log.update({"train/cosine1": loss_gene_val_corr1.item()})
                metrics_to_log.update({"train/cosine2": loss_gene_val_corr2.item()})

            contrastive_loss = contrastive_loss_multilayer(features_gat1, features_gat2, features_transformer1, features_transformer2, features_mlp1, features_mlp2,  is_same_smiles = is_same_smiles, mask=cotrastive_mask)
            loss = loss + contrastive_loss.mean()
            metrics_to_log.update({"train/contrastive": contrastive_loss.mean().item()})


        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
                error_if_nonfinite=False if scaler.is_enabled() else True,
            )
            if len(w) > 0:
                logger.warning(
                    f"Found infinite gradient. This may be caused by the gradient "
                    f"scaler. The current scale is {scaler.get_scale()}. This warning "
                    "can be ignored if no longer occurs after autoscaling of the scaler."
                )
        scaler.step(optimizer)
        scaler.update()

        wandb.log(metrics_to_log)

        total_loss += loss.item()
        total_mse1 += loss_mse1.item()
        total_mse2 += loss_mse2.item()
        total_contrastive += contrastive_loss.mean().item()
        try:
            total_corr_loss1 += (1-loss_gene_val_corr1.item()) 
            total_corr_loss2 += (1-loss_gene_val_corr2.item()) 
        except:
            total_corr_loss1 = 0.0
            total_corr_loss2 = 0.0
        total_gepc += loss_gepc.item() if config.GEPC else 0.0
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_mse1 = total_mse1 / log_interval
            cur_mse2 = total_mse2 / log_interval
            cur_corr1 = total_corr_loss1 / log_interval
            cur_corr2 = total_corr_loss2 / log_interval
            cur_contrastive = total_contrastive / log_interval
            cur_gepc = total_gepc / log_interval if config.GEPC else 0.0
            cur_error = total_error / log_interval
            logger.info(
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} |contrastive {cur_contrastive:5.2f} | mse1 {cur_mse1:5.2f} | mse2 {cur_mse2:5.2f} |"
                + (f"cosine1 {cur_corr1:5.2f} | cosine2 {cur_corr2:5.2f} " if use_gene_val_corr else "")
                + (f"gepc {cur_gepc:5.2f} |" if config.GEPC else "")
            )
            total_loss = 0
            total_mse1 = 0
            total_mse2 = 0
            total_corr_loss1 = 0
            total_corr_loss2 = 0
            total_contrastive = 0
            total_gepc = 0
            start_time = time.time()
    
    if epoch == 2: ## plot once
        concat_train_control_values = torch.cat(train_control_values, dim=0)
        concat_train_target_values = torch.cat(train_target_values, dim=0)

        concat_train_control_values = concat_train_control_values.cpu().numpy()
        concat_train_target_values = concat_train_target_values.cpu().numpy()

        split_control, split_target, unique_smiles = split_arrays_by_smiles(train_smiles_comps, 
                                                                    concat_train_control_values, 
                                                                    concat_train_target_values)

        ###compute perturbation correlations between different compounds 
        targets_corr_matrix = np.zeros((len(split_target), len(split_target)))
        for m, split_target_m in enumerate(split_target):
            split_target_m_mean = np.mean(split_target_m, axis=0)
            for n, split_target_n in enumerate(split_target):
                split_target_n_mean = np.mean(split_target_n, axis=0)
                corr_target_mn, _ = scipy.stats.pearsonr(split_target_m_mean, split_target_n_mean)
                targets_corr_matrix[m, n] =  corr_target_mn
        
        
        ####---plot heat map for two correlation matrix 
        normalize_min = np.min(targets_corr_matrix)
        normalized_targets_corr_matrix = (targets_corr_matrix - normalize_min)/(1.0 - normalize_min)
        fig, ax = plt.subplots(figsize=(8, 6))

        cax = ax.matshow(normalized_targets_corr_matrix, cmap='RdBu_r', vmin=0, vmax=1)
        ax.set_title('Intra-Perturbation Corre. Matrix')
        ax.set_ylabel('Train Comps')
        plt.colorbar(cax, ax=ax)
        plt.tight_layout()

        plt_path = f"save/train_Perturb&PredictHeatmaps.png"
        plt.savefig(plt_path)
        plt.close()

        wandb.log({"Heatmaps": [wandb.Image(fig, caption="Train Perturb & Predict Heatmaps")]})


        control_mean = np.mean(concat_train_control_values, axis=0)
        target_mean = np.mean(concat_train_target_values, axis=0)

        q2_values1 = np.percentile(concat_train_control_values, 20, axis=0)
        q8_values1 = np.percentile(concat_train_control_values, 80, axis=0)

        q2_values2 = np.percentile(concat_train_target_values, 20, axis=0)
        q8_values2 = np.percentile(concat_train_target_values, 80, axis=0)
        
        fig, axs = plt.subplots(4, 1, figsize=(15, 25))
        bar_width = 3
        gap = 3
        
        num_genes = concat_train_target_values.shape[1]

        for i in range(4):
            start_idx = i * 300
            
            end_idx = (i + 1) * 300 if i != 3 else num_genes 

            genes = np.arange(start_idx, start_idx + (300 if i != 3 else 301), gap)

            sub_mean1 = control_mean[start_idx:end_idx][::gap]
            sub_mean2 = target_mean[start_idx:end_idx][::gap]
            sub_q2_1 = q2_values1[start_idx:end_idx][::gap]
            sub_q8_1 = q8_values1[start_idx:end_idx][::gap]
            sub_q2_2 = q2_values2[start_idx:end_idx][::gap]
            sub_q8_2 = q8_values2[start_idx:end_idx][::gap]

            axs[i].bar(genes, height=(sub_q8_1-sub_q2_1), bottom=sub_q2_1, alpha=0.4, width=bar_width, label='Range Control', color='green')
            axs[i].bar(genes + bar_width/3, height=(sub_q8_2-sub_q2_2), bottom=sub_q2_2, alpha=0.4, width=bar_width, label='Range Perturbed', color='blue')

            axs[i].scatter(genes, sub_mean1,label='Mean Control', color='darkgreen', s=10)
            axs[i].scatter(genes + bar_width/3, sub_mean2, label='Mean Perturbed',  color='darkblue', s=10)

            axs[i].set_xlabel('Gene No.')
            axs[i].set_ylabel('Gene Expression')
            axs[i].set_title(f'Genes from {start_idx} to {end_idx}')
            axs[i].legend()
            axs[i].grid(True, which='both', linestyle='--', linewidth=0.5)

        plt.tight_layout()

        plt_path = "save/train_bar_visualization.png"
        plt.savefig(plt_path)
        plt.close()



def define_wandb_metrcis():
    wandb.define_metric("valid/mse", summary="min", step_metric="epoch")
    wandb.define_metric("valid/dab", summary="min", step_metric="epoch")
    wandb.define_metric("valid/mean_pearson", summary="min", step_metric="epoch")
    wandb.define_metric("valid/mean_delta_pearson", summary="min", step_metric="epoch")
    wandb.define_metric("valid/mean_spearman", summary="min", step_metric="epoch")
    wandb.define_metric("valid/mean_delta_spearman", summary="min", step_metric="epoch")
    wandb.define_metric("valid/std_pearsons", summary="min", step_metric="epoch")
    wandb.define_metric("valid/std_delta_pearsons", summary="min", step_metric="epoch")
    wandb.define_metric("valid/std_spearmans", summary="min", step_metric="epoch")
    wandb.define_metric("valid/std_delta_spearmans", summary="min", step_metric="epoch")
    wandb.define_metric("valid/corr_matrix_js_div", summary="min", step_metric="epoch")
    wandb.define_metric("valid/compds_features_distance", summary="min", step_metric="epoch")



def evaluate(model: nn.Module, loader: DataLoader) -> float:
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()
    total_loss = 0.0
    total_avg_correlations = 0.0
    total_error = 0.0
    total_num = 0
    with torch.no_grad():
        total_control_values = []
        total_output_values =[]
        total_target_values = []
        total_smiles_comps = []
        total_features_gat = []
        total_features_mlp = []
        for batch_data in loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            smiles_comps = batch_data["smiles_list"]
            # target_values = batch_data["target_values"].to(device)
            target_values = batch_data["target_values"].to(device) - input_values.clone() ### set target to delta value

            if len(smiles_comps) % 2 == 1:
                # If odd, duplicate the last sample and append to each tensor or list
                input_gene_ids = torch.cat((input_gene_ids, input_gene_ids[-1].unsqueeze(0)))
                input_values = torch.cat((input_values, input_values[-1].unsqueeze(0)))
                target_values = torch.cat((target_values, target_values[-1].unsqueeze(0)))
                smiles_comps.append(smiles_comps[-1])
            half_batch_size = len(smiles_comps) // 2
            
            # Split the data into two halves
            input_gene_ids1 = input_gene_ids[:half_batch_size]
            input_gene_ids2 = input_gene_ids[half_batch_size:]

            input_values1 = input_values[:half_batch_size]
            input_values2 = input_values[half_batch_size:]

            target_values1 = target_values[:half_batch_size]
            target_values2 = target_values[half_batch_size:]

            smiles_comps1 = smiles_comps[:half_batch_size]
            smiles_comps2 = smiles_comps[half_batch_size:]

            src_key_padding_mask1 = input_gene_ids1.eq(vocab[pad_token])
            src_key_padding_mask2 = input_gene_ids2.eq(vocab[pad_token])

            with torch.cuda.amp.autocast(enabled=config.amp):
                

                input1 = (input_gene_ids1, input_values1, smiles_comps1, src_key_padding_mask1)
                input2 = (input_gene_ids2, input_values2, smiles_comps2, src_key_padding_mask2)

                features_gat1, _, features_mlp1, features_gat2, _, features_mlp2, output_dict1, output_dict2 = model(input1, input2)

                output_values1 = output_dict1["mlm_output"]
                output_values2 = output_dict2["mlm_output"]
                
                loss1 = criterion(input_values1, output_values1, target_values1)  ####  weighted mse
                loss2 = criterion(input_values2, output_values2, target_values2)  ####  weighted mse
                
                total_control_values.append(input_values1)
                total_control_values.append(input_values2)
                total_smiles_comps.extend(smiles_comps1)
                total_smiles_comps.extend(smiles_comps2)
                total_features_gat.append(features_gat1)
                total_features_gat.append(features_gat2)
                total_features_mlp.append(features_mlp1)
                total_features_mlp.append(features_mlp2)
                total_output_values.append(output_values1)
                total_output_values.append(output_values2)
                total_target_values.append(target_values1)
                total_target_values.append(target_values2)
            total_loss += (loss1.item() * len(input_gene_ids1) + loss2.item() * len(input_gene_ids2))
            total_num += (len(input_gene_ids1) +len(input_gene_ids2))
        
        concat_total_control_values = torch.cat(total_control_values, dim=0)
        concat_total_output_values = torch.cat(total_output_values, dim=0)
        concat_total_target_values = torch.cat(total_target_values, dim=0)
        concat_total_features_gat = torch.cat(total_features_gat, dim=0)
        concat_total_features_mlp = torch.cat(total_features_mlp, dim=0)
        # concat_total_features_transformer = torch.cat(total_features_transformer, dim=0)
        
        concat_total_control_values = concat_total_control_values.cpu().numpy()
        concat_total_output_values = concat_total_output_values.cpu().numpy()
        concat_total_target_values = concat_total_target_values.cpu().numpy()
        concat_total_features_gat = concat_total_features_gat.cpu().numpy()
        concat_total_features_mlp = concat_total_features_mlp.cpu().numpy()
        # concat_total_features_transformer = concat_total_features_transformer.cpu().numpy()

        split_control, split_output, split_target, unique_smiles = split_arrays_by_smiles(total_smiles_comps, 
                                                                    concat_total_control_values, 
                                                                    concat_total_output_values, 
                                                                    concat_total_target_values)
        
        js_div = intra_corr_matrix_js_divergence(split_output, split_target, unique_smiles)
        
        if 1:
            features_GAT_avg_dist = features_scatter(concat_total_features_gat, total_smiles_comps, feature_name='features_GAT_ad_models') # draw features scatter
            features_mlp_avg_dist = features_scatter(concat_total_features_mlp, total_smiles_comps, feature_name='features_mlp_ad_model') # draw features scatter
        

        corr_means, delta_corr_means, sp_corr_means, delta_sp_corr_means = np.array([]), np.array([]), np.array([]), np.array([])
        
        for split_control_i, split_output_i, split_target_i, unique_smiles_i in zip(split_control, split_output, split_target, unique_smiles):
            corr_mean_i, delta_corr_mean_i, sp_corr_mean_i, delta_sp_corr_mean_i = gene_expression_analysis(
                                                                        split_control_i,
                                                                        split_output_i,
                                                                        split_target_i,
                                                                        unique_smiles_i)
            corr_means = np.append(corr_means, corr_mean_i)
            delta_corr_means = np.append(delta_corr_means, delta_corr_mean_i)
            sp_corr_means = np.append(sp_corr_means, sp_corr_mean_i)
            delta_sp_corr_means = np.append(delta_sp_corr_means, delta_sp_corr_mean_i)

        corr_mean = np.mean(corr_means)
        corr_std = np.std(corr_means)
        delta_corr_mean = np.mean(delta_corr_means)
        delta_corr_std = np.std(delta_corr_means)
        sp_corr_mean = np.mean(sp_corr_means)
        sp_corr_std = np.std(sp_corr_means)
        delta_sp_corr_mean = np.mean(delta_sp_corr_means)
        delta_sp_corr_std = np.std(delta_sp_corr_means)


           
    wandb.log(
        {
            "valid/mse": total_loss / total_num,
            "valid/std_pearsons": corr_std,
            "valid/mean_pearson": corr_mean,
            "valid/std_delta_pearsons": delta_corr_std,
            "valid/mean_delta_pearson": delta_corr_mean,
            "valid/std_spearmans": sp_corr_std,
            "valid/mean_spearman": sp_corr_mean,
            "valid/std_delta_spearmans": delta_sp_corr_std,
            "valid/mean_delta_spearman": delta_sp_corr_mean,
            "valid/corr_matrix_js_div": js_div,
            "valid/compds_features_distance": features_GAT_avg_dist,
            "epoch": epoch,
        },
    )

    return total_loss / total_num, corr_mean, delta_corr_mean,  sp_corr_mean, delta_sp_corr_mean, js_div



# %%
best_val_loss = float("inf")
best_avg_bio = 0.0
best_model = None
define_wandb_metrcis()

for epoch in range(1, config.epochs + 1):
    epoch_start_time = time.time()
    train_data_pt, valid_data_pt = prepare_data(sort_seq_batch=per_seq_batch_sample)
    train_loader = prepare_dataloader(
        train_data_pt,
        batch_size=config.batch_size,
        shuffle=False,
        intra_domain_shuffle=False,
        drop_last=False,
    )
    valid_loader = prepare_dataloader(
        valid_data_pt,
        batch_size=config.batch_size,
        shuffle=False,
        intra_domain_shuffle=False,
        drop_last=False,
    )

    if config.do_train:
        train(
            model,
            loader=train_loader,
        )
    val_loss, val_corr, val_delta_corr, val_sp_corr, val_delta_sp_corr, val_js_div = evaluate(
        model,
        loader=valid_loader,
    )
    elapsed = time.time() - epoch_start_time
    logger.info("-" * 89)
    logger.info(
        f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
        f"valid loss/mse {val_loss:5.4f} | mean pearson {val_corr:5.4f}| mean delta pearson {val_delta_corr:5.4f} | mean spearman {val_sp_corr:5.4f} | mean delta spearman {val_delta_sp_corr:5.4f} | matrix js div {val_js_div:5.4f}"
    )
    logger.info("-" * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model)
        best_model_epoch = epoch
        logger.info(f"Best model with score {best_val_loss:5.4f}")

    if epoch % config.save_eval_interval == 0 or epoch == config.epochs:
        logger.info(f"Saving model to {save_dir}")
        torch.save(best_model.state_dict(), save_dir / f"model_e{best_model_epoch}.pt")

    scheduler.step()


# %%
# save the best model
torch.save(best_model.state_dict(), save_dir / "best_model.pt")

# %% [markdown]
# ## Gene embeddings

# %%
artifact = wandb.Artifact(f"best_model", type="model")
glob_str = os.path.join(save_dir, "best_model.pt")
artifact.add_file(glob_str)
run.log_artifact(artifact)

run.finish()
wandb.finish()
gc.collect()
