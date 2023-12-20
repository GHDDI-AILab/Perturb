__all__ = ['PertData']

import warnings
from typing import Optional
from pathlib import Path
from zipfile import ZipFile

import torch
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy import sparse
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from .preprocess import Preprocessor
from ..tokenizer import GeneVocab
from ..utils import create_logger, data_downloader, read_json, write_json

sc.settings.verbosity = 1
warnings.filterwarnings("ignore")

class PertBase:
    """A base class for perturbation data classes."""
    covariate: str = "cell_type"
    condition_name: str = "condition_name"
    key_de_genes: str = "rank_genes_groups_cov_all"
    num_de_genes: int = 20

    def _check_mode(self) -> None:
        """
        Identify the perturbation mode: 'gene' or 'compound',
        and set the corresponding values to attributes.
        Maybe we can add NEW data modes later.

        Returns:
            None
        """
        if hasattr(self, "mode") and self.mode in ["gene", "compound"]:
            return

        if not hasattr(self, "adata") or self.adata is None:
            raise AttributeError("adata not loaded!")

        if self.covariate not in self.adata.obs.columns:
            raise ValueError(f"Cannot find '{self.covariate}' in adata.obs!")

        if set(["canonical_smiles",]).issubset(self.adata.obs.columns):
            self.mode = "compound"
            #self.cond_col = "treatment"
            #self.ctrl_str = "S0000"
            self.pert_col = "canonical_smiles"
            self.ctrl_str = "Vehicle"
            self.ctrl_group = "Vehicle"  # for calculate DE
        elif set(["condition",]).issubset(self.adata.obs.columns):
            self.mode = "gene"
            #self.cond_col = "condition"
            self.pert_col = "condition"
            self.ctrl_str = "ctrl"
            self.ctrl_group = "ctrl_1"  # for calculate DE
        else:
            raise ValueError("Cannot identify the pert mode!")

        if "symbol" in self.adata.var.columns:
            self.gene_col = "symbol"
        else:
            self.gene_col = "gene_name"

        if self.gene_col not in self.adata.var.columns:
            self.adata.var[self.gene_col] = self.adata.var.index

    def _drop_NA_genes(self) -> None:
        """
        Reset the columns of adata with missing gene names.

        Returns:
            None
        """
        self.adata = self.adata[:,
            self.adata.var.dropna(subset=[self.gene_col]).index]

    def _set_non_dropout_non_zero_genes(
            self, key: Optional[str] = None
        ) -> None:
        """
        Set non-dropout and non-zero genes after ranking genes for groups.

        Reference:
          https://github.com/snap-stanford/GEARS/blob/master/gears/data_utils.py

        Args:
            key (str, optional):
                The key to save the information in `adata.uns`

        Returns:
            None
        """
        if key is None:
            key = self.key_de_genes

        groupby = self.condition_name

        # Calculate mean expression for each condition
        unique_conditions = self.adata.obs[self.pert_col].unique()
        conditions2index = {
            i: np.where(self.adata.obs[self.pert_col] == i)[0]
            for i in unique_conditions
        }
        condition2mean_expression = {
            i: np.mean(self.adata.X[j], axis = 0)
            for i, j in conditions2index.items()
        }
        pert_list = np.array(
            list(condition2mean_expression.keys())
        )
        mean_expression = np.array(
            list(condition2mean_expression.values())
        ).reshape(
            len(unique_conditions), self.adata.X.toarray().shape[1]
        )
        ctrl = mean_expression[np.where(pert_list == self.ctrl_str)[0]]
        
        ## In silico modeling and upperbounding
        pert2pert_full_id = dict(
            self.adata.obs[[self.pert_col, groupby]].values
        )
        pert_full_id2pert = dict(
            self.adata.obs[[groupby, self.pert_col]].values
        )
        gene_id2idx = dict(
            zip(self.adata.var.index.values, range(len(self.adata.var)))
        )
        gene_idx2id = dict(
            zip(range(len(self.adata.var)), self.adata.var.index.values)
        )

        non_zero_gene_idx = {}
        top_non_zero_de_20 = {}
        non_dropout_gene_idx = {}
        top_non_dropout_de_20 = {}

        for pert in self.adata.uns[key].keys():
            p = pert_full_id2pert[pert]
            X = np.mean(
                self.adata[self.adata.obs[self.pert_col] == p].X, axis=0
            )
            non_zero = np.where(np.array(X)[0] != 0)[0]
            zero = np.where(np.array(X)[0] == 0)[0]
            true_zeros = np.intersect1d(
                zero, np.where(np.array(ctrl)[0] == 0)[0]
            )
            non_dropouts = np.concatenate((non_zero, true_zeros))

            top = self.adata.uns[key][pert]
            gene_idx_top = [gene_id2idx[i] for i in top]

            non_dropout_20 = [i for i in gene_idx_top if i in non_dropouts][:20]
            non_dropout_20_gene_id = [gene_idx2id[i] for i in non_dropout_20]
            non_zero_20 = [i for i in gene_idx_top if i in non_zero][:20]
            non_zero_20_gene_id = [gene_idx2id[i] for i in non_zero_20]

            non_zero_gene_idx[pert] = np.sort(non_zero)
            top_non_zero_de_20[pert] = np.array(non_zero_20_gene_id)
            non_dropout_gene_idx[pert] = np.sort(non_dropouts)
            top_non_dropout_de_20[pert] = np.array(non_dropout_20_gene_id)
        
        self.adata.uns['top_non_dropout_de_20'] = top_non_dropout_de_20
        self.adata.uns['non_dropout_gene_idx'] = non_dropout_gene_idx
        self.adata.uns['top_non_zero_de_20'] = top_non_zero_de_20
        self.adata.uns['non_zero_gene_idx'] = non_zero_gene_idx
    
    def set_DE_genes(
            self, key: Optional[str] = None, check_logged: bool = False
        ) -> None:
        """
        Rank genes for characterizing groups.

        Reference:
          https://github.com/snap-stanford/GEARS/blob/master/gears/data_utils.py

        Args:
            key (str, optional):
                The key to save the information in `adata.uns`
            check_logged (bool):
                To check if the data is logarithmized or not (default: False)

        Returns:
            None
        """
        key_to_process = None

        if key is None:
            key = self.key_de_genes

        if check_logged:
            is_logged = Preprocessor.check_logged(self.adata, key_to_process)
            if not is_logged:
                raise ValueError("Expecting logarithmized data.")

        groupby = self.condition_name
        control_group = self.ctrl_group

        if self.mode == "gene":
            self.adata.obs['control'] = self.adata.obs[self.pert_col].apply(
                lambda x: 0 if len(x.split('+')) == 2 else 1
            )
            self.adata.obs['dose_val'] = self.adata.obs[self.pert_col].apply(
                lambda x: '1+1' if len(x.split('+')) == 2 else '1'
            )
            self.adata.obs[groupby] = self.adata.obs.apply(
                lambda x: '_'.join([
                    x[self.covariate], x[self.pert_col], x['dose_val']
                ]), axis = 1
            )
        elif self.mode == "compound":
            self.adata.obs[groupby] = self.adata.obs.apply(
                lambda x: '_'.join([
                    x[self.covariate], x[self.pert_col]
                ]), axis = 1
            )

        self.adata.obs = self.adata.obs.astype('category')
        cov_categories = self.adata.obs[self.covariate].unique()
        gene_dict = {}
        for cov_cat in cov_categories:
            #name of the control group in the groupby obs column
            control_group_cov = '_'.join([cov_cat, control_group])
            #subset adata to cells belonging to a covariate category
            adata_cov = self.adata[self.adata.obs[self.covariate] == cov_cat]
            #compute DEGs
            sc.tl.rank_genes_groups(
                adata_cov,
                groupby=groupby,
                reference=control_group_cov,
                rankby_abs=True,
                n_genes=len(self.adata.var),
                use_raw=False,
                layer=key_to_process,
            )
            #add entries to dictionary of gene sets
            de_genes = pd.DataFrame(adata_cov.uns['rank_genes_groups']['names'])
            for group in de_genes:
                gene_dict[group] = np.array(de_genes[group].tolist())

        self.adata.uns[key] = gene_dict
        self._set_non_dropout_non_zero_genes(key)

    def get_DE_genes(self, key: Optional[str] = None) -> Optional[dict]:
        """
        Extract the information of DE genes from `adata.uns`.

        Args:
            key (str, optional):
                the key corresponding to the information of DE genes.

        Returns:
            dict | None
        """
        if key is None:
            key = self.key_de_genes

        if key in self.adata.uns:
            return self.adata.uns[key]

    def get_gene_ids(self) -> np.ndarray[int]:
        """
        Convert gene names to an integer array using the preloaded vocabulary.

        Returns:
            np.ndarray[int]
        """
        if not hasattr(self, "vocab") or self.vocab is None:
            raise AttributeError('vocab not set!')

        genes = self.adata.var[self.gene_col].tolist()
        return np.array(self.vocab(genes), dtype=int)


class PertDataset(Dataset, PertBase):
    """
    Storing and indexing perturbation data.

    Reference for DE genes:
      https://github.com/snap-stanford/GEARS/blob/master/gears/pertdata.py#L512
    """
    
    def __init__(self, x: ad.AnnData, y: ad.AnnData, vocab: GeneVocab) -> None:
        assert x.shape == y.shape, "x and y do not have the same shape!"
        self.ctrl_adata = self.x = x
        self.adata = self.y = y
        self.vocab = vocab

        self._check_mode()
        self.id2gene = self.adata.var.to_dict()[self.gene_col]  # dict
        self.gene_ids = self.get_gene_ids()  # np.ndarray[int]
        self.de_genes = self.get_DE_genes()  # dict | None

    def __getitem__(self, idx: int) -> dict:
        x_ = self.ctrl_adata[idx]
        y_ = self.adata[idx]

        if self.de_genes is None:
            de_idx = de_genes = [-1]
        elif y_.obs[self.pert_col][0] == self.ctrl_str:
            de_idx = de_genes = [-1] * self.num_de_genes
        else:
            key = y_.obs[self.condition_name][0]  # e.g. "A549_FOXA3+ctrl_1+1"
            de_genes = self.de_genes[key][:self.num_de_genes]
            de_idx = np.where(y_.var.index.isin(de_genes))[0]
            de_genes = self.vocab([self.id2gene[k] for k in de_genes])

        return {
            'x': {
                'X': x_.X.A.reshape(x_.n_vars),  # np.ndarray
                'obs': x_.obs.reset_index().to_dict("records")[0],  # dict
            },
            'y': {
                'X': y_.X.A.reshape(y_.n_vars),  # np.ndarray
                'obs': y_.obs.reset_index().to_dict("records")[0],  # dict
            },
            'de_idx': np.array(de_idx, dtype=int),
            'de_genes': np.array(de_genes, dtype=int),
            'gene_ids': self.gene_ids,  # np.ndarray
        }

    def __len__(self) -> int:
        return self.adata.shape[0]


class SeqDataset(Dataset):
    def __init__(self, data: dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["values"].shape[0]

    def __getitem__(self, idx: int):
        return {k: self.data[k][idx] for k in self.data}


class PertData(PertBase):
    
    ctrl_flag: int = 0
    crispr_flag: int = 1
    pert_pad_id: int = 2
    
    special_tokens: list[str] = ["<pad>", "<cls>", "<eoc>"]
    cls_token: str = "<cls>"
    pad_token: str = "<pad>"
    pad_value: int = -2
    
    urls = {
        # load from harvard dataverse
        'dixit':   'https://dataverse.harvard.edu/api/access/datafile/6154416',
        'norman':  'https://dataverse.harvard.edu/api/access/datafile/6154020',
        'adamson': 'https://dataverse.harvard.edu/api/access/datafile/6154417',
    }

    def __init__(
            self,
            dataset: str,
            workdir: Path = Path('data'),
            keep_ctrl: bool = True,
            seed: int = 1,
            vocab_file: Optional[Path] = None,
        ) -> None:
        self.logger = create_logger(name=self.__class__.__name__)
        self.logger.info('Init...')
        self.dataset = str(dataset)
        self.workdir = Path(workdir)
        if not self.workdir.exists():
            self.workdir.mkdir(parents=True)
        self.keep_ctrl = keep_ctrl
        self.seed = seed

        self._load_adata()
        self._check_mode()
        self.logger.info(f'mode = {repr(self.mode)}')
        self._drop_NA_genes()
        self.set_vocab(vocab_file)
        self.splitter = DataSplitter(self)
        self.logger.info(f'Got a data splitter: seed = {seed}')

    def _load_adata(self) -> None:
        """
        Load an AnnData object with a dataset name or a local data path.
        Attributes will be addded:
            self.adata
            self.dataset_name
            self.dataset_path

        Returns:
            None
        """
        if self.dataset.lower() in ('dixit', 'norman', 'adamson'):
            data_path = self.workdir / self.dataset
            self._zipdata_download_wrapper(
                url=self.urls[self.dataset.lower()],
                save_path=data_path
            )
            data_file = data_path / 'perturb_processed.h5ad'
        elif Path(self.dataset).is_dir():
            data_path = Path(self.dataset)
            data_file = data_path / 'perturb_processed.h5ad'
        elif Path(self.dataset).is_file():
            data_file = Path(self.dataset)
            data_path = data_file.parent
        else:
            raise ValueError(
                "The dataset is either Norman/Adamson/Dixit "
                "or a directory with a perturb_processed.h5ad file."
            )
        self.adata = ad.read_h5ad(data_file)
        self.ctrl_adata = None
        self.dataset_name = data_path.absolute().name
        self.dataset_path = data_path
        self.logger.info('Loaded adata.')

    def _zipdata_download_wrapper(self, url: str, save_path: str) -> None:
        if Path(save_path).exists():
            self.logger.info('Found local copy.')
        else:
            file_path = str(save_path) + '.zip'
            self.logger.info('Downloading...')
            data_downloader(url, file_path)
            self.logger.info('Extracting zip file...')
            with ZipFile(file_path, 'r') as f:
                f.extractall(path=self.workdir)
            self.logger.info('Done.')
 
    def set_vocab(self, vocab_file: Optional[Path]) -> None:
        if vocab_file:
            vocab = GeneVocab.from_file(vocab_file)
            for s in self.special_tokens:
                if s not in vocab:
                    vocab.append_token(s)
        elif self.adata:
            genes = self.adata.var[self.gene_col].tolist()
            vocab = GeneVocab(genes + self.special_tokens)
        else:
            raise AttributeError('vocab_file not given and adata not loaded!')
        vocab.set_default_index(vocab[self.pad_token])
        self.vocab = vocab
        self.logger.info('Loaded vocab.')

    def preprocess(
            self,
            use_key: Optional[str] = None,
            filter_gene_by_counts: int|bool = False,
            filter_cell_by_counts: int|bool = False,
            normalize_total: float|bool = False,
            log1p: bool = False,
            batch_key: Optional[str] = None,
            hvg_use_key: Optional[str] = None,
            subset_hvg: int|bool = False,
            binning: int|bool = False,
        ) -> None:
        """
        Data preprocessing.

        Returns:
            None
        """
        is_logged = Preprocessor.check_logged(self.adata)
        hvg_flavor = "cell_ranger" if is_logged else "seurat_v3"
        preprocessor = Preprocessor(
            use_key=use_key,
            filter_gene_by_counts=filter_gene_by_counts,
            filter_cell_by_counts=filter_cell_by_counts,
            normalize_total=normalize_total,
            log1p=log1p,
            subset_hvg=subset_hvg,
            hvg_use_key=hvg_use_key,
            hvg_flavor=hvg_flavor,
            binning=binning,
        )
        key_to_process = preprocessor(self.adata, batch_key=batch_key)

        if any([filter_gene_by_counts, filter_cell_by_counts, 
                normalize_total, log1p, subset_hvg, binning,]):
            self.adata.X = sparse.csr_matrix(
                sc.get._get_obs_rep(self.adata, layer=key_to_process)
            )
            self.set_DE_genes()
            self.logger.info('Calculated differentially expressed genes.')
            self.logger.info('Preprocessed adata.')

    def prepare_split(self, *args, **kwargs) -> None:
        """
        Prepare splits of train, val, and test perturbations.

        Returns:
            None
        """
        self.splitter.prepare_split(*args, **kwargs)
        self.logger.info('Prepared splits.')

    def transform_data(self, x: ad.AnnData, y: ad.AnnData, vocab: GeneVocab) -> dict:
        """
        Transforming the control and perturbed AnnData into a new format,
        as the input for tokenization.

        Returns:
            dict
        """
        assert x.shape == y.shape, "x and y do not have the same shape!"
        de_dict: dict | None = self.get_DE_genes()
        id2gene: dict = y.var.to_dict()[self.gene_col]

        if de_dict is None:
            de_idx = de_genes = [[-1]] * y.n_obs
        else:
            de_idx = []
            de_genes = []
            for i in range(y.n_obs):
                if y.obs[self.pert_col][i] == self.ctrl_str:
                    de_idx.append([-1] * self.num_de_genes)
                    de_genes.append([-1] * self.num_de_genes)
                else:
                    key = y.obs[self.condition_name][i]
                    de_ = de_dict[key][:self.num_de_genes]
                    de_idx.append(np.where(y.var.index.isin(de_))[0])
                    de_genes.append(vocab([id2gene[k] for k in de_]))

        return {
            'x': {'X': torch.tensor(x.X.A), 'obs': x.obs.to_dict('list')},
            'y': {'X': torch.tensor(y.X.A), 'obs': y.obs.to_dict('list')},
            'de_idx': torch.tensor(de_idx, dtype=int),
            'de_genes': torch.tensor(de_genes, dtype=int),
            'gene_ids': torch.tensor(self.get_gene_ids()).expand([y.n_obs, -1]),
        }

    def _create_dataset(
            self, adata: ad.AnnData, max_len: int = 0, append_cls: bool = True,
        ) -> SeqDataset:
        if self.ctrl_adata is None:
            self.ctrl_adata = self.adata[
                self.adata.obs[self.pert_col] == self.ctrl_str
            ]

        indices = np.random.randint(0, len(self.ctrl_adata), len(adata))
        return SeqDataset(
            self.get_tokenized_batch(
                self.transform_data(self.ctrl_adata[indices], adata, self.vocab),
                max_len=max_len, append_cls=append_cls,
            )
        )

    def set_dataloader(
            self,
            batch_size: int,
            test_batch_size: Optional[int] = None,
            max_len: int = 0,
            test_max_len: int = 0,
            append_cls: bool = True,
        ) -> None:
        """
        Set dataloaders of train, val, and test sets.

        Returns:
            None
        """
        if test_batch_size is None:
            test_batch_size = batch_size

        split = self.splitter.split_data()
        train_loader = DataLoader(
            self._create_dataset(split['train'], max_len, append_cls),
            batch_size=batch_size, shuffle=True, drop_last=True
        )
        val_loader = DataLoader(
            self._create_dataset(split['val'], test_max_len, append_cls),
            batch_size=test_batch_size, shuffle=True
        )
        test_loader = DataLoader(
            self._create_dataset(split['test'], test_max_len, append_cls),
            batch_size=test_batch_size, shuffle=False
        )
        self.dataloader = {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
        }
        self.logger.info('Created dataloader.')

    def _create_dataset_for_prediction(
            self, perturbation: str, pool_size: Optional[int] = None
        ) -> PertDataset:
        if self.ctrl_adata is None:
            self.ctrl_adata = self.adata[
                self.adata.obs[self.pert_col] == self.ctrl_str
            ]

        if pool_size is None:
            pool_size = len(self.ctrl_adata)

        indices = np.random.randint(0, len(self.ctrl_adata), pool_size)
        x = self.ctrl_adata[indices, ]
        fake_y = x.copy()
        fake_y.obs[self.pert_col] = perturbation

        if self.mode == "gene":
            fake_y.obs['dose_val'] = fake_y.obs[self.pert_col].apply(
                lambda x: '1+1' if len(x.split('+')) == 2 else '1'
            )
            fake_y.obs[self.condition_name] = fake_y.obs.apply(
                lambda x: '_'.join([
                    x[self.covariate], x[self.pert_col], x['dose_val']
                ]), axis = 1
            )
        elif self.mode == "compound":
            fake_y.obs[self.condition_name] = fake_y.obs.apply(
                lambda x: '_'.join([
                    x[self.covariate], x[self.pert_col]
                ]), axis = 1
            )

        return PertDataset(x, fake_y, self.vocab)

    def get_dataloader_for_prediction(
            self,
            test_batch_size: int,
            perturbation: str,
            pool_size: Optional[int] = None,
        ) -> DataLoader:
        return DataLoader(
            self._create_dataset_for_prediction(perturbation, pool_size),
            batch_size=test_batch_size, shuffle=False,
        )

    def get_pert_flags(self, condition: str) -> Optional[np.ndarray[int]]:
        """
        Generate an integer array as pert flags for the given condition.

        Args:
            condition (str): a condition string, e.g. 'FOXA3+ctrl'.

        Returns:
            np.ndarray[int] | None
        """
        genes = self.adata.var[self.gene_col].tolist()
        perts = [g for g in condition.split('+') if g != self.ctrl_str]
        #TODO: make perts always a subset of genes
        if not set(perts).issubset(genes):
            return None
        elif not len(perts):
            return np.full(len(genes), self.pert_pad_id, dtype=int)
        else:
            return np.where(np.isin(genes, perts),
                            self.crispr_flag, self.ctrl_flag)

    def tokenize_and_pad_batch(
            self,
            data: torch.Tensor,
            gene_ids: torch.Tensor,
            max_len: int,
            append_cls: bool = True,
            include_zero_gene: bool = True,
        ) -> dict[str, torch.Tensor]:
        """
        Tokenize and pad the given genes and expression values.

        Args:
            data (torch.Tensor):
                expression values
            gene_ids (torch.Tensor):
                integer gene ids
            max_len (int):
                the maximum length of output columns
            append_cls (bool):
                whether to append cell embedding (default: True)
            include_zero_gene (bool):
                whether to include zero genes (default: True)

        Returns:
            dict[str, torch.Tensor]
        """
        gene_ids = gene_ids.long()
        if len(gene_ids.shape) == 2:
            gene_ids = gene_ids[0]

        if data.shape[-1] != len(gene_ids):
            raise ValueError(
                f"The number of features in data ({data.shape[-1]}) does "
                f"not match the number of gene_ids ({gene_ids.shape[-1]})."
            )

        cls_id = self.vocab[self.cls_token]
        pad_id = self.vocab[self.pad_token]
        pad_value = self.pad_value

        genes_list = []
        values_list = []
        for i in range(data.shape[0]):

            if include_zero_gene:
                genes = gene_ids
                values = data[i]
            else:
                idx = data[i].nonzero().squeeze()
                genes = gene_ids[idx]
                values = data[i][idx]

            if append_cls:
                genes = torch.cat([torch.tensor([cls_id]), genes])
                values = torch.cat([torch.tensor([0]), values])

            if len(genes) > max_len:
                if append_cls:
                    idx = torch.randperm(len(genes) - 1)[:max_len - 1]
                    idx = torch.cat([torch.tensor([0]), idx + 1])
                else:
                    idx = torch.randperm(len(genes))[:max_len]
                genes = genes[idx]
                values = values[idx]

            if len(genes) < max_len:
                genes = torch.cat([
                    genes,
                    torch.full(
                        (max_len - len(genes),), pad_id, dtype=genes.dtype
                    ),
                ])
                values = torch.cat([
                    values,
                    torch.full(
                        (max_len - len(values),), pad_value, dtype=values.dtype
                    ),
                ])

            genes_list.append(genes)
            values_list.append(values)

        return {
            "genes": torch.stack(genes_list, dim=0),
            "values": torch.stack(values_list, dim=0),
        }

    def get_tokenized_batch(
            self,
            batch_data: dict,
            max_len: int = 0,
            append_cls: bool = True,
            include_zero_gene: str|bool = True,
        ) -> dict[str, torch.Tensor]:
        """
        Tokenize and pad a batch of data.

        Args:
            batch_data (dict):
                a batch of data
            max_len (int):
                the maximum length of output columns (default: None)
            append_cls (bool):
                whether to append cell embedding (default: True)
            include_zero_gene (bool):
                whether to include zero genes (default: True)

        Returns:
            dict[str, torch.Tensor]:
                for gene perturbations, the keys will be:
                    ["genes", "pert_flags", "pert", "values", "target_values", "de_idx"]
                    or ["genes", "pert_flags", "pert", "values", "target_values"]
                for compound perturbations, the keys will be:
                    ["genes", "pert", "values", "target_values", "de_idx"]
                    or ["genes", "pert", "values", "target_values"]
        """
        #TODO: allow include_zero_gene = False
        batch_size: int = batch_data['y']['X'].shape[0]
        n_genes: int = batch_data['y']['X'].shape[1]
        gene_ids = batch_data['gene_ids'][0]
        if max_len <= 0:
            max_len: int = n_genes + int(append_cls)

        if n_genes + int(append_cls) > max_len:
            return_de_idx: bool = False
            input_gene_idx = torch.randperm(n_genes)[
                :max_len - int(append_cls)
            ]
        else:
            return_de_idx: bool = True
            input_gene_idx = torch.arange(n_genes)

        if self.mode == "gene":
            pert_cond_list = []
            pert_list = []
            row_idx = []
            for idx, cond in enumerate(batch_data['y']['obs'][self.pert_col]):
                flags = self.get_pert_flags(cond)
                if flags is not None:
                # Do not write as "if flags:", given that it's an array
                    pert_cond_list.append(cond)
                    pert_list.append(flags)
                    row_idx.append(idx)

            if len(pert_list):
                pert_tensor = torch.from_numpy(np.stack(pert_list, axis=0))
            else:
                return {}

            #TODO: pert flag for cell embedding in self.tokenize_and_pad_batch()
            pert = self.tokenize_and_pad_batch(
                pert_tensor[:, input_gene_idx],
                gene_ids[input_gene_idx],
                max_len=max_len,
                append_cls=append_cls,
                include_zero_gene=include_zero_gene,
            )
            pert['pert_flags'] = pert.pop('values', None)
            pert['pert'] = pert_cond_list

        elif self.mode == "compound":
            pert = {'pert': batch_data['y']['obs'][self.pert_col]}
            row_idx = list(range(batch_size))

        x = self.tokenize_and_pad_batch(
            batch_data['x']['X'].view(batch_size, n_genes
                )[row_idx][:, input_gene_idx],
            gene_ids[input_gene_idx],
            max_len=max_len,
            append_cls=append_cls,
            include_zero_gene=include_zero_gene,
        )
        x['values'] = x['values'].float()
        y = self.tokenize_and_pad_batch(
            batch_data['y']['X'].view(batch_size, n_genes
                )[row_idx][:, input_gene_idx],
            gene_ids[input_gene_idx],
            max_len=max_len,
            append_cls=append_cls,
            include_zero_gene=include_zero_gene,
        )
        y['target_values'] = y.pop('values').float()
        if return_de_idx:
            de = {'de_idx': batch_data['de_idx'][row_idx] + int(append_cls)}
        else:
            de = {}
        return pert|x|y|de


class DataSplitter:

    split_types = ['unseen', 'shuffle']

    def __init__(
            self,
            pert_data: PertData,
            save_dir: Optional[Path] = None,
            seed: Optional[int] = None,
        ) -> None:
        self.adata = pert_data.adata
        self.pert_col = pert_data.pert_col
        self.ctrl_str = pert_data.ctrl_str
        self.keep_ctrl = pert_data.keep_ctrl

        if save_dir is None:
            self.save_dir = pert_data.dataset_path
        else:
            self.save_dir = Path(save_dir)

        if seed is None:
            self.seed = pert_data.seed
        else:
            self.seed = int(seed)

    def get_save_file(self) -> Path:
        return self.save_dir / 'train_test_split.json'

    def prepare_split(
            self,
            split_type: str = 'unseen',
            test_perts: Optional[list] = None,
            val_perts: Optional[list] = None,
            test_size: float = 0.1,
            val_size: float = 0.1,
        ) -> None:
        """
        Prepare splits of train, val, and test perturbations.

        Returns:
            None
        """
        if split_type not in self.split_types:
            raise ValueError(f"Invalid split_type: {repr(split_type)}!")

        if split_type == 'unseen':
            perts = self.adata.obs[self.pert_col].unique()
            if not self.keep_ctrl:
                perts = np.setdiff1d(perts, self.ctrl_str)

            perts_no_ctrl = np.setdiff1d(perts, self.ctrl_str)
            test_not_given = (
                test_perts is None
                or len(np.intersect1d(test_perts, perts)) <= 0
            )
            val_not_given = (
                val_perts is None
                or len(np.intersect1d(val_perts, perts)) <= 0
            )
            np.random.seed(self.seed)
            if test_not_given and val_not_given:
                num_test = round(test_size * len(perts_no_ctrl))
                num_val = round(val_size * len(perts_no_ctrl))
                perm = np.random.permutation(perts_no_ctrl)
                test_perts = perm[0:num_test]
                val_perts = perm[num_test:num_test+num_val]
            elif test_not_given:
                num_test = round(test_size * len(perts_no_ctrl))
                test_perts = np.random.permutation(
                    np.setdiff1d(perts_no_ctrl, val_perts)
                )[0:num_test]
            elif val_not_given:
                num_val = round(val_size * len(perts_no_ctrl))
                val_perts = np.random.permutation(
                    np.setdiff1d(perts_no_ctrl, test_perts)
                )[0:num_val]

            train_perts = np.setdiff1d(perts, np.union1d(test_perts, val_perts))
            write_json(
                {
                    'type': split_type,
                    'train': train_perts,
                    'val': val_perts,
                    'test': test_perts,
                },
                self.get_save_file()
            )
        elif split_type == 'shuffle':
            write_json(
                {
                    'type': split_type,
                    'keep_ctrl': self.keep_ctrl,
                    'val_size': val_size,
                    'test_size': test_size,
                },
                self.get_save_file()
            )

    def split_data(self) -> dict:
        try:
            split = read_json(self.get_save_file())
        except OSError:
            split = None

        if split is None:
            adata = self.adata if self.keep_ctrl else self.adata[
                self.adata.obs[self.pert_col] != self.ctrl_str
            ]
            train_data, test_data = train_test_split(
                adata, test_size=0.1, shuffle=True,
            )
            train_data, val_data = train_test_split(
                train_data, test_size=0.1, shuffle=True,
            )
        elif split['type'] == 'shuffle':
            adata = self.adata if split['keep_ctrl'] else self.adata[
                self.adata.obs[self.pert_col] != self.ctrl_str
            ]
            train_data, test_data = train_test_split(
                adata, test_size=split['test_size'], shuffle=True,
            )
            train_data, val_data = train_test_split(
                train_data, test_size=split['val_size'], shuffle=True,
            )
        elif split['type'] == 'unseen':
            train_perts, val_perts, test_perts = (
                split['train'], split['val'], split['test']
            )
            # Are shuffles needed for train and val sets here?
            train_data, val_data, test_data = (
                self.adata[self.adata.obs[self.pert_col].isin(train_perts)],
                self.adata[self.adata.obs[self.pert_col].isin(val_perts)],
                self.adata[self.adata.obs[self.pert_col].isin(test_perts)],
            )
            
        return {'train': train_data, 'val': val_data, 'test': test_data}

