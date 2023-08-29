"""Modified from the scGPT/scgpt/preprocess.py file."""

__all__ = ['Preprocessor']

import numpy as np
import scanpy as sc
import anndata as ad
from scipy import sparse
from typing import Optional
from .utils import create_logger

class Preprocessor:
    """
    Prepare data for later use:
      - Filter genes/cells
      - Normalize raw values
      - Log-transform
      - HVG selection
      - Binning.
    """

    def __init__(
        self,
        use_key: Optional[str] = None,
        filter_gene_by_counts: int | bool = False,
        filter_cell_by_counts: int | bool = False,
        normalize_total: float | bool = False,
        result_normed_key: Optional[str] = "X_normed",
        log1p: bool = False,
        result_log1p_key: str = "X_log1p",
        subset_hvg: int | bool = False,
        hvg_use_key: Optional[str] = None,
        hvg_flavor: str = "seurat_v3",
        binning: int | bool = False,
        result_binned_key: str = "X_binned",
    ) -> None:
        r"""
        Set up the preprocessor, use the args to config the workflow steps.

        Args:

        use_key (:class:`str`, optional):
            The key of :class:`~anndata.AnnData` to use for preprocessing.
        filter_gene_by_counts (:class:`int` or :class:`bool`, default: ``False``):
            Whther to filter genes by counts, if :class:`int`, filter genes with counts
        filter_cell_by_counts (:class:`int` or :class:`bool`, default: ``False``):
            Whther to filter cells by counts, if :class:`int`, filter cells with counts
        normalize_total (:class:`float` or :class:`bool`, default: ``False``):
            Whether to normalize the total counts of each cell to a specific value.
        result_normed_key (:class:`str`, default: ``"X_normed"``):
            The key of :class:`~anndata.AnnData` to store the normalized data. If
            :class:`None`, will use normed data to replce the :attr:`use_key`.
        log1p (:class:`bool`, default: ``False``):
            Whether to apply log1p transform to the normalized data.
        result_log1p_key (:class:`str`, default: ``"X_log1p"``):
            The key of :class:`~anndata.AnnData` to store the log1p transformed data.
        subset_hvg (:class:`int` or :class:`bool`, default: ``False``):
            Whether to subset highly variable genes.
        hvg_use_key (:class:`str`, optional):
            The key of :class:`~anndata.AnnData` to use for calculating highly variable
            genes. If :class:`None`, will use :attr:`adata.X`.
        hvg_flavor (:class:`str`, default: ``"seurat_v3"``):
            The flavor of highly variable genes selection. See
            :func:`scanpy.pp.highly_variable_genes` for more details.
        binning (:class:`int` or :class:`bool`, default: ``False``):
            Whether to bin the data into discrete values of number of bins provided.
        result_binned_key (:class:`str`, default: ``"X_binned"``):
            The key of :class:`~anndata.AnnData` to store the binned data.
        """
        self.logger = create_logger(name=self.__class__.__name__)
        self.use_key = use_key
        self.filter_gene_by_counts = filter_gene_by_counts
        self.filter_cell_by_counts = filter_cell_by_counts
        self.normalize_total = normalize_total
        self.result_normed_key = result_normed_key
        self.log1p = log1p
        self.result_log1p_key = result_log1p_key
        self.subset_hvg = subset_hvg
        self.hvg_use_key = hvg_use_key
        self.hvg_flavor = hvg_flavor
        self.binning = binning
        self.result_binned_key = result_binned_key

    def __call__(
            self, adata: ad.AnnData, batch_key: Optional[str] = None
        ) -> str:
        r"""
        format controls the different input value wrapping,
        including categorical binned style, fixed-sum normalized counts,
        log1p fixed-sum normalized counts, etc.

        Args:

        adata (:class:`AnnData`):
            The :class:`AnnData` object to preprocess.
        batch_key (:class:`str`, optional):
            The key of :class:`AnnData.obs` to use for batch information.
            This arg is used in the highly variable gene selection step.
        """
        key_to_process = self.use_key
        if key_to_process == "X":
            key_to_process = None  # the scanpy APIs use None for "X"

        # preliminary checks, will use later
        is_logged = self.check_logged(adata, obs_key=key_to_process)

        # step 1: filter genes
        if self.filter_gene_by_counts:
            self.logger.info("Filtering genes by counts...")
            sc.pp.filter_genes(
                adata,
                min_counts=self.filter_gene_by_counts
                    if isinstance(self.filter_gene_by_counts, int) else None,
            )

        # step 2: filter cells
        if self.filter_cell_by_counts:
            self.logger.info("Filtering cells by counts...")
            sc.pp.filter_cells(
                adata,
                min_counts=self.filter_cell_by_counts
                    if isinstance(self.filter_cell_by_counts, int) else None,
            )

        # step 3: normalize total
        if self.normalize_total:
            self.logger.info("Normalizing total counts...")
            normed_ = sc.pp.normalize_total(
                adata,
                target_sum=self.normalize_total
                    if isinstance(self.normalize_total, float) else None,
                layer=key_to_process,
                inplace=False,
            )["X"]
            key_to_process = self.result_normed_key or key_to_process
            sc.get._set_obs_rep(adata, normed_, layer=key_to_process)

        # step 4: log1p
        if self.log1p:
            self.logger.info("Log1p transforming...")
            if is_logged:
                self.logger.warning(
                    "The input data seems to be already log1p transformed. "
                    "Set `log1p=False` to avoid double log1p transform."
                )
            if self.result_log1p_key:
                sc.get._set_obs_rep(
                    adata,
                    sc.get._get_obs_rep(adata, layer=key_to_process),
                    layer=self.result_log1p_key,
                )
                key_to_process = self.result_log1p_key
            sc.pp.log1p(adata, layer=key_to_process)

        # step 5: subset hvg
        if self.subset_hvg:
            self.logger.info("Subsetting highly variable genes...")
            if batch_key is None:
                self.logger.warning(
                    "No batch_key given. Use all cells for HVG selection."
                )
            sc.pp.highly_variable_genes(
                adata,
                layer=self.hvg_use_key,
                n_top_genes=self.subset_hvg
                    if isinstance(self.subset_hvg, int) else None,
                batch_key=batch_key,
                flavor=self.hvg_flavor,
                subset=True,
            )

        # step 6: binning
        if self.binning:
            self.logger.info("Binning data...")
            if not isinstance(self.binning, int):
                raise ValueError(
                    f"Binning arg must be an integer, but got {self.binning}."
                )
            n_bins = self.binning  # NOTE: the first bin is always for zero
            binned_rows = []
            bin_edges = []
            layer_data = sc.get._get_obs_rep(adata, layer=key_to_process)
            layer_data = (layer_data.A if sparse.issparse(layer_data)
                                       else layer_data)
            for row in layer_data:
                non_zero_ids = row.nonzero()
                non_zero_row = row[non_zero_ids]
                bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins-1))
                # bins = np.sort(np.unique(bins))
                # NOTE: comment this line for now, since this will make the
                # each category has different relative meaning across datasets
                non_zero_digits = self._digitize(non_zero_row, bins)
                assert non_zero_digits.min() >= 1
                assert non_zero_digits.max() <= n_bins - 1
                binned_row = np.zeros_like(row, dtype=np.int64)
                binned_row[non_zero_ids] = non_zero_digits
                binned_rows.append(binned_row)
                bin_edges.append(np.concatenate([[0], bins]))

            key_to_process = self.result_binned_key
            adata.layers[key_to_process] = np.stack(binned_rows)
            adata.obsm["bin_edges"] = np.stack(bin_edges)

        return key_to_process

    def _digitize(self, x: np.ndarray, bins: np.ndarray) -> np.ndarray:
        r"""
        Digitize the data into bins. This method spreads data
        uniformly when bins have same values.

        Args:

        x (:class:`np.ndarray`):
            The data to digitize.
        bins (:class:`np.ndarray`):
            The bins to use for digitization, in increasing order.

        Returns:

        :class:`np.ndarray`:
            The digitized data.
        """
        assert x.ndim == 1 and bins.ndim == 1

        left_digits = np.digitize(x, bins)
        right_difits = np.digitize(x, bins, right=True)
        rands = np.random.rand(len(x))  # uniform random numbers

        digits = rands * (right_difits - left_digits) + left_digits
        digits = np.ceil(digits).astype(np.int64)
        return digits

    @staticmethod
    def check_logged(adata: ad.AnnData, obs_key: Optional[str] = None) -> bool:
        r"""
        Check if the data is already log1p transformed.

        Args:

        adata (:class:`AnnData`):
            The :class:`AnnData` object to preprocess.
        obs_key (:class:`str`, optional):
            The key of :class:`AnnData.obs` to use for batch information.
            This arg is used in the highly variable gene selection step.
        """
        data = sc.get._get_obs_rep(adata, layer=obs_key)
        max_, min_ = data.max(), data.min()
        if max_ > 30:
            return False
        if min_ < 0:
            return False

        non_zero_min = data[data > 0].min()
        if non_zero_min >= 1:
            return False

        return True

