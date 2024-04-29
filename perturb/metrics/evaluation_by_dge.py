from pathlib import Path
import anndata as ad
import scanpy as sc
import numpy as np
import pandas as pd
import plotnine as pn


class TestEvaluation:
    DE_key = 'rank_genes_groups_all'
    gene_colname = 'gene'
    cond_colname = 'condition'
    ctrl_condition = 'ctrl'

    def __init__(
        self,
        workdir: str|Path,
        savedir: str|Path = '.',
        mode: str = 'gene',
        by: str|list = ['cell_type', 'condition'],
        sample_n: int = 200,
        sum_method: str = 'median',
    ) -> None:
        self.path = Path(workdir)
        self.save = Path(savedir)
        self.save.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.by = by
        self.sample_n = sample_n
        self.sum_method = sum_method
        self.have_random_sample = False

    def load(self) -> None:
        self.ctrl = sc.read(self.path / 'h5ad/ctrl.h5ad')
        self.pert = sc.read(self.path / 'h5ad/testtruth.h5ad')
        self.pred = sc.read(self.path / 'h5ad/scGPTpred.h5ad')
        self.ctrl.var.set_index('gene_name', drop=False, inplace=True)
        self.pert.var.set_index('gene_name', drop=False, inplace=True)
        self.pred.var.set_index('gene_name', drop=False, inplace=True)

    @staticmethod
    def create_name_cond_map(
        adata: ad.AnnData,
        col1: str = 'sm_name',
        col2: str = 'condition',
    ) -> pd.DataFrame:
        return adata.obs[[col1, col2]].drop_duplicates().reset_index(drop=True)

    @staticmethod
    def sample_from_adata(
        adata: ad.AnnData,
        by: str|list,
        n: int,
        replace: bool = False,
        seed: int|None = 42,
    ) -> ad.AnnData:
        obs = adata.obs.reset_index()
        # counts = obs.groupby(by=by, observed=True).size()
        # groups = counts[counts >= sample_n].reset_index()
        if not replace:
            obs = obs.groupby(by=by, observed=True
                ).filter(lambda x: len(x) >= n)
        new = obs.groupby(by=by, observed=True
            ).sample(n=n, replace=replace, random_state=seed)
        return adata[new.index]

    @staticmethod
    def summary_expr_values(
        adata: ad.AnnData,
        by: str|list,
        value_col: str = 'expr',
        var_col: str = 'gene',
        funcname: str = 'median',
    ) -> pd.DataFrame:
        def create_one_df(
            key: str|tuple,
            idx: pd.Index
        ) -> pd.DataFrame:
            G = pd.DataFrame({
                var_col: adata.var.gene_name,
                value_col: getattr(np, funcname)(adata[idx].X, axis=0)
            })
            if isinstance(by, str):
                C = pd.DataFrame([{by: key}])
            elif isinstance(by, list) and len(by) == 1:
                C = pd.DataFrame([{by[0]: key}])
            else:
                C = pd.DataFrame([dict(zip(by, key))])
            return pd.concat([pd.concat([C] * len(G)).reset_index(drop=True),
                              G.reset_index(drop=True)], axis=1)
        
        grp = adata.obs.reset_index().groupby(by=by, observed=True)
        return pd.concat(
            [create_one_df(key, idx) for key, idx in grp.groups.items()],
            ignore_index=True
        )

    @staticmethod
    def rank_genes(
        adata: ad.AnnData,
        groupby: str,
        ref: str = 'ctrl'
    ) -> ad.AnnData:
        covariates = adata.obs['cell_type'].unique()
        de_genes = {}
        for cov in covariates:
            adata_sub = adata[adata.obs['cell_type'] == cov]
            sc.tl.rank_genes_groups(
                adata_sub,
                groupby=groupby,
                reference=ref,
                rankby_abs=True,
                n_genes=len(adata.var),
                method="wilcoxon",
                use_raw=False,
                layer=None,
            )
            de_genes[cov] = adata_sub.uns['rank_genes_groups']
        adata.uns[TestEvaluation.DE_key] = de_genes
        return adata

    @staticmethod
    def get_logFC(
        adata: ad.AnnData,
        var_name: str,
        value_name: str,
    ) -> pd.DataFrame:
        de_key = TestEvaluation.DE_key
        gene_col = TestEvaluation.gene_colname
        if de_key not in adata.uns:
            raise ValueError('No DE results!')
        
        def from_one_celltype(celltype: str):
            df = pd.DataFrame(adata.uns[de_key][celltype]['logfoldchanges'])
            df[gene_col] = adata.var.index
            df['cell_type'] = celltype
            return pd.melt(df, id_vars=['cell_type', gene_col],
                           var_name=var_name, value_name=value_name)
        return pd.concat(
            [from_one_celltype(ct) for ct in adata.uns[de_key]],
            ignore_index=True
        )

    @staticmethod
    def compute_correlation(
        df: pd.DataFrame,
        by: str|list = ['cell_type', 'condition'],
        sum_method: str = 'median',
    ) -> pd.DataFrame:
        def create_one_df(key, idx):
            if isinstance(by, str):
                A = pd.DataFrame([{by: key}])
            elif isinstance(by, list) and len(by) == 1:
                A = pd.DataFrame([{by[0]: key}])
            else:
                A = pd.DataFrame([dict(zip(by, key))])
            sub = df.iloc[idx].reset_index(drop=True)
            expr_corr = sub['pert'+suffix].corr(sub['pred'+suffix])
            logFC_corr = sub['pert_logFC'].corr(sub['pred_logFC'])
            B = pd.DataFrame([{'expr_corr': expr_corr, 'logFC_corr': logFC_corr}])
            return pd.concat([A, B], axis=1)

        suffix = '_' + sum_method
        grp = df.groupby(by=by, observed=True)
        return pd.concat(
            [create_one_df(key, idx) for key, idx in grp.groups.items()],
            ignore_index=True
        )

    @staticmethod
    def pred_vs_actual_plot(
        data: pd.DataFrame,
        actual: str,
        pred: str,
        facets: str|list|tuple,
        alpha: float = 0.1,
        title: str = "Mean Expression Values of Each Gene",
        xlab: str = "Truth",
        ylab: str = "Prediction",
    ) -> pn.ggplot:
        return (
            pn.ggplot(data, pn.aes(actual, pred))
            + pn.geom_point(alpha=alpha)
            + pn.geom_abline(intercept=0, slope=1, color="steelblue")
            + pn.coord_fixed(ratio=1)
            + pn.facet_wrap(facets=facets)
            + pn.labs(x=xlab, y=ylab, title=title)
        )

    def sample_data(self, *args, **kwargs) -> None:
        if not self.have_random_sample:
            self.pert_sample = self.sample_from_adata(self.pert, *args, **kwargs)
            self.pred_sample = self.sample_from_adata(self.pred, *args, **kwargs)
            self.have_random_sample = True

    def merge_data(self) -> pd.DataFrame:
        gene_col = self.gene_colname
        cond_col = self.cond_colname
        suffix = '_' + self.sum_method
        self.sample_data(self.by, self.sample_n)
        summary_ctrl = self.summary_expr_values(
            self.ctrl, self.by, 'ctrl'+suffix, gene_col, self.sum_method)
        summary_pert = self.summary_expr_values(
            self.pert_sample, self.by, 'pert'+suffix, gene_col, self.sum_method)
        summary_pred = self.summary_expr_values(
            self.pred_sample, self.by, 'pred'+suffix, gene_col, self.sum_method)
        return summary_ctrl[['cell_type', gene_col, 'ctrl'+suffix]
            ].merge(summary_pert, on = ['cell_type', gene_col]
            ).merge(summary_pred, on = ['cell_type', cond_col, gene_col]
            ).sort_values(self.by
            ).reset_index(drop=True)

    def add_logFC_to_data(self) -> pd.DataFrame:
        cond_col = [i for i in self.by if i != 'cell_type'][0]
        self.sample_data(self.by, self.sample_n)
        ctrl_pert = self.get_logFC(
            self.rank_genes(
                ad.concat([self.ctrl, self.pert_sample]),
                groupby=cond_col,
            ),
            var_name=cond_col,
            value_name='pert_logFC',
        )
        ctrl_pred = self.get_logFC(
            self.rank_genes(
                ad.concat([self.ctrl, self.pred_sample]),
                groupby=cond_col,
            ),
            var_name=cond_col,
            value_name='pred_logFC',
        )
        return self.data.merge(ctrl_pert).merge(ctrl_pred)
    
    def modify_corr_for_plot(self) -> None:
        self.corr['expr_corr_label'] = self.corr['expr_corr'].apply(
            lambda x: 'R = {:.4f}'.format(x))
        self.corr['logFC_corr_label'] = self.corr['logFC_corr'].apply(
            lambda x: 'R = {:.4f}'.format(x))
        
    def plot(self) -> None:
        facets = 'sm_name' if self.mode == 'drug' else 'condition'
        suffix = '_' + self.sum_method
        (
            self.pred_vs_actual_plot(
                self.data, "pert"+suffix, "pred"+suffix, facets=facets,
                title='{} Expression Values of Each Gene'.format(
                    self.sum_method.title()),
                )
            + pn.geom_text(pn.aes(label="expr_corr_label"), data=self.corr, x=3, y=6)
            + pn.theme_bw(base_size=25)
        ).save(
            self.save / 'expr_values.png', width=25, height=25
        )
        (
            self.pred_vs_actual_plot(
                self.data, "pert_logFC", "pred_logFC", facets=facets,
                title='Log2 of Fold Changes of Each Gene',
                )
            + pn.geom_text(pn.aes(label="logFC_corr_label"), data=self.corr, x=0, y=4)
            + pn.lims(x=(-5, 5), y=(-5, 5))
            + pn.theme_bw(base_size=25)
        ).save(
            self.save / 'logFC.png', width=25, height=25
        )

    def run(self):
        self.load()
        self.sample_data(by=self.by, n=200, seed=42)
        self.data = self.merge_data()
        self.data = self.add_logFC_to_data()
        self.corr = self.compute_correlation(self.data, self.by, self.sum_method)
        if self.mode in ('drug',):
            name_cond = self.create_name_cond_map(self.pert
                ).sort_values(by='sm_name')
            self.data = name_cond.merge(self.data)
            self.corr = name_cond.merge(self.corr)
        self.data.to_csv(self.save / "ctrl_pert_pred.csv", index=False)
        self.corr.to_csv(self.save / "corr.csv", index=False)
        self.modify_corr_for_plot()
        self.plot()

