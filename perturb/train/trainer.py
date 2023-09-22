__all__ = ['Trainer']

import copy
import gc
import time
import warnings
from typing import Optional
from pathlib import Path

import torch
import wandb
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error as mse
from torch import nn

from ..loss import masked_mse_loss, masked_relative_error
from ..metrics import plot_corr_matrix
from ..utils import create_logger, to_numpy, write_json, write_pickle
from ..utils.attr_dict import AttrDict

warnings.filterwarnings("ignore")

def define_wandb_metrics():
    wandb.define_metric("valid/mse", summary="min", step_metric="epoch")
    wandb.define_metric("valid/mre", summary="min", step_metric="epoch")
    wandb.define_metric("test/corr", summary="max")

class Trainer:
    def __init__(self, config: dict) -> None:
        self.config = AttrDict(**config)
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else 'cpu'
        )
        self.epochs = self.config.epochs
        self.logger = create_logger(name=self.config.project)
        self.run = wandb.init(
            config=self.config,
            project=self.config.project,
            reinit=True,
            settings=wandb.Settings(start_method="fork"),
        )
        self.save_dir = Path(
            "save"
            f"/{self.config.project}"
            f"-{Path(self.config.dataset).name}"
            f"-{time.strftime('%b%d-%H%M%S')}"
        )
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def prepare_data(self, data) -> None:
        self.data = data
        self.train_loader = data.dataloader["train_loader"]
        self.val_loader = data.dataloader.get("val_loader")
        self.test_loader = data.dataloader.get("test_loader")
        self.num_train_batches = len(self.train_loader)
        self.num_val_batches = (0 if self.val_loader is None
                                  else len(self.val_loader))
        self.num_test_batches = (0 if self.test_loader is None
                                   else len(self.test_loader))

    def prepare_model(self, model: nn.Module) -> None:
        model.trainer = self
        self.model = model
        self.model.to(self.device)
        wandb.watch(self.model)
        self.criterion = masked_mse_loss
        self.criterion_cls = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.lr
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, self.config.schedule_interval,
            gamma=self.config.schedule_ratio
        )
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.config.amp
        )

    def prepare_batch(self, *args, **kwargs) -> dict:
        return self.data.get_tokenized_batch(*args, **kwargs)

    def fit_epoch(self, epoch: int) -> None:
        self.model.train()
        device = self.device
        log_interval = self.config.log_interval
        # input_zero_ratios = []
        # target_zero_ratios = []
        total_loss = 0.0
        total_mse = 0.0
        total_gepc = 0.0
        total_error = 0.0
        start_time = time.time()

        for idx, batch_data in enumerate(self.train_loader):
            tokenized_batch = self.prepare_batch(
                batch_data, max_len=self.config.max_seq_len,
                append_cls=self.config.GEPC,
            )
            if not tokenized_batch:
                continue

            input_gene_ids = tokenized_batch['genes'].to(device)
            input_values = tokenized_batch['values'].to(device)
            target_values = tokenized_batch['target_values'].to(device)
            if self.data.mode == "gene":
                input_pert = tokenized_batch['pert_flags'].to(device)
            elif self.data.mode == "compound":
                input_pert = tokenized_batch['pert']

            src_key_padding_mask = input_gene_ids.eq(
                self.data.vocab[self.data.pad_token]
            )
            with torch.cuda.amp.autocast(enabled=self.config.amp):
                output_dict = self.model(
                    input_gene_ids,
                    input_values,
                    input_pert,
                    src_key_padding_mask=src_key_padding_mask,
                    CLS=self.config.CLS,
                    CCE=self.config.CCE,
                    MVC=self.config.GEPC,
                    ECS=self.config.ECS,
                )
                masked_positions = torch.ones_like(input_values, dtype=bool)  # Use all
                loss = loss_mse = self.criterion(
                    output_dict["mlm_output"],
                    target_values,
                    masked_positions
                )
                metrics_to_log = {"train/mse": loss_mse.item()}
                if self.config.GEPC:
                    loss_gepc = self.criterion(
                        output_dict["mvc_output"],
                        target_values,
                        masked_positions
                    )
                    loss = loss + loss_gepc
                    metrics_to_log.update({"train/mvc": loss_gepc.item()})
    
            self.model.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)

            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("always")
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=1.0,
                    error_if_nonfinite=False if self.scaler.is_enabled()
                                             else True,
                )
                if len(w) > 0:
                    self.logger.warning(
                        f"Found infinite gradient. This may be caused by "
                        f"the gradient scaler. The current scale is "
                        f"{self.scaler.get_scale()}. This warning "
                        f"can be ignored if no longer occurs "
                        f"after autoscaling of the scaler."
                    )

            self.scaler.step(self.optimizer)
            self.scaler.update()
            wandb.log(metrics_to_log)
    
            with torch.no_grad():
                mre = masked_relative_error(
                    output_dict["mlm_output"],
                    target_values,
                    masked_positions
                )

            total_loss += loss.item()
            total_mse += loss_mse.item()
            total_gepc += loss_gepc.item() if self.config.GEPC else 0.0
            total_error += mre.item()
            if idx % log_interval == 0 and idx > 0:
                lr = self.scheduler.get_last_lr()[0]
                ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                cur_loss = total_loss / log_interval
                cur_mse = total_mse / log_interval
                cur_gepc = total_gepc / log_interval if self.config.GEPC else 0.0
                cur_error = total_error / log_interval
                # ppl = math.exp(cur_loss)
                self.logger.info(
                    f"| epoch {epoch:3d} |"
                    f" {idx:3d}/{self.num_train_batches:3d} batches |"
                    f" lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} |"
                    f" loss {cur_loss:5.2f} | mse {cur_mse:5.2f} |"
                    f" mre {cur_error:5.2f} |"
                    + (f" gepc {cur_gepc:5.2f} |" if self.config.GEPC else "")
                )
                total_loss = 0.0
                total_mse = 0.0
                total_gepc = 0.0
                total_error = 0.0
                start_time = time.time()

    def evaluate_epoch(self, epoch: int) -> tuple[float]:
        self.model.eval()
        device = self.device
        total_loss = 0.0
        total_error = 0.0
        total_num = 0
        with torch.no_grad():
            for batch_data in self.val_loader:
                tokenized_batch = self.prepare_batch(
                    batch_data, max_len=0,
                    append_cls=self.config.GEPC,
                )
                if not tokenized_batch:
                    continue

                input_gene_ids = tokenized_batch['genes'].to(device)
                input_values = tokenized_batch['values'].to(device)
                target_values = tokenized_batch['target_values'].to(device)
                if self.data.mode == "gene":
                    input_pert = tokenized_batch['pert_flags'].to(device)
                elif self.data.mode == "compound":
                    input_pert = tokenized_batch['pert']

                src_key_padding_mask = input_gene_ids.eq(
                    self.data.vocab[self.data.pad_token]
                )
                with torch.cuda.amp.autocast(enabled=self.config.amp):
                    output_dict = self.model(
                        input_gene_ids,
                        input_values,
                        input_pert,
                        src_key_padding_mask=src_key_padding_mask,
                        CLS=self.config.CLS,
                        CCE=self.config.CCE,
                        MVC=self.config.GEPC,
                        ECS=self.config.ECS,
                    )
                    masked_positions = torch.ones_like(input_values, dtype=bool)  # Use all
                    loss = self.criterion(
                        output_dict["mlm_output"],
                        target_values,
                        masked_positions
                    )
    
                total_loss += loss.item() * len(input_gene_ids)
                mre = masked_relative_error(
                    output_dict["mlm_output"],
                    target_values,
                    masked_positions
                )
                total_error += mre.item() * len(input_gene_ids)
                total_num += len(input_gene_ids)

        wandb.log({
            "valid/mse": total_loss / total_num,
            "valid/mre": total_error / total_num,
            "epoch": epoch,
        })
        return total_loss / total_num, total_error / total_num

    def fit(self) -> None:
        self.best_model = self.model
        self.best_model_epoch = 0
        best_val_loss = float("inf")
        define_wandb_metrics()
        for epoch in range(1, self.epochs + 1):
            epoch_start_time = time.time()
            if self.config.do_train: self.fit_epoch(epoch)
            val_loss, val_mre = self.evaluate_epoch(epoch)
            
            elapsed = time.time() - epoch_start_time
            self.logger.info("-" * 85)
            self.logger.info(
                f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
                f"valid loss/mse {val_loss:5.4f} | mre {val_mre:5.4f}"
            )
            self.logger.info("-" * 85)
        
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_model = copy.deepcopy(self.model)
                self.best_model_epoch = epoch
                self.logger.info(f"Best model with score {best_val_loss:5.4f}")

            self.scheduler.step()

    def predict_batch(
            self, batch_data, include_zero_gene: str|bool = True,
        ) -> dict[str, torch.Tensor]:
        """
        Predict a batch of data with the best model.

        Args:
            batch_data: a dictionary of input data with keys.

        Returns:
            An empty dict,
            or a dict with the following keys:
                genes, de_idx, values, target, pred.
        """
        self.model.eval()
        device = self.device
        tokenized_batch = self.prepare_batch(
            batch_data,
            max_len=0,
            append_cls=self.config.GEPC,
            include_zero_gene=include_zero_gene,
        )
        if not tokenized_batch:
            return {}

        input_gene_ids = tokenized_batch['genes'].to(device)
        input_values = tokenized_batch['values'].to(device)
        target_values = (tokenized_batch['target_values'].to(device)
                         if 'target_values' in tokenized_batch else None)
        if self.data.mode == "gene":
            input_pert = tokenized_batch['pert_flags'].to(device)
        elif self.data.mode == "compound":
            input_pert = tokenized_batch['pert']

        #TODO:
        #if not include_zero_gene:
        #    ...
        src_key_padding_mask = input_gene_ids.eq(
            self.data.vocab[self.data.pad_token]
        )
        with torch.cuda.amp.autocast(enabled=self.config.amp):
            output_dict = self.best_model(
                input_gene_ids,
                input_values,
                input_pert,
                src_key_padding_mask=src_key_padding_mask,
                CLS=self.config.CLS,
                CCE=self.config.CCE,
                MVC=self.config.GEPC,
                ECS=self.config.ECS,
            )
            pred_values = output_dict["mlm_output"].float()
        #TODO: 
        #if not include_zero_gene:
        #    pred_values = torch.zeros_like(ori_gene_values)
        #    pred_values[:, input_gene_ids] = output_values
        return {
            "genes": input_gene_ids,
            "de_idx": tokenized_batch.get('de_idx'),
            "values": input_values,
            "target": target_values,
            "pred": pred_values,
        }

    def eval_perturb(self, loader) -> dict[str, np.ndarray]:
        """
        Get the truth and prediction values using a given loader.

        Args:
            loader (torch.utils.data.DataLoader)

        Returns:
            dict[str, numpy.ndarray]
        """
        self.model.eval()
        pert_cat = []
        logvar = []  # if uncertainty
        de = []
        pred = []
        truth = []
        pred_de = []
        truth_de = []
        delta_pred = []
        delta_truth = []
        delta_pred_de = []
        delta_truth_de = []
    
        with torch.no_grad():
            for idx, batch_data in enumerate(loader):
                d = self.predict_batch(batch_data)
                if not d or d["de_idx"] is None:
                    continue

                pert_cat.extend(batch_data['y']['obs'][self.data.pert_col])
                de.extend(d["de_idx"])

                delta_p = d["pred"] - d["values"]
                delta_t = d["target"] - d["values"]

                pred.extend(d["pred"])
                truth.extend(d["target"])
                delta_pred.extend(delta_p)
                delta_truth.extend(delta_t)
    
                # Differentially expressed genes
                for itr, de_idx in enumerate(d["de_idx"]):
                    pred_de.append(d["pred"][itr, de_idx])
                    truth_de.append(d["target"][itr, de_idx])
                    delta_pred_de.append(delta_p[itr, de_idx])
                    delta_truth_de.append(delta_t[itr, de_idx])
    
        return {
            "pert_cat": np.array(pert_cat),
            "de_idx": to_numpy(torch.stack(de)),
            "pred": to_numpy(torch.stack(pred)),
            "truth": to_numpy(torch.stack(truth)),
            "pred_de": to_numpy(torch.stack(pred_de)),
            "truth_de": to_numpy(torch.stack(truth_de)),
            "delta_pred": to_numpy(torch.stack(delta_pred)),
            "delta_truth": to_numpy(torch.stack(delta_truth)),
            "delta_pred_de": to_numpy(torch.stack(delta_pred_de)),
            "delta_truth_de": to_numpy(torch.stack(delta_truth_de)),
        }

    @staticmethod
    def reshape_results(
            results: dict[str, np.ndarray]
        ) -> dict[str, dict[str, np.ndarray]]:
        """
        Reshape the results of self.eval_perturb() for further use.
        
        Args:
            results (dict[str, numpy.ndarray]):
                the output dict of self.eval_perturb()
        
        Returns:
            dict[str, dict[str, numpy.ndarray]:
                the outer dict with original keys
                the inner dict with perturbations as keys
        """
        pert_key = "pert_cat"
        return {
            key: {
                pert: results[key][np.where(results[pert_key] == pert)[0]]
                for pert in np.unique(results[pert_key])
            }
            for key in results if key != pert_key
        }

    def corr_heatmaps(
            self, results: dict[str, dict[str, np.ndarray]]
        ) -> dict[str, dict[str, np.ndarray]]:
        """
        Compute and plot the correlation matrices of 
        the truth and prediction values, respectively.

        Args:
            results (dict[str, dict[str, numpy.ndarray]]):
                the reshaped results

        Returns:
            dict[str, dict[str, numpy.ndarray]]:
                a dict of dicts with correlation matrices
        """
        corr1 = plot_corr_matrix(
            results['truth'].values(),
            results['pred'].values(),
            results['pred'].keys(),
            save_path=self.save_dir / 'Corr_Heatmaps_allgenes.png'
        )
        wandb.log({'Heatmaps': [wandb.Image(
            corr1.pop('fig'), caption='Corr heatmaps, all genes'
        )]})
        corr2 = plot_corr_matrix(
            results['delta_truth'].values(),
            results['delta_pred'].values(),
            results['delta_pred'].keys(),
            save_path=self.save_dir / 'Corr_Heatmaps_delta_allgenes.png'
        )
        wandb.log({'Heatmaps, delta': [wandb.Image(
            corr2.pop('fig'), caption='Corr heatmaps of delta values, all genes'
        )]})
        return {'allgenes': corr1, 'delta_allgenes': corr2}

    def compute_metrics(
            self, results: dict[str, dict[str, np.ndarray]]
        ) -> tuple[dict]:
        """
        Compute the MSE and correlations between the truth
        and prediction values perturbation by perturbation.

        Args:
            results (dict[str, dict[str, numpy.ndarray]]):
                the reshaped results

        Returns:
            tuple[dict]:
                metrics_overall and metrics_pert
        """
        perts: list = list({
            pert: 0 for key in results for pert in results[key]
            if pert != self.data.ctrl_str
        })  # Use dict instead of set to avoid a wrong order
        mean: dict[str, dict[str, np.ndarray]] = {
            key: {
                pert: np.mean(results[key][pert], axis=0)
                for pert in perts
            }
            for key in results
        }
        ## create metrics_pert:
        _: dict[str, dict[str, float]] = {
            pert: {
                'mse': mse(mean['pred'][pert],
                           mean['truth'][pert]),
                'corr': pearsonr(mean['pred'][pert],
                                 mean['truth'][pert])[0],
                'corr_delta': pearsonr(mean['delta_pred'][pert],
                                       mean['delta_truth'][pert])[0],
                'mse_de': mse(mean['pred_de'][pert],
                              mean['truth_de'][pert]),
                'corr_de': pearsonr(mean['pred_de'][pert],
                                    mean['truth_de'][pert])[0],
                'corr_delta_de': pearsonr(mean['delta_pred_de'][pert],
                                          mean['delta_truth_de'][pert])[0],
            }
            for pert in perts
        }
        # remove NaN values
        metrics_pert: dict[str, dict[str, float]] = {
            pert: {
                key: 0 if np.isnan(_[pert][key]) else _[pert][key]
                for key in _[pert]
            }
            for pert in _
        }
        ## create metrics_overall:
        keys: list = list({
            key: 0 for pert in metrics_pert for key in metrics_pert[pert]
        })
        metrics_overall: dict[str, float] = {
            key: np.mean([
                metrics_pert[pert][key] for pert in metrics_pert
            ])
            for key in keys
        }
        return metrics_overall, metrics_pert

    def eval_testdata(self) -> None:
        preds_and_truths = self.eval_perturb(self.test_loader)
        results = self.reshape_results(preds_and_truths)
        coef = self.corr_heatmaps(results)
        test_metrics, test_pert_metrics = self.compute_metrics(results)
        # NOTE: mse and pearson corr here are computed for the 
        # mean pred expressions vs. the truth mean across all genes.
        # Further, one can compute the distance of two distributions.
        self.logger.info(test_metrics)
        write_pickle(preds_and_truths,
                     self.save_dir / 'preds_and_truths.pkl.xz')
        write_pickle(coef,
                     self.save_dir / 'pert_corr_matrix.pkl.xz')
        write_json(test_metrics,
                   self.save_dir / 'test_metrics.json.xz')
        write_json(test_pert_metrics,
                   self.save_dir / 'test_pert_metrics.json.xz')
        
    def predict(
            self, pert_list: list[str], pool_size: Optional[int] = None
        ) -> dict[str, np.ndarray]:
        """
        Predict gene expression values for the given perturbations.

        Args:
            pert_list (list[str]): The list of perturbations to predict.
            pool_size (int, optional): For each perturbation, use this number
                of cells in the control and predict their perturbation results.
                If `None`, use all control cells.

        Returns:
            dict[str, numpy.ndarray]
        """
        if self.data.mode == "gene":
            gene_list = self.data.adata.var[self.data.gene_col].tolist()
            for pert in pert_list:
                for i in pert.split('+'):
                    if i != self.data.ctrl_str and i not in gene_list:
                        self.logger.warning(
                            f"The pert gene '{i}' is not in the gene list!"
                        )

        self.model.eval()
        with torch.no_grad():
            results = {}
            for pert in pert_list:
                preds = [
                    self.predict_batch(
                        batch_data,
                        include_zero_gene=self.config.include_zero_gene,
                    )["pred"]
                    for batch_data in iter(
                        self.data.get_dataloader_for_prediction(
                            self.config.eval_batch_size, pert, pool_size
                        )
                    )
                ]
                results[pert] = to_numpy(torch.cat(preds, dim=0)).mean(axis=0)
            return results

    def plot_perturb(
        self, query: str, pool_size: int = None, save_file: str = None
    ):
        sns.set_theme(
            style="ticks", rc={"axes.facecolor": (0, 0, 0, 0)}, font_scale=1.5
        )
        cond2name = dict(
            self.data.adata.obs[
                [self.data.pert_col, self.data.condition_name]
            ].values
        )
        id2gene = dict(zip(
            self.data.adata.var.index.values,
            self.data.adata.var[self.data.gene_col].values
        ))
        de_idx = []
        genes = []
        for i in self.data.adata.uns["top_non_dropout_de_20"][cond2name[query]]:
            de_idx.append(np.where(self.data.adata.var.index == i)[0][0])
            genes.append(id2gene[i])
        
        pred = self.predict([query], pool_size=pool_size)
        pred = pred[query][de_idx]
        truth = self.data.adata[
            self.data.adata.obs[self.data.pert_col] == query
        ].X.A[:, de_idx]
        ctrl_means = self.data.adata[
            self.data.adata.obs[self.data.pert_col] == self.data.ctrl_str
        ].to_df().mean()[de_idx].values
    
        pred = pred - ctrl_means
        truth = truth - ctrl_means
    
        plt.figure(figsize=[16.5, 4.5])
        plt.title(query)
        plt.boxplot(truth, showfliers=False, medianprops=dict(linewidth=0))
    
        for i in range(pred.shape[0]):
            _ = plt.scatter(i + 1, pred[i], color="red")
    
        plt.axhline(0, linestyle="dashed", color="green")
        ax = plt.gca()
        ax.xaxis.set_ticklabels(genes, rotation=90)
        plt.ylabel("Change in Gene Expression over Control", labelpad=10)
        plt.tick_params(axis="x", which="major", pad=5)
        plt.tick_params(axis="y", which="major", pad=5)
        sns.despine()
    
        if save_file:
            save_path = self.save_dir / Path(save_file).name
            plt.savefig(save_path, bbox_inches="tight", transparent=False)
        else:
            plt.show()

    def save_checkpoint(self) -> None:
        best_model_file = self.save_dir / "best_model.pt"
        torch.save(self.best_model.state_dict(), best_model_file)
        artifact = wandb.Artifact(f"best_model", type="model")
        artifact.add_file(best_model_file)
        self.run.log_artifact(artifact)

    def finish(self) -> None:
        self.run.finish()
        wandb.finish()
        gc.collect()

