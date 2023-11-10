'''
Finetune the pretrained model for compound perturbation response prediction.
'''

import os
import sys
import json
import tomli
import torch
import argparse
from pathlib import Path

sys.path.insert(0, '../..')
from perturb.data import PertData
from perturb.model import Transformer4Cmpd
from perturb.train import Trainer
from perturb.utils.attr_dict import AttrDict

os.environ["KMP_WARNINGS"] = "off"
os.environ["WANDB_MODE"] = "online"


def get_config():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog='Contact: chen.liang@ghddi.org',
    )
    parser.add_argument("-c", "--config", required=True)
    args = parser.parse_args()

    with open(args.config, 'rb') as f:
        return AttrDict(** tomli.load(f))


if __name__ == '__main__':
    conf = get_config()

    if conf.h.p.load_model:
        with open(Path(conf.h.p.load_model) / 'args.json') as f:
            model_configs = json.load(f)
        for key in ('embsize', 'd_hid', 'nheads', 'nlayers'):
            conf.h.p[key] = model_configs[key]
        vocab_file = Path(conf.h.p.load_model) / 'vocab.json'
    else:
        vocab_file = None

    perts_to_plot = [
        'O=C(Nc1cccc(Cl)c1)Nc1ncc(CCNc2ncnc3ccsc23)s1',
        'Cc1ccc(S(=O)(=O)OCC(=O)Nc2ccc(C(=O)O)c(O)c2)cc1',
    ]
    pert_data = PertData(
        dataset=conf.h.p.dataset,
        keep_ctrl=conf.h.p.keep_ctrl,
        seed=conf.h.p.seed,
        vocab_file=vocab_file,
    )
    pert_data.preprocess(
        use_key=conf.p.p.use_key,
        normalize_total=conf.p.p.normalize_total,
        log1p=conf.p.p.log1p,
        subset_hvg=conf.p.p.subset_hvg,
        binning=conf.p.p.binning,
    )
    pert_data.prepare_split(
        split_type=conf.h.p.split,
        test_size=conf.h.p.test_size,
    )
    pert_data.set_dataloader(
        batch_size=conf.h.p.batch_size,
        test_batch_size=conf.h.p.eval_batch_size,
    )

    trainer = Trainer(config=conf.h.p)
    model = Transformer4Cmpd(
        ntoken=len(pert_data.vocab),
        d_model=conf.h.p.embsize,
        nhead=conf.h.p.nheads,
        d_hid=conf.h.p.d_hid,
        nlayers=conf.h.p.nlayers,
        nlayers_cls=conf.h.p.nlayers_cls,
        n_cls=len(set(pert_data.adata.obs['cell_type'])) if conf.h.p.CLS else 1,
        dropout=conf.h.p.dropout,
        vocab=pert_data.vocab,
        pad_token=conf.h.p.pad_token,
        pad_value=conf.h.p.pad_value,
        do_mvc=conf.h.p.GEPC,
        do_dab=conf.h.p.DAB,
        use_batch_labels=conf.h.p.use_batch_labels,
        domain_spec_batchnorm=conf.h.p.DSBN,
        ecs_threshold=conf.h.p.ecs_threshold,
        explicit_zero_prob=conf.h.p.explicit_zero_prob,
        pre_norm=conf.h.p.pre_norm,
        cell_emb_style=conf.h.p.cell_emb_style,
        mvc_decoder_style=conf.h.p.mvc_decoder_style,
        use_fast_transformer=conf.h.p.use_fast_transformer,
    )
    if conf.h.p.load_model:
        model_file = Path(conf.h.p.load_model) / 'best_model.pt'
        if conf.h.p.load_param_prefixes:
            # only load params that start with the prefix
            model_dict = model.state_dict()
            pretrained_dict = torch.load(model_file, trainer.device)
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items()
                if any([k.startswith(prefix) for prefix in conf.h.p.load_param_prefixes])
            }
            for k, v in pretrained_dict.items():
                model.logger.info(f"Loading params {k} with shape {v.shape}")
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=False)
        else:
            try:
                model.load_state_dict(torch.load(model_file, trainer.device))
                model.logger.info(f"Loading all model params from {model_file}")
            except:
                # only load params that are in the model and match the size
                model_dict = model.state_dict()
                pretrained_dict = torch.load(model_file, trainer.device)
                pretrained_dict = {
                    k: v for k, v in pretrained_dict.items()
                    if k in model_dict and v.shape == model_dict[k].shape
                }
                for k, v in pretrained_dict.items():
                    model.logger.info(f"Loading params {k} with shape {v.shape}")
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)

    trainer.prepare_data(pert_data)
    trainer.prepare_model(model)
    trainer.fit()
    trainer.eval_testdata()
    for pert in perts_to_plot:
        try:
            trainer.plot_perturb(pert, pool_size=300, save_file=f"{pert}.png")
        except:
            pass
    trainer.save_checkpoint()
    trainer.finish()

