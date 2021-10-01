from pathlib import Path
from typing import List

import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

from src.dataset.datamodule import G2netDatamodule
from src.modeling.pl_model import LitModel
from src.utils.util import set_random_seed

pd.set_option("display.max_rows", 100)
SEED = 42


@hydra.main(config_path="./src/config", config_name="test_config")
def main(conf: DictConfig) -> None:
    set_random_seed(SEED)

    for model_path in conf.test_weights.model_paths:
        model_path = Path(conf.test_weights.weights_dir, model_path[1])
        conf_path = model_path / conf.test_weights.conf_name
        model_conf = OmegaConf.load(conf_path)
        ckpt_path = list(model_path.glob(conf.test_weights.ckpt_regex))

        assert len(ckpt_path) == 1
        model_conf.ckpt_path = str(ckpt_path[0])
        print("\t\t ==== TEST MODE ====")
        print("load from: ", model_conf.ckpt_path)

        # add missing keys
        model_conf.test_with_val = conf.test_with_val
        datamodule = G2netDatamodule(
            conf=model_conf,
            batch_size=model_conf.batch_size,
            aug_mode=model_conf.aug_mode,
            num_workers=model_conf.num_workers,
            is_debug=model_conf.is_debug,
        )
        datamodule.prepare_data()
        datamodule.setup(stage="test")
        model = LitModel.load_from_checkpoint(
            model_conf.ckpt_path, conf=model_conf, dataset_len=-1
        )
        pl.trainer.seed_everything(seed=SEED)
        trainer = pl.Trainer(gpus=1, limit_test_batches=1.0)
        trainer.test(model, datamodule=datamodule)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
