import os
import sys
from typing import Any, List

import hydra
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger

from src.dataset.datamodule import G2netDatamodule
from src.modeling.pl_model import LitModel
from src.utils.util import set_random_seed

try:
    LOGGER = "neptune"
    from neptune.new.integrations.pytorch_lightning import NeptuneLogger
except Exception:
    print("use TensorBoardLogger")
    LOGGER = "tensorboard"


def drop_wave_transform_params(conf: DictConfig, ckpt_path: str):
    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    keys = checkpoint["state_dict"].keys()
    new_state_dict = checkpoint["state_dict"].copy()
    # new_transform = LitModel.get_wave_transform(conf=conf)
    # new_params = new_transform.state_dict()
    is_dropped = False
    for key in keys:
        if key.startswith("wave_transform"):
            print("pop params from ckpt: ", key)
            _ = new_state_dict.pop(key)
            is_dropped = True
            # checkpoint["state_dict"][key] = new_params[key]
    temp_path = "./init_model.ckpt"
    checkpoint["state_dict"] = new_state_dict
    torch.save(checkpoint, temp_path)
    return temp_path, is_dropped


@hydra.main(config_path="./src/config", config_name="config")
def main(conf: DictConfig) -> None:

    set_random_seed(conf.seed)

    datamodule = G2netDatamodule(
        conf=conf,
        batch_size=conf.batch_size,
        aug_mode=conf.aug_mode,
        num_workers=conf.num_workers,
        is_debug=conf.is_debug,
    )

    datamodule.prepare_data()
    print("\t\t ==== TRAIN MODE ====")
    datamodule.setup(stage="fit")
    print(
        "training samples: {}, valid samples: {}".format(
            len(datamodule.train_dataset), len(datamodule.val_dataset)
        )
    )

    if conf.ckpt_path is not None:
        # load_from_checkpoint( is different from
        # Trainer(resume_from_checkpoint="some/path/to/my_checkpoint.ckpt")
        # load weight with args, args override hparams
        ckpt_path, is_param_dropped = drop_wave_transform_params(
            conf=conf, ckpt_path=conf.ckpt_path
        )
        model = LitModel.load_from_checkpoint(
            ckpt_path,
            conf=conf,
            dataset_len=len(datamodule.train_dataset),
            logger_name=LOGGER,
            strict=not is_param_dropped,
        )
    else:
        model = LitModel(
            conf=conf,
            dataset_len=len(datamodule.train_dataset),
            logger_name=LOGGER,
        )

    pl.trainer.seed_everything(seed=conf.seed)
    if LOGGER == "tensorboard":
        logger = TensorBoardLogger("tb_logs", name="my_model")
    elif LOGGER == "neptune":
        logger = NeptuneLogger(
            project="your_prject_name",
            name="lightning-run",  # Optional
            mode="debug" if conf.is_debug else "async",
        )
        logger.experiment["params/conf"] = conf
        logger.experiment["env/dir"] = os.getcwd()
        if conf.nept_tags[0] is not None:
            logger.experiment["sys/tags"].add(list(conf.nept_tags))

    trainer_params = OmegaConf.to_container(conf.trainer)
    trainer_params["callbacks"] = get_callbacks(
        monitor=conf.monitor, mode=conf.monitor_mode
    )
    trainer_params["logger"] = logger
    trainer = pl.Trainer(**trainer_params)

    # Run lr finder
    if conf.find_lr:
        lr_finder = trainer.tuner.lr_find(model, datamodule=datamodule)
        lr_finder.plot(suggest=True)
        plt.savefig("./lr_finder.png")
        plt.show()
        sys.exit()

    # Run Training
    trainer.fit(model, datamodule=datamodule)

    # run test after training
    ckpt_path = trainer_params["callbacks"][0].best_model_path
    del model
    torch.cuda.empty_cache()

    if conf.is_debug:
        logger_name = "tensorborad"
        logger = TensorBoardLogger("tb_logs", name="my_model")
    elif LOGGER == "neptune":
        # need resume  after fit end
        logger_name = "neptune"
        logger = NeptuneLogger(
            project="your_prject_name",
            run=logger.version,  # get closed run name
        )

    datamodule.setup(stage="test")
    model = LitModel.load_from_checkpoint(
        ckpt_path, conf=conf, dataset_len=-1, logger_name=logger_name
    )
    tester = pl.Trainer(
        gpus=1, limit_test_batches=1.0 if not conf.is_debug else 0.001, logger=logger
    )
    tester.test(model, datamodule=datamodule)


def get_callbacks(
    ema_decay: float = 0.9, monitor: str = "val_loss", mode: str = "min"
) -> list:
    callbacks: List[Any] = []
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=monitor,
        save_last=True,
        mode=mode,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)
    return callbacks


if __name__ == "__main__":
    main()
