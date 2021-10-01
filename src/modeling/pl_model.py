import math
import os
from typing import Any, Callable, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import torchvision
from kornia.augmentation import RandomCrop, RandomMixUp
from kornia.augmentation.utils.helpers import _transform_output_shape
from nnAudio.Spectrogram import STFT, CQT1992v2
from omegaconf import DictConfig
from sklearn.metrics import RocCurveDisplay, roc_auc_score, roc_curve
from torch.cuda.amp import autocast
from torch.distributions import Bernoulli
from torchvision.transforms import Normalize
from torchvision.utils import make_grid

from src.dataset.datamodule import IMG_MEAN, IMG_STD, G2netDatamodule
from src.dataset.dataset import WaveformDataset
from src.modeling.losses.conv_losses import DenoiseLoss
from src.modeling.model_arch.conv_models import (
    PsdDenoise,
    SegAct,
    SegDenoise,
    SiteSplitConv,
    TimmConv,
)


class LitModel(pl.LightningModule):
    def __init__(
        self, conf: DictConfig, dataset_len: int = 72899, logger_name="tensorboard"
    ) -> None:
        super().__init__()
        # self.save_hyperparameters()  # type: ignore
        # self.hparams = conf  # type: ignore[misc]
        self.conf = conf
        self.dataset_len = dataset_len
        self.logger_name = logger_name

        print("\t >>do classification")

        self.num_inchannels = self.conf.sampling_info.num_sites * 3
        if self.conf.spec_conf.use_second_spec:
            # add another spec for inputs, +3 ch
            self.num_inchannels += 3
        self.num_classes = self.conf.num_classes

        net_kwargs = {
            "timm_params": self.conf.model.timm_params,
            "gem_power": self.conf.model.pool.gem_power,
            "gem_requires_grad": self.conf.model.pool.gem_requires_grad,
            "num_classes": self.num_classes,
        }

        (
            self.normalize,
            self._img_mean,
            self._img_std,
        ) = LitModel.set_image_normalization(num_inchannels=self.num_inchannels)

        input_height, input_width = G2netDatamodule.get_spec_size(
            spec_conf=self.conf.spec_conf,
            fs=self.conf.sampling_info.fs,
            T=self.conf.sampling_info.T,
        )
        self.crop_size = (input_height - 1, input_width - 1)
        type_conf = self.conf.model[self.conf.model.type]
        if self.conf.model.type == "split_site":
            self.num_inchannels = self.num_inchannels // type_conf["channels_per_site"]
            net_kwargs.update(
                {
                    "channels_per_site": type_conf["channels_per_site"],
                    "norm_feature": type_conf["norm_feature"],
                    "fuse_type": type_conf["fuse_type"],
                }
            )
            net_class = SiteSplitConv
        elif self.conf.model.type == "concat_site":
            net_class = TimmConv
        elif self.conf.model.type == "concat_site_seg":
            net_class = SegDenoise
            net_kwargs = {
                "smp_params": self.conf.model.smp_params,
                "img_mean": self._img_mean,
                "img_std": self._img_std,
                "crop_size": self.crop_size,
                "num_sites": self.conf.sampling_info.num_sites,
                "predict_mask": type_conf["predict_mask"],
                "use_full_encoder": type_conf["use_full_encoder"],
                "use_cam": type_conf["use_cam"],
                "skip_max_pool": type_conf["skip_max_pool"],
            }
        elif self.conf.model.type == "psd_denoise":
            net_class = PsdDenoise
            net_kwargs.update(
                {
                    "num_sites": self.conf.sampling_info.num_sites,
                    "num_features_per_site": 512,
                }
            )

        else:
            raise NotImplementedError(f"model.type : {self.conf.model.type}")

        self.model = net_class(**net_kwargs)

        patch_first_conv(
            self.model,
            in_channels=self.num_inchannels,
            stride_override=self.conf.model.stride_override,
        )

        if self.conf.model.channels_last:
            # Need to be done once, after model initialization (or load)
            self.model = self.model.to(memory_format=torch.channels_last)

        if self.conf.model.loss.type == "mse":
            self.criterion = torch.nn.MSELoss(reduction="none")
        elif self.conf.model.loss.type == "bce":
            self.criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        elif self.conf.model.loss.type == "bce_seg_denoise":
            assert (
                self.conf.model.type == "concat_site_seg"
            ), f"wrong type combination, {self.conf.model.type}"
            self.criterion = DenoiseLoss(
                crop_size=type_conf["crop_size"],
                batch_size=self.conf.batch_size,
                num_sites=self.conf.sampling_info.num_sites,
                target_loss_weight=type_conf["target_loss_weight"],
                l1_loss_weight=type_conf["l1_loss_weight"],
                var_loss_weight=type_conf["var_loss_weight"],
                predict_mask=type_conf["predict_mask"],
            )
        else:
            raise NotImplementedError

        if self.conf.model.metrics == "mse":
            self.metrics = pl.metrics.MeanSquaredError()
        else:
            raise NotImplementedError
        self.aux_cirterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.criterions = {
            "outputs": self.criterion,
            "aux_out": self.aux_cirterion,
        }
        self.loss_weights = self.conf.model.loss.weights

        if self.conf.model.last_act == "sigmoid":
            self.activation = torch.nn.Sigmoid()
        elif self.conf.model.last_act == "tanh":
            self.activation = torch.nn.Tanh()
        elif self.conf.model.last_act == "identity":
            self.activation = torch.nn.Identity()
        elif self.conf.model.last_act == "seg_act":
            assert (
                self.conf.model.type == "concat_site_seg"
            ), f"wrong type combination, {self.conf.model.type}"
            self.activation = SegAct(predict_mask=type_conf["predict_mask"])
        else:
            raise NotImplementedError
        self.aux_activation = torch.nn.Sigmoid()
        self.activations = {
            "outputs": self.activation,
            "aux_out": self.aux_activation,
        }

        self.val_sync_dist = self.conf.trainer.gpus > 1
        self.is_debug = self.conf.is_debug

        # self.h_, self.w_ = get_input_size_wo_pad(
        #     n_fft=self.conf.stft_params.n_fft, input_width=self.conf.input_width
        # )

        self.mixup = Mixup(
            p=self.conf.mix_up_conf.p,
            lambda_val=self.conf.mix_up_conf.lambda_val,
            use_mixup_label=self.conf.mix_up_conf.use_mixup_label,
            is_aux_in=self.conf.model.type in ["psd_denoise"],
        )

        self.pred_df = pd.DataFrame({})

        # gpu spec generation configuration
        spec_conf = self.conf.spec_conf
        if spec_conf.with_gpu:
            self.spec_params = spec_conf[spec_conf.params_type]
            self.spec_lib = spec_conf.spec_lib
            self.db_conf = spec_conf.db_conf
            self.is_db = spec_conf.is_db
            self.wave_transform, self.wave_transform_sec = LitModel.get_wave_transform(
                conf=self.conf
            )

            pad_size = int(input_width * self.conf.pad_conf.pad_ratio)
            # left, right, top,, bottom
            # https://pytorch.org/docs/stable/generated/torch.nn.ConstantPad2d.html?highlight=pad2d#torch.nn.ConstantPad2d
            self.time_pad = torch.nn.ConstantPad2d(
                padding=(pad_size, pad_size, 0, 0),
                value=self.conf.pad_conf.constant_values,
            )
            # height, width
            # https://kornia.readthedocs.io/en/latest/augmentation.module.html
            self.random_crop = RandomCrop((input_height, input_width), p=1.0)

    def forward(self, x):
        x = self.model(x)
        return x

    @staticmethod
    def get_wave_transform(conf: DictConfig) -> Tuple[Callable, Callable]:
        spec_conf = conf.spec_conf
        if spec_conf.with_gpu:
            spec_params = spec_conf[spec_conf.params_type]
            spec_lib = spec_conf.spec_lib

            if spec_conf.params_type == "stft_params":
                if spec_lib == "librosa":
                    raise NotImplementedError("librosa is not allowed for gpu")
                elif spec_lib == "nnAudio":
                    wave_transform = STFT(**spec_params)

            elif spec_conf.params_type == "cqt_params":
                wave_transform = CQT1992v2(**spec_params)
            else:
                raise NotImplementedError(
                    f"unexpected value for params_type: {spec_conf.params_type}"
                )
            if conf.spec_conf.use_second_spec:
                wave_transform_sec = CQT1992v2(**spec_conf["cqt_params"])
            else:
                wave_transform_sec = None
        return wave_transform, wave_transform_sec

    @staticmethod
    def power_to_db_torch(S, ref=1.0, amin=1e-10, top_db=80.0):
        """
        copy and modify from librosa power_to_db
        """
        log_spec = 10.0 * torch.log10(torch.maximum(amin, S))
        log_spec -= 10.0 * torch.log10(torch.maximum(amin, ref))
        if top_db is not None:
            log_spec = torch.maximum(log_spec, log_spec.max() - top_db)

        return log_spec

    @staticmethod
    def amplitude_to_db_torch(S, ref=1.0, amin=1e-5, top_db=80.0, device="cuda"):
        """
        copy and modify from librosa amplitude_to_db
        """
        # power = np.square(magnitude, out=magnitude)
        power = torch.pow(S, 2)
        ref = torch.tensor(ref, dtype=torch.float32, device=device)
        amin = torch.tensor(amin, dtype=torch.float32, device=device)
        top_db = torch.tensor(top_db, dtype=torch.float32, device=device)
        return LitModel.power_to_db_torch(
            power, ref=ref ** 2, amin=amin ** 2, top_db=top_db
        )

    @staticmethod
    def generate_image_from_wave(
        sample_data: np.ndarray,
        wave_transform: Callable,
        spec_params: Dict[str, Any],
        lib: str = "librosa",
        is_db: bool = False,
        is_pad: bool = False,
        amin: float = 1.0e-15,
        top_db: int = 200,
        db_offset: int = 130,
        ref: float = 1.0,
        dtype: np.dtype = np.float64,
        device: str = "cuda",
    ) -> torch.Tensor:

        D_sites = []
        for site_ind in range(sample_data.shape[1]):
            # batch x H x W
            D_sites.append(wave_transform(sample_data[:, site_ind]))

        # batch x num_sites x H x W
        D = torch.stack(D_sites, dim=1)
        # # batch_size = sample_data.shape[0]
        # D_sites = wave_transform(
        #     torch.cat(torch.split(sample_data, 1, dim=1), dim=0).squeeze()
        # )
        # D = torch.stack(torch.split(D_sites, batch_size, dim=0), dim=1)

        D_abs = torch.sqrt(D[..., 0].pow(2) + D[..., 1].pow(2))
        D_theta = torch.atan2(D[..., 1] + 0.0, D[..., 0])

        if is_db:
            # D_abs = librosa.amplitude_to_db(D_abs, ref=ref, amin=amin, top_db=top_db)
            D_abs = LitModel.amplitude_to_db_torch(
                D_abs, ref=ref, amin=amin, top_db=top_db, device=device
            )
            D_theta = torch.where(
                D_abs == (D_abs.max() - top_db),
                torch.tensor(0.0, dtype=torch.float32, device=device),
                D_theta,
            )

        D_cos = torch.cos(D_theta)
        D_sin = torch.sin(D_theta)
        img = torch.cat([D_abs, D_cos, D_sin], axis=1)

        img = WaveformDataset.handle_spec_normalize(
            img=img,
            num_abs_channels=sample_data.shape[1],
            is_encode=True,
            is_db=is_db,
            db_offset=db_offset,
            image_fomat="torch",
        )

        return img

    @staticmethod
    def show(imgs: torch.Tensor):
        """
        for tensor debugging
        imgs (ba, ch, h, w) shape
        """
        import torchvision.transforms.functional as F

        imgs = make_grid(imgs)

        if not isinstance(imgs, list):
            imgs = [imgs]
        fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        for i, img in enumerate(imgs):
            img = img.detach()
            img = F.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    def training_step(self, batch, batch_idx):
        if self.conf.spec_conf.with_gpu:
            inputs = self.preproc_wave_data(
                sample_data=batch["sample_data_bp"],
                mode="train",
                channels_last=self.conf.model.channels_last,
            )
        else:
            inputs = batch["image"]
            if self.conf.model.channels_last:
                # Need to be done for every input
                inputs = inputs.to(memory_format=torch.channels_last)

        targets = batch["target"]

        inputs, targets, aux_inputs = self.mixup(
            inputs=inputs, targets=targets, aux_inputs=batch["site_psds"]
        )
        # outputs is dict for multi heads
        outputs = self.model(inputs, aux_inputs=aux_inputs)
        preds = {key: None for key in outputs.keys()}
        losses = {key: None for key in outputs.keys()}
        for output_key, output in outputs.items():
            if output is None:
                continue
            preds[output_key] = self.activations[output_key](output).squeeze()
            losses[output_key] = self.mixup.process_loss(
                logits=output.squeeze(),
                targets_mixup=targets,
                criterion=self.criterions[output_key],
            ).mean()

        for output_key, loss in losses.items():
            if output_key == "outputs":
                if self.logger_name == "tensorboard":
                    self.log("train_loss", loss)
                elif self.logger_name == "neptune":
                    self.logger.experiment["loss/train"].log(loss)
            elif loss is not None:
                if self.logger_name == "neptune":
                    self.logger.experiment[f"loss/train_{output_key}"].log(loss)

        total_loss = 0
        for output_key, loss in losses.items():
            if loss is None:
                continue

            total_loss += self.loss_weights[output_key] * loss
        return total_loss

    @torch.no_grad()  # disable gradients for effiency
    def preproc_wave_data(
        self,
        sample_data: torch.Tensor,
        mode: str = "val",
        channels_last: bool = False,
    ) -> torch.Tensor:
        with autocast(enabled=False):
            inputs = LitModel.generate_image_from_wave(
                sample_data=sample_data,
                wave_transform=self.wave_transform,
                spec_params=self.spec_params,
                lib=self.spec_lib,
                is_db=self.is_db,
                amin=self.db_conf.amin,
                top_db=self.db_conf.top_db,
                db_offset=self.db_conf.db_offset,
                ref=self.db_conf.ref,
                device=self.device,
            )
            if self.conf.spec_conf.use_second_spec:
                inputs_sec = LitModel.generate_image_from_wave(
                    sample_data=sample_data,
                    wave_transform=self.wave_transform_sec,
                    spec_params=self.spec_params,
                    lib=self.spec_lib,
                    is_db=self.is_db,
                    amin=self.db_conf.amin,
                    top_db=self.db_conf.top_db,
                    db_offset=self.db_conf.db_offset,
                    ref=self.db_conf.ref,
                    device=self.device,
                )
                inputs_sec = torchvision.transforms.functional.resize(
                    inputs_sec, size=(inputs.shape[2], inputs.shape[3]), antialias=True
                )
                inputs = torch.cat([inputs_sec[:, :3, :, :], inputs], dim=1)

            if channels_last:
                # Need to be done for every input
                inputs = inputs.to(memory_format=torch.channels_last)
            inputs /= 255.0
            inputs = self.normalize(inputs)

            if mode == "train":
                inputs_crop = self.random_crop(self.time_pad(inputs))
                batch_mask = get_aug_mask(
                    batch_size=inputs.size()[0:1], probs=self.conf.pad_conf.p
                )
                batch_mask = (
                    batch_mask.view(-1, 1, 1, 1).expand_as(inputs).to(self.device)
                )
                inputs = (1.0 - batch_mask) * inputs + batch_mask * inputs_crop

        if mode == "test":
            # use all time information
            inputs = inputs[:, :, : self.crop_size[0]]
        else:
            inputs = inputs[:, :, : self.crop_size[0], : self.crop_size[1]]
        return inputs

    def validation_step(self, batch, batch_idx):
        if self.conf.spec_conf.with_gpu:
            inputs = self.preproc_wave_data(
                sample_data=batch["sample_data_bp"],
                mode="val",
                channels_last=self.conf.model.channels_last,
            )
        else:
            inputs = batch["image"]
            if self.conf.model.channels_last:
                # Need to be done for every input
                inputs = inputs.to(memory_format=torch.channels_last)

        targets = batch["target"]

        aux_inputs = batch["site_psds"]
        outputs = self.model(inputs, aux_inputs=aux_inputs)

        preds = {key: None for key in outputs.keys()}
        losses = {key: None for key in outputs.keys()}
        for output_key, output in outputs.items():
            if output is None:
                continue
            preds[output_key] = self.activations[output_key](output).squeeze()
            losses[output_key] = self.criterions[output_key](
                output.squeeze(), targets
            ).mean()

        for output_key, loss in losses.items():
            if output_key == "outputs":
                self.log("val_loss", loss)
                if self.logger_name == "neptune":
                    self.logger.experiment["loss/val"].log(loss)
            elif loss is not None:
                if self.logger_name == "neptune":
                    self.logger.experiment[f"loss/val_{output_key}"].log(loss)

        return {"id": batch["id"], "pred": preds["outputs"], "targets": targets}

    def validation_epoch_end(self, val_step_outputs):
        keys = list(val_step_outputs[0].keys())
        met_dict = {key: [] for key in keys}
        for pred_batch in val_step_outputs:
            for key in keys:
                met_dict[key].append(pred_batch[key])

        for key in keys:
            if isinstance(met_dict[key][0], torch.Tensor):
                met_dict[key] = (
                    torch.cat(met_dict[key]).cpu().numpy().astype(np.float32)
                )

            elif isinstance(met_dict[key][0], np.ndarray):
                met_dict[key] = np.concatenate(met_dict[key])

            elif isinstance(met_dict[key][0], list):
                met_dict[key] = np.concatenate(met_dict[key])

            else:
                raise ValueError(f"unexpected type {type(met_dict[key])}")

        pred_df = self.generate_pred_df(
            met_dict=met_dict,
            is_test=False,
        )

        try:
            roc = roc_auc_score(y_true=pred_df["targets"], y_score=pred_df["pred"])
        except ValueError as e:
            if len(pred_df) < 10000:
                print("sanity check error for roc calc with single calss")
                roc = 0.5
            else:
                raise ValueError(e)

        self.log("val_roc", roc)
        if self.logger_name == "neptune":
            self.logger.experiment["met/val"].log(roc)

        fig, _ = LitModel.vis_pred_stats(pred_df=pred_df, roc=roc)
        if self.logger_name == "tensorboard":
            self.logger.experiment.add_figure(
                "prediction_fig",
                fig,
                global_step=self.trainer.global_step,
            )
        elif self.logger_name == "neptune":
            self.logger.experiment["val/pred_stats"].log(fig)
        plt.close()

    @staticmethod
    def vis_pred_stats(
        pred_df: pd.DataFrame, roc: Optional[float] = None, is_test: bool = False
    ):

        nrows = 1
        ncols = 2
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(12, 6),
            sharey=False,
            sharex=False,
        )
        if not is_test:
            fpr, tpr, thresholds = roc_curve(
                y_true=pred_df["targets"], y_score=pred_df["pred"], pos_label=1
            )
            hue = "targets"
            _ = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc).plot(
                ax=axes[0], name="g2net ROC"
            )
        else:
            hue = None
        sns.histplot(
            data=pred_df, x="pred", hue=hue, ax=axes[1], bins=25, multiple="layer"
        )
        return fig, axes

    def generate_pred_df(
        self,
        met_dict: Dict[str, Any],
        is_test: bool = False,
        is_save: bool = True,
        save_dir: str = "./",
    ) -> pd.DataFrame:
        pred_df = pd.DataFrame(met_dict)
        if is_test:
            csv_name = "submission.csv"
            save_path = os.path.join(save_dir, csv_name)
            pred_df.rename(columns={"pred": "target"}).to_csv(save_path, index=False)
            print(f"save submission file on {os.getcwd()} {save_path}")
            return pred_df

        # skip check step
        elif len(pred_df) < self.conf.batch_size * 9:
            return pred_df

        if len(self.pred_df) == 0:
            self.pred_df = pred_df
            self.pred_df = self.pred_df.rename(columns={"pred": "pred_epoch_1"})
        else:
            epoch = (self.pred_df.shape[1] - 2) + 1  # there are "id" "targets" col
            self.pred_df[f"pred_epoch_{epoch}"] = pred_df.loc[:, "pred"]

        if is_save:
            csv_name = "val_epochs.csv"
            # print("save prediction csv", save_path)
            save_path = os.path.join(save_dir, csv_name)
            self.pred_df.to_csv(save_path, index=False)
        return pred_df

    def test_step(self, batch, batch_idx):
        if self.conf.spec_conf.with_gpu:
            inputs = self.preproc_wave_data(
                sample_data=batch["sample_data_bp"],
                mode="test",
                channels_last=self.conf.model.channels_last,
            )
        else:
            inputs = batch["image"]
            if self.conf.model.channels_last:
                # Need to be done for every input
                inputs = inputs.to(memory_format=torch.channels_last)

        aux_inputs = batch["site_psds"]
        outputs = self.model(inputs, aux_inputs=aux_inputs)
        preds = {key: None for key in outputs.keys()}
        for output_key, output in outputs.items():
            if output is None:
                continue
            preds[output_key] = self.activations[output_key](output).squeeze()

        return {"id": batch["id"], "pred": preds["outputs"]}

    def test_epoch_end(self, test_step_outputs):
        keys = list(test_step_outputs[0].keys())
        met_dict = {key: [] for key in keys}
        for pred_batch in test_step_outputs:
            for key in keys:
                met_dict[key].append(pred_batch[key])

        for key in keys:
            if isinstance(met_dict[key][0], torch.Tensor):
                met_dict[key] = (
                    torch.cat(met_dict[key]).cpu().numpy().astype(np.float32)
                )

            elif isinstance(met_dict[key][0], np.ndarray):
                met_dict[key] = np.concatenate(met_dict[key])

            elif isinstance(met_dict[key][0], list):
                met_dict[key] = np.concatenate(met_dict[key])

            else:
                raise ValueError(f"unexpected type {type(met_dict[key])}")

        pred_df = self.generate_pred_df(
            met_dict=met_dict,
            is_test=True,
        )

        fig, _ = LitModel.vis_pred_stats(pred_df=pred_df, roc=None, is_test=True)
        if self.logger_name == "tensorboard":
            self.logger.experiment.add_figure(
                "prediction_test_fig",
                fig,
                global_step=self.trainer.global_step,
            )
        elif self.logger_name == "neptune":
            self.logger.experiment["test/pred_stats"].log(fig)
            self.logger.experiment["test/submission"].upload("submission.csv")
        plt.savefig("submission.png", dpi=300)

        plt.close()

    @staticmethod
    def set_image_normalization(num_inchannels: int = 9) -> tuple:
        img_mean = IMG_MEAN[:num_inchannels]  # type: ignore[union-attr]
        img_std = IMG_STD[:num_inchannels]  # type: ignore[union-attr]
        transform = Normalize(mean=img_mean, std=img_std)

        img_mean = torch.tensor(
            np.array(img_mean, dtype=np.float32)[None, :, None, None],
        )
        img_std = torch.tensor(np.array(img_std, dtype=np.float32)[None, :, None, None])
        return (
            transform,
            img_mean,
            img_std,
        )

    def optimizer_step(
        self,
        current_epoch,
        batch_nb,
        optimizer,
        optimizer_idx,
        closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        if not self.conf.find_lr:
            if self.trainer.global_step < self.warmup_steps:
                lr_scale = min(
                    1.0, float(self.trainer.global_step + 1) / self.warmup_steps
                )
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_scale * self.conf.lr
            else:
                pct = (self.trainer.global_step - self.warmup_steps) / (
                    self.total_steps - self.warmup_steps
                )
                pct = min(1.0, pct)
                for pg in optimizer.param_groups:
                    pg["lr"] = self._annealing_cos(pct, start=self.conf.lr, end=0.0)

        if self.logger_name == "neptune":
            self.logger.experiment["train/lr"].log(optimizer.param_groups[0]["lr"])
        optimizer.step(closure=closure)
        optimizer.zero_grad()

    def _annealing_cos(self, pct: float, start: float = 0.1, end: float = 0.0) -> float:
        """
        https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#CosineAnnealingLR
        Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0.
        """
        cos_out = math.cos(math.pi * pct) + 1
        return end + (start - end) / 2.0 * cos_out

    def configure_optimizers(self):
        self.total_steps = (
            self.dataset_len // self.conf.batch_size
        ) * self.conf.trainer.max_epochs
        self.warmup_steps = int(self.total_steps * self.conf.warmup_ratio)

        if self.conf.optim_name == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.conf.lr,
                momentum=0.9,
                weight_decay=4e-5,
            )
        elif self.conf.optim_name == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.conf.lr)
        else:
            raise NotImplementedError
        # steps_per_epoch = self.hparams.dataset_len // self.hparams.batch_size
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer,
        #     max_lr=self.hparams.lr,
        #     max_epochs=self.hparams.max_epochs,
        #     steps_per_epoch=steps_per_epoch,
        # )
        # return [optimizer], [scheduler]
        return optimizer


def patch_first_conv(
    model, in_channels: int = 4, stride_override: Optional[Tuple[int, int]] = None
) -> None:
    """
    from segmentation_models_pytorch/encoders/_utils.py
    Change first convolution layer input channels.
    In case:
        in_channels == 1 or in_channels == 2 -> reuse original weights
        in_channels > 3 -> make random kaiming normal initialization
    """

    # get first conv
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            break

    # change input channels for first conv
    module.in_channels = in_channels
    weight = module.weight.detach()
    # reset = False

    if in_channels == 1:
        weight = weight.sum(1, keepdim=True)
    elif in_channels == 2:
        weight = weight[:, :2] * (3.0 / 2.0)
    elif in_channels == 3:
        pass
    elif in_channels == 4:
        weight = torch.nn.Parameter(torch.cat([weight, weight[:, -1:, :, :]], dim=1))
    elif in_channels % 3 == 0:
        weight = torch.nn.Parameter(torch.cat([weight] * (in_channels // 3), dim=1))

    module.weight = weight
    if stride_override is not None:
        assert module.stride == (2, 2), "wrong stride_override target"
        module.stride = tuple(stride_override)


def get_aug_mask(
    batch_size: torch.Size,
    probs: float = 0.3,
    device: str = "cuda",
) -> torch.Tensor:
    # dist = Uniform(
    #     low=torch.as_tensor(low, device=device, dtype=torch.int64),
    #     heigh=torch.as_tensor(heigh, device=device, dtype=torch.int64),
    #     validate_args=False,
    # )

    dist = Bernoulli(
        probs=torch.as_tensor(probs, device=device, dtype=torch.float32),
        validate_args=False,
    )
    batch_mask = dist.sample(batch_size).view(batch_size, 1, 1, 1)

    return batch_mask


class MixupWithAux(RandomMixUp):
    def __init__(
        self,
        lambda_val: Optional[Union[torch.Tensor, Tuple[float, float]]] = None,
        same_on_batch: bool = False,
        p: float = 1.0,
        keepdim: bool = False,
    ) -> None:
        """
        from kornia RandomMixUp.
        modify for additonal mixup for aux input
        """
        super().__init__(
            lambda_val=lambda_val,
            same_on_batch=same_on_batch,
            p=p,
            keepdim=keepdim,
        )
        self.lambda_val = lambda_val

    def apply_transform(  # type: ignore
        self,
        input: torch.Tensor,
        label: torch.Tensor,
        aux_input: torch.Tensor,
        params: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_permute = input.index_select(
            dim=0, index=params["mixup_pairs"].to(input.device)
        )
        labels_permute = label.index_select(
            dim=0, index=params["mixup_pairs"].to(label.device)
        )
        aux_input_permute = aux_input.index_select(
            dim=0, index=params["mixup_pairs"].to(aux_input.device)
        )
        lam = (
            params["mixup_lambdas"].view(-1, 1, 1, 1).expand_as(input).to(label.device)
        )
        lam_aux = (
            params["mixup_lambdas"].view(-1, 1, 1).expand_as(aux_input).to(label.device)
        )
        inputs = input * (1 - lam) + input_permute * lam
        aux_inputs = aux_input * (1 - lam_aux) + aux_input_permute * lam_aux
        out_labels = torch.stack(
            [
                label.to(input.dtype),
                labels_permute.to(input.dtype),
                params["mixup_lambdas"].to(label.device, input.dtype),
            ],
            dim=-1,
        ).to(label.device)
        return inputs, out_labels, aux_inputs

    def apply_func(  # type: ignore
        self,
        in_tensor: torch.Tensor,
        label: torch.Tensor,
        aux_input: torch.Tensor,
        params: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        to_apply = params["batch_prob"]

        # if no augmentation needed
        if torch.sum(to_apply) == 0:
            output = in_tensor
            aux_output = aux_input
        # if all data needs to be augmented
        elif torch.sum(to_apply) == len(to_apply):
            output, label, aux_output = self.apply_transform(
                in_tensor,
                label,
                aux_input,
                params,
            )
        else:
            raise ValueError(
                "Mix augmentations must be performed batch-wisely. Element-wise augmentation is not supported."
            )

        return output, label, aux_output

    def forward(  # type: ignore
        self,
        input: torch.Tensor,
        label: Optional[torch.Tensor] = None,
        aux_input: torch.Tensor = None,
        params: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        in_tensor, in_trans = self.__unpack_input__(input)
        ori_shape = in_tensor.shape
        in_tensor = self.transform_tensor(in_tensor)
        # If label is not provided, it would output the indices instead.
        if label is None:
            if isinstance(input, (tuple, list)):
                device = input[0].device
            else:
                device = input.device
            label = torch.arange(0, in_tensor.size(0), device=device, dtype=torch.long)
        if params is None:
            batch_shape = in_tensor.shape
            params = self.forward_parameters(batch_shape)
        self._params = params

        output, lab, aux_inputs = self.apply_func(
            in_tensor=in_tensor,
            label=label,
            aux_input=aux_input,
            params=self._params,
        )
        output = _transform_output_shape(output, ori_shape) if self.keepdim else output  # type: ignore
        if in_trans is not None:
            return (output, in_trans), lab
        return output, lab, aux_inputs


class Mixup(torch.nn.Module):
    def __init__(
        self,
        p: float = 0.3,
        lambda_val: Tuple[float, float] = [0.0, 1.0],
        use_mixup_label: bool = True,
        is_aux_in: bool = False
        # p_elemnt: flaot = 1.0,
    ):
        super().__init__()
        if is_aux_in:
            self.mixup_layer = MixupWithAux(p=p, lambda_val=lambda_val)
        else:
            self.mixup_layer = RandomMixUp(p=p, lambda_val=lambda_val)
        self.p = p
        self.use_mixup_label = use_mixup_label
        self.is_aux_in = is_aux_in
        if not self.use_mixup_label:
            assert np.max(lambda_val) < 0.2

    # from
    # https://colab.sandbox.google.com/github/kornia/tutorials/blob/master/source/data_augmentation_kornia_lightning_gpu.ipynb
    @torch.no_grad()  # disable gradients for effiency
    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor, aux_inputs: torch.Tensor
    ):
        if self.p > 0.0:
            if self.is_aux_in:
                inputs, targets, aux_inputs = self.mixup_layer(
                    input=inputs, label=targets, aux_input=aux_inputs
                )
            else:
                inputs, targets = self.mixup_layer(input=inputs, label=targets)

        return inputs, targets, aux_inputs

    def process_loss(
        self,
        logits: torch.Tensor,
        targets_mixup: torch.Tensor,
        criterion: Callable,
    ) -> torch.Tensor:
        if (self.p == 0.0) or (targets_mixup.ndim == 1):
            loss = criterion(logits, targets_mixup)
            return loss

        if self.use_mixup_label:
            loss_a = criterion(logits, targets_mixup[:, 0])
            loss_b = criterion(logits, targets_mixup[:, 1])
            return (1 - targets_mixup[:, 2]) * loss_a + targets_mixup[:, 2] * loss_b
        else:
            loss = criterion(logits, targets_mixup[:, 0])
            return loss
