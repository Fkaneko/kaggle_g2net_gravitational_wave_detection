import os
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as albu
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, ListConfig, OmegaConf
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from src.dataset.dataset import WaveformDataset
from src.dataset.utils.process_df import add_data_path, path_2_id
from src.dataset.utils.waveform_preprocessings import load_psd
from src.postprocess.visualize import vis_spec

IMG_MEAN = (0.485, 0.456, 0.406) * 5
IMG_STD = (0.229, 0.224, 0.225) * 5


class G2netDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        conf: DictConfig,
        batch_size: int = 64,
        num_workers: int = 16,
        aug_mode: int = 0,
        is_debug: bool = False,
    ) -> None:
        super().__init__()
        self.conf = conf
        self.batch_size = batch_size
        self.aug_mode = aug_mode
        self.num_workers = num_workers
        self.is_debug = is_debug
        self.input_width = conf["input_width"]
        self.num_inchannels = 3 * 3  # site num * abs, cos, sin

        self.img_mean = np.array(IMG_MEAN[: self.num_inchannels])
        self.img_std = np.array(IMG_STD[: self.num_inchannels])

    def prepare_data(self):
        # check
        assert Path(get_original_cwd(), self.conf["data_dir"]).is_dir()

    def _onehot_to_set(self, onehot: np.ndarray):
        return set(np.where(onehot == 1)[0].astype(str).tolist())

    def setup(self, stage: Optional[str] = None):
        # Assign Train/val split(s) for use in Dataloaders

        conf = self.conf
        if stage == "fit" or stage is None:
            # for hydra
            cwd = get_original_cwd()

            # load data
            data_dir = Path(cwd, self.conf["data_dir"])
            self.train_df = pd.read_csv(data_dir / "training_labels.csv")
            self.test_df = pd.read_csv(data_dir / "sample_submission.csv")

            # ext data load
            if conf.ext_data.data_csv is not None:
                pass
            else:
                ext_data_paths = []
                ext_psd_paths = []
                ext_targets = []

            self.train_df = add_data_path(
                df=self.train_df, is_train=True, data_dir=data_dir
            )
            avg_psd = load_psd(
                df=self.train_df,
                fs=conf.sampling_info.fs,
                T=conf.sampling_info.T,
                num_workers=conf.num_workers,
                window=conf.wave_prepro_conf.psd.window,
                mode=conf.wave_prepro_conf.psd.avg_mode,
                avg_psd_cache=Path(
                    get_original_cwd(), conf.wave_prepro_conf.psd.cache_path
                ),
                plot_psds=conf.is_debug,
            )

            # train/val split
            self.train_df = make_split(df=self.train_df, n_splits=conf.n_splits)

            train_df = self.train_df.loc[self.train_df["fold"] != conf.val_fold, :]
            val_df = self.train_df.loc[self.train_df["fold"] == conf.val_fold, :]

            self.train_dataset = WaveformDataset(
                sample_path_list=train_df["path"].to_numpy().tolist() + ext_data_paths,
                wave_prepro_conf=conf.wave_prepro_conf,
                spec_conf=conf.spec_conf,
                avg_psd=avg_psd,
                targets=train_df["target"].to_numpy().tolist() + ext_targets,
                spec_with_gpu=conf.spec_conf.with_gpu,
                T=conf.sampling_info.T,
                fs=conf.sampling_info.fs,
                image_transforms=self.train_transform(),
                is_test=False,
                psd_aug_p=conf.psd_aug.p,
                psd_aug_mean=conf.psd_aug.mean,
                psd_aug_sigma_x3=conf.psd_aug.sigma_x3,
                use_ext_psd=conf.ext_data.use_ext_psd,
                ext_psd_path_list=[None] * len(train_df) + ext_psd_paths,
                is_psd_denoise=conf.model.type in ["psd_denoise"],
            )

            self.val_dataset = WaveformDataset(
                sample_path_list=val_df["path"].to_numpy().tolist(),
                wave_prepro_conf=conf.wave_prepro_conf,
                spec_conf=conf.spec_conf,
                avg_psd=avg_psd,
                targets=val_df["target"].to_numpy().tolist(),
                spec_with_gpu=conf.spec_conf.with_gpu,
                T=conf.sampling_info.T,
                fs=conf.sampling_info.fs,
                image_transforms=self.val_transform(),
                is_test=False,
                psd_aug_p=0.0,
                is_psd_denoise=conf.model.type in ["psd_denoise"],
            )
            self.plot_dataset(self.train_dataset)
            self.train_df = train_df
            self.val_df = val_df

        # Assign Test split(s) for use in Dataloaders
        if stage == "test" or stage is None:
            # read data
            data_dir = Path(get_original_cwd(), self.conf["data_dir"])

            # for hydra
            cwd = get_original_cwd()

            # load data
            data_dir = Path(cwd, self.conf["data_dir"])
            self.train_df = pd.read_csv(data_dir / "training_labels.csv")

            if self.conf.test_with_val:
                pass
            else:
                self.test_df = pd.read_csv(data_dir / "sample_submission.csv")

            self.test_df = add_data_path(
                df=self.test_df, is_train=False, data_dir=data_dir
            )
            # use train avg_psd instead
            avg_psd = load_psd(
                df=self.train_df,
                fs=conf.sampling_info.fs,
                T=conf.sampling_info.T,
                num_workers=conf.num_workers,
                window=conf.wave_prepro_conf.psd.window,
                mode=conf.wave_prepro_conf.psd.avg_mode,
                avg_psd_cache=Path(
                    get_original_cwd(), conf.wave_prepro_conf.psd.cache_path
                ),
                plot_psds=conf.is_debug,
            )
            self.test_dataset = WaveformDataset(
                sample_path_list=self.test_df["path"].to_numpy().tolist(),
                wave_prepro_conf=conf.wave_prepro_conf,
                spec_conf=conf.spec_conf,
                avg_psd=avg_psd,
                targets=None,
                spec_with_gpu=conf.spec_conf.with_gpu,
                T=conf.sampling_info.T,
                fs=conf.sampling_info.fs,
                image_transforms=self.test_transform(),
                is_test=True,
                psd_aug_p=0.0,
            )
            self.plot_dataset(self.test_dataset)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def train_transform(self):
        return self.get_transforms(mode=self.aug_mode)

    def val_transform(self):
        return self.get_transforms(mode=0)

    def test_transform(self):
        return self.get_transforms(mode=0)

    @staticmethod
    def get_spec_size(
        spec_conf: DictConfig, fs: int = 2048, T: float = 2.0
    ) -> Tuple[int, int]:
        spec_type = spec_conf.params_type
        spec_params = spec_conf[spec_type]
        input_width = int((fs * T) // spec_params.hop_length + 1)
        if spec_type == "stft_params":
            input_height = spec_params.n_fft // 2 + 1
        else:
            bins_per_octave = spec_conf.cqt_params.bins_per_octave
            bins_octave_2_size_dict = {
                128: 727,
                96: 546,
                80: 455,
                64: 364,
                48: 273,
                32: 182,
                16: 91,
                8: 46,
            }
            input_height = bins_octave_2_size_dict[bins_per_octave]

        print("\t >> h, w", input_height, input_width)
        return input_height, input_width

    def get_transforms(self, mode: int = 0) -> albu.Compose:

        self.input_height, self.input_width = G2netDatamodule.get_spec_size(
            spec_conf=self.conf.spec_conf
        )

        def horizontal_pad(
            image: np.ndarray,
            input_width: List[int],
            pad_ratio: float = 0.1,
            constant_values: float = 0.0,
            **kwargs,
        ):
            # pad_size = (input_size[0] - image.shape[0], input_size[1] - image.shape[1])
            pad_size = int(input_width * pad_ratio)
            if np.any(np.array(pad_size) > 0):
                image = np.pad(
                    image,
                    [[0, 0], [pad_size, pad_size], [0, 0]],  # h, w, ch
                    constant_values=constant_values,
                )
            return image

        if mode == 0:
            transforms = [
                # albu.Lambda(image=add_pad_img, mask=add_pad_mask, name="padding"),
                albu.Normalize(mean=self.img_mean, std=self.img_std),
            ]
        elif mode == 1:
            add_pad_img = partial(
                horizontal_pad,
                input_width=self.input_width,
                pad_ratio=self.conf.pad_conf.pad_ratio,
                constant_values=self.conf.pad_conf.pad_ratio,
            )
            random_horizontal_pad = albu.Compose(
                [
                    albu.Lambda(image=add_pad_img, name="padding"),
                    albu.RandomCrop(
                        height=self.input_height, width=self.input_width, p=1.0
                    ),
                ],
                p=self.conf.pad_conf.p,
            )
            transforms = [
                random_horizontal_pad,
                albu.Normalize(mean=self.img_mean, std=self.img_std),
            ]
        else:
            raise NotImplementedError
        # if self.conf.gt_as_mask:
        #     additional_targets = {"target_image": "mask"}
        # else:
        #     additional_targets = {"target_image": "image"}

        # composed = albu.Compose(transforms, additional_targets=additional_targets)
        # return composed
        return albu.Compose(transforms)

    def plot_dataset(
        self,
        dataset,
        plot_num: int = 3,
        df: Optional[pd.DataFrame] = None,
        use_clear_event: bool = True,
    ) -> None:
        if use_clear_event:
            ids = np.array([path_2_id(path=path) for path in dataset.sample_path_list])
            clear_ids = ["e33aac5f2a", "098a464da9", "bacb347b79"]
            inds = [np.argmax(ids == id_) for id_ in clear_ids]
        else:
            inds = np.random.choice(len(dataset), plot_num)

        for ind in inds:
            data = dataset[ind]
            im = data["image"].numpy().transpose(1, 2, 0)
            # === PLOT ===
            nrows = 4
            ncols = 3
            site_num = 3
            fig, ax = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=(16, 12),
                sharey=False,
                sharex=False,
            )

            for site_ind in range(site_num):
                ax[0][site_ind].plot(data["sample_data_bp"][site_ind])

            if not self.conf.spec_conf.with_gpu:
                d_specs = WaveformDataset.handle_spec_normalize(
                    img=im,
                    num_abs_channels=site_num,
                    is_encode=False,
                    is_db=self.conf.spec_conf.is_db,
                    db_offset=self.conf.spec_conf.db_conf.db_offset,
                    img_mean=self.img_mean,
                    img_std=self.img_std,
                )
                spec_params = self.conf.spec_conf[self.conf.spec_conf.params_type]
                for site_ind in range(site_num):
                    for i in range(1, nrows):
                        vis_spec(
                            spec=d_specs[i - 1][..., site_ind],  # (time, hz, site_id)
                            y_axis="hz",
                            title=None,
                            fig=fig,
                            ax=ax[i][site_ind],
                            spec_params=spec_params,
                        )

            title = f'id: {data["id"]}' + " " + self.conf.spec_conf.params_type
            has_value = isinstance(data["target"].numpy().tolist(), float)
            if has_value:
                title += " target: " + str(data["target"].numpy().tolist())

            fig.suptitle(title)


def make_split(
    df: pd.DataFrame,
    n_splits: int = 3,
    target_key: str = "target",
    is_reset_index: bool = True,
    verbose: int = 1,
    shuffle: bool = True,
) -> pd.DataFrame:

    if shuffle:
        df = df.sample(frac=1.0)

    if is_reset_index:
        df.reset_index(drop=True, inplace=True)
    df["fold"] = -1
    # gkf = GroupKFold(n_splits=n_splits)
    gkf = StratifiedKFold(n_splits=n_splits)

    for i, (train_idx, valid_idx) in enumerate(gkf.split(df, df[target_key])):
        df.loc[valid_idx, "fold"] = i
    if verbose == 1:
        print(">> check split \n", pd.crosstab(df.fold, df[target_key]))

    return df
