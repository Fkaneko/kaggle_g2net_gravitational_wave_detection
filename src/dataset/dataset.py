import random
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import albumentations as albu
import librosa
import numpy as np
import torch
from nnAudio.Spectrogram import STFT, CQT1992v2
from omegaconf import OmegaConf
from scipy.interpolate import interp1d

from src.dataset.utils.process_df import path_2_id
from src.dataset.utils.spectgram import calc_stft, get_librosa_params
from src.dataset.utils.waveform_preprocessings import (
    get_psd,
    get_window,
    preprocess_strain,
)


class WaveformDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        sample_path_list: List[str],
        wave_prepro_conf: OmegaConf,
        spec_conf: OmegaConf,
        avg_psd: np.ndarray,
        targets: Optional[List[int]] = None,
        spec_with_gpu: bool = False,
        T: float = 2.0,
        fs: int = 2048,
        image_transforms: albu.Compose = None,
        is_test: bool = False,
        rand_ratio: float = 0.0,
        rand_freq: float = 0.0,
        sigma: float = 5.0e-5,
        psd_aug_p: float = 0.0,
        psd_aug_mean: float = 1.0,
        psd_aug_sigma_x3: float = 0.3,
        use_ext_psd: bool = False,
        ext_psd_path_list: Optional[List[str]] = None,
        is_psd_denoise: bool = False,
    ):
        self.sample_path_list = sample_path_list
        self.wave_prepro_conf = wave_prepro_conf
        self.avg_psd = avg_psd
        self.targets = targets
        self.spec_with_gpu = spec_with_gpu
        self.T = T
        self.fs = fs
        self.is_test = is_test
        self.image_transforms = image_transforms
        self.rand_ratio = rand_ratio
        self.rand_freq = rand_freq
        self.sigma = sigma
        self.ext_psd_path_list = ext_psd_path_list
        self.use_ext_psd = use_ext_psd

        self.psd_aug_p = psd_aug_p
        self.psd_aug_mean = psd_aug_mean
        self.psd_aug_sigma_x3 = psd_aug_sigma_x3

        self.is_psd_denoise = is_psd_denoise
        freqs = np.linspace(0, fs / 2, int(T) * fs // 2 + 1)
        self.interp_avg_psd = []
        for site_ind in range(avg_psd.shape[0]):
            self.interp_avg_psd.append(
                interp1d(freqs, avg_psd[site_ind], fill_value="extrapolate")
            )

        if self.rand_freq > 0.0:
            print(
                '"xxx_diff_prev" inputs are assumed for the current wave augmentation'
            )

        self.max = 0.0
        self.min = 255.0

        _, self.whiten_window = get_window(
            nfft=int(T * fs), window=wave_prepro_conf.whiten.window
        )
        if not spec_with_gpu:
            self.spec_params = spec_conf[spec_conf.params_type]
            self.db_conf = spec_conf.db_conf
            self.is_db = spec_conf.is_db
            self.spec_lib = spec_conf.spec_lib

            if spec_conf.params_type == "stft_params":
                if self.spec_lib == "librosa":
                    self.wave_transform = partial(
                        librosa.stft, **get_librosa_params(stft_params=self.spec_params)
                    )
                elif self.spec_lib == "nnAudio":
                    self.wave_transform = STFT(**self.spec_params)

            elif spec_conf.params_type == "cqt_params":
                self.wave_transform = CQT1992v2(**self.spec_params)
            else:
                raise NotImplementedError(
                    f"unexpected value for params_type: {spec_conf.params_type}"
                )

    def __len__(self):
        return len(self.sample_path_list)

    @staticmethod
    def handle_spec_normalize(
        img: np.ndarray,
        num_abs_channels: int = 3,
        is_encode: bool = False,
        is_db: bool = True,
        db_offset: int = 130,
        img_std: np.ndarray = None,
        img_mean: np.ndarray = None,
        gt_as_mask: bool = False,
        image_fomat: str = "matplot",
    ):
        """
        encode: normalize spectgraom within [0, 255] or [0, 1]
        """
        assert img.shape[-1] >= num_abs_channels
        acceptable_types = (np.ndarray, torch.Tensor)
        if is_encode:

            if image_fomat == "matplot":
                if is_db:
                    img[..., :num_abs_channels] = (
                        img[..., :num_abs_channels] + db_offset
                    )
                img[..., num_abs_channels:] = (
                    (img[..., num_abs_channels:] + 1.0) * 0.5 * 255.0
                )
            elif image_fomat == "torch":
                if is_db:
                    img[:, :num_abs_channels] = img[:, :num_abs_channels] + db_offset
                img[:, num_abs_channels:] = (
                    (img[:, num_abs_channels:] + 1.0) * 0.5 * 255.0
                )

            if gt_as_mask:
                return img / 255.0
            else:
                return img
        else:
            if not (
                isinstance(img_mean, acceptable_types)
                or isinstance(img_std, acceptable_types)
            ):
                raise TypeError
            if img.dtype == np.float16:
                img = img.astype(np.float32)

            if not gt_as_mask:
                img = img * img_std + img_mean
            img *= 255.0
            if image_fomat == "matplot":
                if is_db:
                    D_abs = img[..., :num_abs_channels] - db_offset

                D_cos = (
                    img[..., num_abs_channels : num_abs_channels * 2] * 2.0 / 255.0
                    - 1.0
                )
                D_sin = (
                    img[..., num_abs_channels * 2 : num_abs_channels * 3] * 2.0 / 255.0
                    - 1.0
                )
            elif image_fomat == "torch":
                if is_db:
                    D_abs = img[:, :num_abs_channels] - db_offset
                D_cos = (
                    img[:, num_abs_channels : num_abs_channels * 2] * 2.0 / 255.0 - 1.0
                )
                D_sin = (
                    img[:, num_abs_channels * 2 : num_abs_channels * 3] * 2.0 / 255.0
                    - 1.0
                )

            return D_abs, D_cos, D_sin

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
    ) -> np.ndarray:

        D_abs, D_cos, D_sin = [], [], []
        for site_ind, strain in enumerate(sample_data):
            abs_, theta_ = calc_stft(
                x=strain,
                wave_transform=wave_transform,
                stft_params=spec_params,
                lib=lib,
                is_db=is_db,
                amin=amin,
                top_db=top_db,
                ref=ref,
                # dtype=np.float32,
            )
            D_abs.append(abs_)
            D_cos.append(np.cos(theta_))
            D_sin.append(np.sin(theta_))

        img = np.stack(D_abs + D_cos + D_sin, axis=-1)

        img = WaveformDataset.handle_spec_normalize(
            img=img,
            num_abs_channels=sample_data.shape[0],
            is_encode=True,
            is_db=is_db,
            db_offset=db_offset,
        )

        return img

    def __getitem__(self, idx: int):
        sample_path = self.sample_path_list[idx]

        sample_data = np.load(sample_path)
        psds = self.avg_psd

        sample_data_bp = np.zeros_like(sample_data)
        # 512
        nfft_for_psd = sample_data.shape[1] // 4
        site_psds = np.ones((sample_data.shape[0], nfft_for_psd // 2), dtype=np.float64)
        for site_ind, strain in enumerate(sample_data):
            if self.is_psd_denoise:
                _, site_psd, freqs = get_psd(
                    strain=strain,
                    nfft=nfft_for_psd,
                    fs=self.fs,
                    use_overlap=True,
                    return_interp=False,
                )
                # normalize -1 < psd < 1 for tanh activation, get 512 not 513
                site_psds[site_ind] = (
                    site_psd / self.interp_avg_psd[site_ind](freqs)
                ).clip(1e-3, 1e3)[:-1]
            strain_wh, strain_bp = preprocess_strain(
                strain=strain.squeeze(),
                interp_psd=None,
                psd=psds[site_ind],
                window=self.whiten_window,
                fs=self.fs,
                fband=self.wave_prepro_conf.bandpass.range,
                skip_bandpass=self.wave_prepro_conf.skip_bandpass,
                skip_whiten=self.wave_prepro_conf.skip_whiten,
                aug_p=self.psd_aug_p,
                aug_mean=self.psd_aug_mean,
                aug_sigma_x3=self.psd_aug_sigma_x3,
            )
            sample_data_bp[site_ind] = strain_bp

        if self.spec_with_gpu:
            img = torch.empty((3, 1, 1), dtype=torch.float32)
        else:
            img = self.generate_image_from_wave(
                sample_data=sample_data_bp,
                wave_transform=self.wave_transform,
                spec_params=self.spec_params,
                lib=self.spec_lib,
                is_db=self.is_db,
                amin=self.db_conf.amin,
                top_db=self.db_conf.top_db,
                db_offset=self.db_conf.db_offset,
                ref=self.db_conf.ref,
            )
            augmented = self.image_transforms(image=img)
            img = augmented["image"]
            img = torch.from_numpy(img.transpose(2, 0, 1))

        if self.is_test:
            target = torch.empty(0)
        else:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)

        # psd normalize 0.0 < psd < 1.0, and 512 dim for linear layer
        site_psds = torch.tensor((np.log10(site_psds) + 3.0) / 6.0, dtype=torch.float32)
        sample_data_bp = torch.tensor(sample_data_bp, dtype=torch.float32)
        return {
            "id": path_2_id(sample_path),
            "sample_data_bp": sample_data_bp,
            "image": img,
            "target": target,
            "site_psds": site_psds,
        }
