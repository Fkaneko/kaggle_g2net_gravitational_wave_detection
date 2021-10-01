from typing import Any, Callable, Dict, Optional, Tuple

import librosa
import librosa.display
import numpy as np
import torch
from nnAudio.Spectrogram import iSTFT


def calc_stft(
    x: np.ndarray,
    wave_transform: Callable,
    stft_params: Dict[str, Any],
    lib: str = "librosa",
    is_db: bool = False,
    is_pad: bool = False,
    amin: float = 1.0e-15,
    top_db: int = 200,
    ref: float = 1.0,
    dtype: np.dtype = np.float64,
) -> Tuple[np.ndarray, np.ndarray]:

    if lib == "librosa":
        x = x.astype(dtype)
    elif lib == "nnAudio":
        x = torch.tensor(x.astype(np.float32))

    # need ref change
    length = x.shape[0]
    if is_pad:
        x = librosa.util.fix_length(x, length + stft_params["n_fft"] // 2)

    # Short-time Fourier transform (STFT)

    D = wave_transform(x)
    if isinstance(D, torch.Tensor):
        D = D.squeeze()
        D_abs = torch.sqrt(D[..., 0].pow(2) + D[..., 1].pow(2)).numpy()
        D_theta = torch.atan2(D[..., 1] + 0.0, D[..., 0]).numpy()
    else:
        D_abs = np.abs(D)
        D_theta = np.angle(D)

    # Convert an amplitude spectrogram to Decibels-scaled spectrogram.
    if is_db:
        D_abs = librosa.amplitude_to_db(D_abs, ref=ref, amin=amin, top_db=top_db)
        D_theta = np.where(D_abs == (D_abs.max() - top_db), 0.0, D_theta)
    return D_abs, D_theta


def get_librosa_params(stft_params: Dict[str, Any]) -> Dict[str, Any]:
    keys = ["n_fft", "hop_length", "win_length"]
    stft_params = {key: value for key, value in stft_params.items() if key in keys}
    return stft_params


def calc_istft(
    D_abs: np.ndarray,
    D_theta: np.ndarray,
    stft_params: Dict[str, Any],
    lib: str = "librosa",
    is_db: bool = False,
    ref: float = 1.0,
    wave_transform: Optional[Callable] = None,
) -> np.ndarray:
    if is_db:
        D_abs = librosa.db_to_amplitude(D_abs, ref=ref)

    D = np.exp(1j * D_theta) * D_abs
    # Short-time Fourier transform (STFT)
    if lib == "librosa":
        stft_params = get_librosa_params(stft_params=stft_params)
        _ = stft_params.pop("n_fft")
        x_rec = librosa.istft(D, **stft_params)

    elif lib == "nnAudio":
        if isinstance(D_abs, np.ndarray):
            D = np.stack([np.real(D), np.imag(D)], axis=-1)
            if D.ndim == 3:
                D = D[np.newaxis]

            D = torch.Tensor(D.astype(np.float32))

        if wave_transform is None:
            _ = stft_params.pop("iSTFT")
            _ = stft_params.pop("pad_mode")
            _ = stft_params.pop("output_format")
            _ = stft_params.pop("trainable")
            wave_transform = iSTFT(**stft_params)
            # wave_func = STFT(**stft_params, iSTFT=True)
            x_rec = wave_transform(D, onesided=True, length=4096)
        else:
            x_rec = wave_transform.inverse(D)
        x_rec = x_rec.numpy().squeeze()
    else:
        raise NotImplementedError(f"unexpected value for lib: {lib}")

    return x_rec
