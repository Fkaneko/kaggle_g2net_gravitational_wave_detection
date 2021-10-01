import os
import random
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import matplotlib.mlab as mlab
import numpy as np
import pandas as pd
import scipy
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from tqdm import tqdm


def get_window(nfft: int, window: str = "tukey") -> Tuple[np.ndarray, np.ndarray]:
    # NFFT = 4 * fs  # Use 4 seconds of data for each fourier transform
    NOVL = (
        1 * nfft / 2
    )  # The number of points of overlap between segments used in Welch averaging
    if window == "tukey":
        window = scipy.signal.tukey(nfft, alpha=1.0 / 4)
    else:
        raise NotImplementedError

    return NOVL, window


# from
# https://github.com/losc-tutorial/Data_Guide/blob/master/Guide_Notebook.ipynb
def get_psd(
    strain: np.ndarray,
    nfft: int = None,
    nfft_factor: float = 1.0,
    fs: int = 2048,
    use_overlap: bool = False,
    window: str = "tukey",
    return_interp: bool = True,
) -> interp1d:

    if nfft is None:
        assert nfft_factor <= 1.0
        NFFT = int(strain.shape[0] * nfft_factor)
    else:
        NFFT = nfft
    NOVL_short, psd_window_short = get_window(nfft=NFFT, window=window)

    if not use_overlap:
        NOVL_short = None
    # psd using a tukey window but no welch averaging
    psd, freqs = mlab.psd(
        strain,
        Fs=fs,
        NFFT=NFFT,
        window=psd_window_short,
        noverlap=NOVL_short,
    )
    if return_interp:
        # We will use interpolations of the PSDs computed above for whitening:
        interp_psd = interp1d(freqs, psd)
    else:
        interp_psd = None

    return interp_psd, psd, freqs


def get_psd_from_path(
    path: str,
    nfft_factor=1.0,
    fs: int = 2048,
    use_overlap=False,
    window: str = "tukey",
    return_interp=False,
    cache_path_suffix: str = None,
):

    if cache_path_suffix is not None:
        cache_path = path.replace(".npy", cache_path_suffix)
        if os.path.exists(cache_path):
            return np.load(cache_path)

    sample_data = np.load(path)
    # psd return only postive freq part + nyqyst, so shape[1]//2 + 1
    sample_psd = np.zeros_like(sample_data[:, : sample_data.shape[1] // 2 + 1])
    for i, strain in enumerate(sample_data):
        _, psd, freqs = get_psd(
            strain=strain,
            nfft_factor=nfft_factor,
            fs=fs,
            use_overlap=use_overlap,
            window=window,
            return_interp=return_interp,
        )
        sample_psd[i] = psd
    if cache_path_suffix is not None:
        np.save(path.replace(".npy", cache_path_suffix), sample_psd)
    return sample_psd


# form
# https://github.com/losc-tutorial/Data_Guide/blob/master/Guide_Notebook.ipynb
def whiten(
    strain,
    dt,
    phase_shift=0,
    time_shift=0,
    interp_psd: Optional[np.ndarray] = None,
    psd: Optional[np.ndarray] = None,
):
    """Whitens strain data given the psd and sample rate, also applying a phase
    shift and time shift.

    Args:
        strain (ndarray): strain data
        interp_psd (interpolating function): function to take in freqs and output
            the average power at that freq
        dt (float): sample time interval of data
        phase_shift (float, optional): phase shift to apply to whitened data
        time_shift (float, optional): time shift to apply to whitened data (s)

    Returns:
        ndarray: array of whitened strain data
    """
    if (interp_psd is None) and (psd is None):
        raise ValueError(" both interp_psd and psd should not be None or not None")
    elif (interp_psd is not None) and (psd is not None):
        raise ValueError(" both interp_psd and psd should not be None or not None")

    hf, freqs = real_fft(strain=strain, dt=dt)

    if psd is None:
        psd = interp_psd(freqs)

    # apply time and phase shift
    hf = hf * np.exp(-1.0j * 2 * np.pi * time_shift * freqs - 1.0j * phase_shift)
    norm = 1.0 / np.sqrt(1.0 / (dt * 2))
    white_hf = hf / np.sqrt(psd) * norm
    white_ht = np.fft.irfft(white_hf, n=len(strain))
    return white_ht


# from
# https://github.com/losc-tutorial/Data_Guide/blob/master/Guide_Notebook.ipynb
def real_fft(
    strain: np.ndarray, dt: float = 1.0 / 2048.0
) -> Tuple[np.ndarray, np.ndarray]:
    Nt = len(strain)
    # take the fourier transform of the data
    freqs = np.fft.rfftfreq(Nt, dt)

    # whitening: transform to freq domain, divide by square root of psd, then
    # transform back, taking care to get normalization right.
    hf = np.fft.rfft(strain)
    return hf, freqs


# form
# https://github.com/losc-tutorial/Data_Guide/blob/master/Guide_Notebook.ipynb
def bandpass(strain, fband, fs):
    """Bandpasses strain data using a butterworth filter.

    Args:
        strain (ndarray): strain data to bandpass
        fband (ndarray): low and high-pass filter values to use
        fs (float): sample rate of data
    another ref:
        https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
    Returns:
        ndarray: array of bandpassed strain data
    """
    bb, ab = butter(4, [fband[0] * 2.0 / fs, fband[1] * 2.0 / fs], btype="band")
    normalization = np.sqrt((fband[1] - fband[0]) / (fs / 2))
    strain_bp = filtfilt(bb, ab, strain) / normalization
    return strain_bp


def preprocess_strain(
    strain: np.ndarray,
    interp_psd: Optional[Callable] = None,
    psd: Optional[np.ndarray] = None,
    window: Union[str, np.ndarray] = "tukey",
    fs: int = 2048,
    fband: List[int] = [10, 912],
    skip_whiten: bool = False,
    skip_bandpass: bool = False,
    aug_p: float = 0.0,
    aug_mean: float = 1.0,
    aug_sigma_x3: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(window, str):
        _, window = get_window(nfft=strain.shape[0], window=window)
    if skip_whiten:
        # for float32 calculation, espicially spec**2
        strain_wh = strain / 1.0e-20
    else:
        if random.random() < aug_p:
            factor = np.random.normal(loc=aug_mean, scale=aug_sigma_x3 / 3.0)
        else:
            factor = 1.0
        strain_w = strain * window
        strain_wh = whiten(
            strain_w,
            interp_psd=interp_psd,
            dt=1.0 / fs,
            phase_shift=0,
            time_shift=0,
            psd=psd * factor,
        )
    if skip_bandpass:
        strain_bp = strain_wh
    else:
        strain_bp = bandpass(strain_wh, fband, fs)
    return strain_wh, strain_bp


def calc_avg_psd(
    df: pd.DataFrame,
    fs: int = 2048,
    T: float = 2.0,
    num_workers: int = 8,
    cache_path_suffix: str = None,
    chunk_size: int = 100000,
    site_num: int = 3,
    nfft_factor: float = 1.0,
    use_overlap=False,
    window: str = "tukey",
    return_interp: bool = False,
    mode: str = "all",
) -> np.ndarray:

    if mode == "all":
        pass
    elif mode == "signal":
        path_lists = df.loc[df["target"] == 1, "path"].tolist()
    elif mode == "noise":
        path_lists = df.loc[df["target"] == 0, "path"].tolist()
    else:
        raise ValueError(f"Unexpected value for mode: {mode}")

    # including nyqust freq
    avg_psd = np.zeros((site_num, int(fs // 2 * T) + 1), dtype=np.float64)

    func_ = partial(
        get_psd_from_path,
        nfft_factor=nfft_factor,
        fs=fs,
        use_overlap=use_overlap,
        window=window,
        return_interp=return_interp,
        cache_path_suffix=cache_path_suffix,
    )
    split_lists = np.array_split(path_lists, len(path_lists) // chunk_size)
    for paths in tqdm(split_lists, total=len(path_lists) // chunk_size):
        if num_workers > 1:
            with Pool(processes=num_workers) as pool:
                results = pool.map(func_, paths)
        else:
            results = []
            for path in paths:
                res = func_(path=path)
                results.append(res)

        results = np.stack(results, axis=0)
        avg_psd += np.sum(results, axis=0) / len(path_lists)
        # plt.plot(avg_psd[0]*len(df)/1000);
        # for j in range(3):
        #     plt.plot(results[j][0], alpha=0.3);
        # plt.yscale("log");
        # plt.xscale("log", base=2);
        # plt.xlim(2**5, 2**9)
    return avg_psd


def load_psd(
    df: pd.DataFrame,
    fs: int = 2048,
    T: float = 2.0,
    num_workers: int = 16,
    window: str = "tukey",
    mode: str = "all",
    avg_psd_cache: Union[str, Path] = "./avg_psd.npy",
    plot_psds: bool = False,
) -> np.ndarray:
    if isinstance(avg_psd_cache, Path):
        avg_psd_cache = str(avg_psd_cache)

    avg_psd_cache = avg_psd_cache.replace(".npy", f"_{mode}.npy")
    if os.path.exists(avg_psd_cache):
        avg_psd = np.load(avg_psd_cache)
    else:
        avg_psd = calc_avg_psd(
            df=df,
            fs=fs,
            T=T,
            num_workers=num_workers,
            cache_path_suffix="_psd.npy",
            chunk_size=10000,
            nfft_factor=1.0,
            use_overlap=False,
            window=window,
            return_interp=False,
            mode=mode,
        )
        os.makedirs(os.path.dirname(avg_psd_cache), exist_ok=True)
        np.save(avg_psd_cache, avg_psd)

    # if plot_psds:
    #     freqs = np.linspace(0, fs / 2, T * fs // 2 + 1)
    #     vis_psd(psds=[psd.squeeze() for psd in np.split(avg_psd, 3)], freqs=freqs)
    return avg_psd
