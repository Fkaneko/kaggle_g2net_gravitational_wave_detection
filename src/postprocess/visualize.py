import os
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nnAudio.Spectrogram import STFT

from src.dataset.utils.spectgram import calc_istft, calc_stft, get_librosa_params
from src.dataset.utils.waveform_preprocessings import preprocess_strain


def visualize_data(sample_data: np.ndarray, title: Optional[str] = None) -> None:
    for i in range(sample_data.shape[0]):
        plt.plot(np.arange(sample_data.shape[1]), sample_data[i], label=str(i))
        plt.legend()
        plt.grid()
        if title is not None:
            plt.title(title)


def vis_from_df(data_id: str, df: pd.DataFrame, is_train: bool = True) -> None:
    target = df[df.id == data_id].to_dict(orient="list")
    sample_data = np.load(target["path"][0])
    title = f'id:{target["id"][0]}'
    if is_train:
        title += f', class:{target["target"][0]}'
    visualize_data(sample_data=sample_data, title=title)


def vis_psd(
    psds: List[np.ndarray], freqs: np.ndarray, labels: Optional[List[str]] = None
):

    plt.figure(figsize=(8, 5))
    # scale x and y axes
    plt.xscale("log", base=2)
    plt.yscale("log", base=10)

    _colors = ["red", "green", "blue", "black", "grey", "magenta"]
    if labels is None:
        labels = list(range(0, len(psds)))
    # plot nowindow, tukey, welch together
    for i, psd in enumerate(psds):
        plt.plot(
            freqs,
            psd,
            _colors[i],
            label=labels[i],
            alpha=0.8,
            linewidth=0.5,
        )

    # plot 1/f^2
    # give it the right starting scale to fit with the rest of the plots
    # don't include zero frequency
    inverse_square = np.array(list(map(lambda f: 1 / (f ** 2), freqs[1:])))
    # inverse starts at 1 to take out 1/0
    scale_index = 500  # chosen by eye to fit the plot
    scale = psds[0][scale_index] / inverse_square[scale_index]
    plt.plot(
        freqs[1:],
        inverse_square * scale,
        "red",
        label=r"$1 / f^2$",
        alpha=0.8,
        linewidth=1,
    )

    #     plt.axis([20, 512, 1e-48, 1e-41])
    plt.axis([20, 2048, 1e-48, 1e-44])
    plt.ylabel("Sn(t)")
    plt.xlabel("Freq (Hz)")
    plt.legend(loc="upper center")
    # plt.title("LIGO PSD data near " + eventname + " at H1")
    plt.show()


def vis_stft(
    strain: np.ndarray,
    stft_params: Dict[str, Any],
    title_1: str,
    lib: str = "librosa",
    is_db: bool = False,
    y_axis: str = "hz",
    amin: float = 1.0e-25,
    top_db: int = 200,
    ref: float = 1.0,
    target: str = "lngDeg_diff_center",
    target_gt: Optional[str] = None,
    time_range: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
    dtype: np.dtype = np.float64,
):

    if lib == "librosa":
        wave_transform = partial(
            librosa.stft, **get_librosa_params(stft_params=stft_params)
        )
    elif lib == "nnAudio":
        wave_transform = STFT(**stft_params)
    else:
        raise NotImplementedError(f"unexpected value for lib: {lib}")

    D_abs, D_theta = calc_stft(
        x=strain,
        wave_transform=wave_transform,
        stft_params=stft_params,
        lib=lib,
        is_db=is_db,
        amin=amin,
        top_db=top_db,
        ref=ref,
        dtype=dtype,
    )
    win_length = stft_params["win_length"]
    sr = stft_params["sr"]
    hop_length = stft_params["hop_length"]

    # n_fft = stft_params["n_fft"]

    # === PLOT ===
    nrows = 4
    ncols = 2
    fig, ax = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(12, 6),
        sharey=False,
        sharex=False,
    )
    fig.suptitle("Log Frequency Spectrogram", fontsize=16)
    # fig.delaxes(ax[1, 2])

    title_text = f"W:{win_length}"
    if is_db:
        title_text = "Axis:dB, " + title_text
    else:
        title_text = "Axis:Lin, " + title_text
    ax[0][0].set_title(target + ", " + title_text, fontsize=10)
    ax[0][1].set_title(title_1, fontsize=10)

    time = np.arange(0, strain.shape[0] / sr, 1.0 / sr)
    ax[0][0].plot(time, strain, label=f"{target}")

    ax[0][0].legend(loc="upper right")
    ax[0][1].legend(loc="upper right")
    ax[0][0].set_xlim(time.min(), time.max())

    for nrow, mat in enumerate([D_abs, np.cos(D_theta), np.sin(D_theta)]):

        img = librosa.display.specshow(
            mat,
            sr=sr,
            hop_length=hop_length,
            x_axis="time",
            y_axis=y_axis,
            cmap="cool",
            ax=ax[nrow + 1][0],
            fmin=stft_params["fmin"],
            fmax=stft_params["fmax"],
        )
        plt.colorbar(img, ax=ax[nrow + 1][0])

    plt.colorbar(img, ax=ax[0][0])
    plt.colorbar(img, ax=ax[0][1])

    if save_path is not None:
        suffix = ".png"
        if time_range is not None:
            suffix = f"time_range{time_range[0]}:{time_range[1]}" + suffix
        suffix = f"win_{win_length}_" + suffix
        if is_db:
            suffix = "_dB_" + suffix
        else:
            suffix = "_Lin_" + suffix

        save_path = save_path.replace(".png", suffix)
        plt.savefig(save_path, dpi=300)
        # plt.close()

    plot_rec(
        strain=strain,
        D_abs=D_abs,
        D_theta=D_theta,
        stft_params=stft_params,
        lib=lib,
        is_db=is_db,
        ref=ref,
        save_path=save_path,
        wave_transform=wave_transform,
    )


def plot_rec(
    strain: np.ndarray,
    D_abs: np.ndarray,
    D_theta: np.ndarray,
    stft_params: Dict[str, Any],
    lib: str = "librosa",
    is_db: bool = True,
    ref: float = 1.0,
    save_path: Optional[str] = None,
    logger: Optional = None,
    log_name: Optional[str] = None,
    logger_name: str = "tensorboard",
    global_step: int = 0,
    x: Optional[np.ndarray] = None,
    target_name: Optional[str] = None,
    wave_transform: Optional[Callable] = None,
) -> None:

    # if is_db:
    #     D_abs = librosa.db_to_amplitude(D_abs, ref=1.0)
    # x_rec = librosa.istft(
    #     np.exp(1j * D_theta) * D_abs,
    #     hop_length=hop_length,
    #     win_length=win_length,
    #     length=length,
    # )
    x_rec = calc_istft(
        D_abs=D_abs,
        D_theta=D_theta,
        stft_params=stft_params,
        lib=lib,
        is_db=is_db,
        ref=ref,
        wave_transform=wave_transform,
    )

    # === PLOT ===
    nrows = 2
    ncols = 1
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(12, 6),
        sharey=False,
        sharex=True,
    )
    rec_error = x_rec - strain
    axes[0].plot(np.abs(rec_error), label="absolute reconstruction error")
    median01 = np.ones_like(strain) * np.median(np.abs(strain)) * 0.1
    rec_error_rate = np.where(
        strain == 0,
        (rec_error / np.percentile(np.abs(strain), 1)) * 100,
        (rec_error / strain) * 100,
    )

    percentile = np.ones_like(strain) * np.percentile(np.abs(strain), 5) * 0.1
    axes[1].plot(rec_error_rate, label="reconstrucation rate")
    axes[1].set_ylabel("Error Rate[%]")
    axes[1].set_xlabel("Time Step")
    axes[1].grid()
    axes[1].legend()

    axes[0].plot(median01, label="median x 0.1")
    axes[0].plot(percentile, label="percentaile5")
    if x is not None:
        axes.plot(np.abs(x - strain), label="pred-baseline")
    rec_err = np.max(np.abs(x_rec - strain))
    axes[0].set_yscale("log")
    axes[0].set_ylabel("mean abs error")
    axes[0].set_xlabel("time step")
    axes[0].grid()
    axes[0].legend()

    title = "stft reconstruction check"
    if target_name is not None:
        title = target_name + ": " + title
    fig.suptitle(title)

    if save_path is not None:
        recon_dir = os.path.join(os.path.dirname(save_path), "recon")
        print(
            f"{recon_dir}: max_abs_error {rec_err:4.3e}, median01: {median01[0]:4.3e}"
            f" percentile: {percentile[0]:4.3e}"
        )
        os.makedirs(recon_dir, exist_ok=True)
        save_path = os.path.join(recon_dir, os.path.basename(save_path))
        plt.savefig(save_path, dpi=300)

    if logger is not None:
        axes.set_ylim(1.0e-9, 1.0e-4)
        if logger_name == "tensorboard":
            logger.experiment.add_figure(
                "sequernce_prediction",
                fig,
                global_step=global_step,
            )
        elif logger_name == "neptune":
            logger.experiment[log_name].log(fig)

    # plt.close()


def show_spec_sample_catalog(
    df: pd.DataFrame,
    wave_transform: Callable,
    spec_params: Dict[str, Any],
    catalog_inds: np.ndarray,
    interp_psd: Optional[Callable] = None,
    psds: Optional[np.ndarray] = None,
    lib: str = "nnAudio",
    fs: int = 2048,
    window: str = "tukey",
    fband: List[int] = [10, 912],
    site_ind: int = 0,
    ref: float = 1.0,
    amin: float = 1e-25,
    top_db: int = 200,
    y_axis: str = "hz",
    dtype: np.dtype = np.float32,
    is_db: bool = True,
    sample_id: Optional[str] = None,
    sweep_param: str = "win_length",
    spec_func: Callable = None,
    print_min_max_db: bool = False,
    skip_whiten: bool = False,
    skip_bandpass: bool = False,
) -> None:
    nrows = catalog_inds.shape[0]
    ncols = catalog_inds.shape[1]
    fig, ax = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        sharex=True,
        sharey=True,
        figsize=(int(25 * ncols / 5), int(18 * nrows / 5)),
    )
    spec_min_db = 0
    spec_max_db = -150

    for i in range(nrows):
        for j in range(ncols):

            if sample_id is not None:
                sample = df.loc[df.id == sample_id]
                assert len(sample) == 1
                sample = sample.iloc[0]
                spec_params[sweep_param] = catalog_inds[i, j]
                wave_transform = spec_func(**spec_params)
                title = (
                    f"{spec_func.__name__}, {sweep_param}:{spec_params[sweep_param]}"
                )

            else:
                sample = df.iloc[catalog_inds[i, j]]
                title = f'id: {sample["id"]} target: {sample["target"]}'

            sample_data = np.load(sample["path"])
            strain = sample_data[site_ind].squeeze()

            strain_wh, strain_bp = preprocess_strain(
                strain=strain,
                interp_psd=interp_psd,
                psd=psds[site_ind],
                window=window,
                skip_whiten=skip_whiten,
                skip_bandpass=skip_bandpass,
                fs=fs,
                fband=fband,
            )
            spec_bp, D_theta = calc_stft(
                x=strain_bp,
                wave_transform=wave_transform,
                stft_params=spec_params,
                lib=lib,
                is_db=is_db,
                amin=amin,
                top_db=top_db,
                ref=ref,
                dtype=dtype,
            )

            vis_spec(
                spec=spec_bp,
                y_axis=y_axis,
                ax=ax[i][j],
                fig=fig,
                title=title,
                spec_params=spec_params,
            )
            spec_min_db = min(spec_min_db, np.min(spec_bp))
            spec_max_db = max(spec_max_db, np.max(spec_bp))
    if print_min_max_db:
        print(f"spec min db: {spec_min_db :>6.1f}" f"spec max db: {spec_max_db :>6.1f}")


def vis_spec(
    spec: np.ndarray,
    y_axis: str = "cqt_note",
    title: str = "Constant-Q power spectrum",
    fig=None,
    ax=None,
    spec_params: Dict[str, Any] = {"n_fft": 512},
) -> Any:
    keys = ["sr", "fmin," "fmax", "hop_length", "bins_per_octave"]
    spec_params = {key: value for key, value in spec_params.items() if key in keys}
    if ax is None:
        fig, ax = plt.subplots()
    img = librosa.display.specshow(
        spec,
        x_axis="time",
        y_axis=y_axis,
        ax=ax,
        cmap="cool",
        **spec_params,
    )
    ax.set_title(title)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    return img
