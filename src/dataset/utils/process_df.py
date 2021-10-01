import os
from functools import partial
from multiprocessing import Pool
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.dataset.utils.waveform_preprocessings import preprocess_strain


def id_2_path(
    image_id: str,
    is_train: bool = True,
    data_dir: str = "../input/g2net-gravitational-wave-detection",
) -> str:
    """
    modify from https://www.kaggle.com/ihelon/g2net-eda-and-modeling
    """
    folder = "train" if is_train else "test"
    return "{}/{}/{}/{}/{}/{}.npy".format(
        data_dir, folder, image_id[0], image_id[1], image_id[2], image_id
    )


def path_2_id(path: str) -> str:
    return os.path.basename(path).replace(".npy", "")


def add_dir(df: pd.DataFrame) -> pd.DataFrame:
    df["top_dir"] = df["id"].apply(lambda x: x[0])
    df["bottom_dir"] = df["id"].apply(lambda x: x[:3])
    return df


def add_data_path(
    df: pd.DataFrame,
    is_train: bool = False,
    data_dir: str = "../input/g2net-gravitational-wave-detection",
) -> pd.DataFrame:
    df = add_dir(df=df)
    df["path"] = df["id"].apply(
        lambda x: id_2_path(image_id=x, is_train=is_train, data_dir=data_dir)
    )
    return df


def get_agg_feats(
    path: str,
    interp_psd: Optional[Callable] = None,
    psds: Optional[np.ndarray] = None,
    window: str = "tukey",
    fs: int = 2048,
    fband: List[int] = [10, 912],
    psd_cache_path_suffix: Optional[str] = None,
    T: float = 2.0,
) -> Dict[str, Any]:
    sample_data = np.load(path)
    data_id = path_2_id(path)
    if interp_psd is None:
        for i, strain in enumerate(sample_data):
            _, strain_bp = preprocess_strain(
                strain=strain,
                interp_psd=interp_psd,
                psd=psds[i],
                window=window,
                fs=fs,
                fband=fband,
            )
            sample_data[i] = strain_bp

    mean = sample_data.mean(axis=-1)
    std = sample_data.std(axis=-1)
    minim = sample_data.min(axis=-1)
    maxim = sample_data.max(axis=-1)
    ene = (sample_data ** 2).sum(axis=-1)
    agg_dict = {
        "id": data_id,
        "mean_site0": mean[0],
        "mean_site1": mean[1],
        "mean_site2": mean[2],
        "std_site0": std[0],
        "std_site1": std[1],
        "std_site2": std[2],
        "min_site0": minim[0],
        "min_site1": minim[1],
        "min_site2": minim[2],
        "max_site0": maxim[0],
        "max_site1": maxim[1],
        "max_site2": maxim[2],
        "ene_site0": ene[0],
        "ene_site1": ene[1],
        "ene_site2": ene[2],
    }

    if psd_cache_path_suffix is not None:
        cache_path = path.replace(".npy", psd_cache_path_suffix)
        if os.path.exists(cache_path):
            psd = np.load(cache_path)
            psd_ranges = [10, 35, 350, 500, 912]
            psd_hz_begin = 0
            for psd_hz_end in psd_ranges:
                psd_mean = psd[:, int(psd_hz_begin * T) : int(psd_hz_end * T)].mean(
                    axis=-1
                )
                for site_id, psd_mean_for_site in enumerate(psd_mean):
                    agg_dict[
                        f"psd_{psd_hz_begin}-{psd_hz_end}hz_site{site_id}"
                    ] = psd_mean_for_site

                psd_hz_begin = psd_hz_end

            for site_id, psd_mean_for_site in enumerate(psd.mean(axis=-1)):
                agg_dict[f"psd_all-hz_site{site_id}"] = psd_mean_for_site

    return agg_dict


def get_site_metrics(
    df: pd.DataFrame,
    interp_psd: Optional[Callable] = None,
    psds: Optional[np.ndarray] = None,
    window: str = "tukey",
    fs: int = 2048,
    fband: List[int] = [10, 912],
    psd_cache_path_suffix: Optional[str] = None,
    num_workers: int = 8,
):
    """
    Compute for each id the metrics for each site.
    df: the complete df
    modify from
    https://www.kaggle.com/andradaolteanu/g2net-searching-the-sky-pytorch-effnet-w-meta
    """

    func_ = partial(
        get_agg_feats,
        interp_psd=interp_psd,
        psds=psds,
        window=window,
        fs=fs,
        fband=fband,
        psd_cache_path_suffix=psd_cache_path_suffix,
    )

    if num_workers > 1:
        with Pool(processes=num_workers) as pool:
            agg_dicts = list(
                tqdm(
                    pool.imap(func_, df["path"].tolist()),
                    total=len(df),
                )
            )
    else:
        agg_dicts = []
        for ID, path in tqdm(zip(df["id"].values, df["path"].values)):
            # First extract the cronological info
            agg_dict = func_(path=path)
            agg_dicts.append(agg_dict)

    agg_df = pd.DataFrame(agg_dicts)

    df = pd.merge(df, agg_df, on="id")

    return df
