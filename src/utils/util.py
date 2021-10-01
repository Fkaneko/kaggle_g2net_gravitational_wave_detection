import csv
import os
import random
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm


def set_random_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    # set Python random seed
    random.seed(seed)
    # set NumPy random seed
    np.random.seed(seed)


def request_and_save(
    url: str, is_save: bool = True, save_path: Optional[str] = None
) -> None:
    """
    download and save it on save_path
    if thre is cache, automatically use cache
    """
    if save_path is None:
        save_path = url.split("/")[-1]

    if os.path.exists(save_path):
        print(f"use chached file for {url}")
        return save_path

    print(f"download from {url}")
    print(f"and save it  {save_path}")
    r = requests.get(url=url)
    # for 404 error
    r.raise_for_status()
    with open(save_path, "wb") as f:
        f.write(r.content)


def logging_download(
    csv_path: str, mode: str = "w", log_msg: List[str] = ["row_ind", "ID"]
):
    with open(csv_path, mode, newline="") as f:
        writer = csv.writer(f)
        writer.writerow(log_msg)


def download_with_logging(
    download_func: Callable,
    download_list: List[str],
    logging_dir: str,
    is_debug: bool = False,
    restart_download: bool = False,
):
    """
    usage:
    download_func = partial(
        xx_function,
        save_dir=save_dir,
        dataset=dataset,
    )
    download_with_logging(
        download_func=download_func,
        download_list=download_list.tolist(),
        logging_dir=save_dir,
        is_debug=False,
    )
    """
    if restart_download:
        downloaded_df = pd.read_csv(
            os.path.join(logging_dir, "downloaded.csv"), header=["row_ind", "ID"]
        )
        dowonloaded_target = downloaded_df["ID"].to_numpy().tolist()
        download_list = list(set(download_list) - set(dowonloaded_target))

    for i, target in tqdm(enumerate(download_list), total=len(download_list)):
        try:
            download_func(target)
            logging_download(
                csv_path=os.path.join(logging_dir, "downloaded.csv"),
                mode="a",
                log_msg=[i, target],
            )

        except Exception as e:
            print(f"failed to download: {str(target)}")
            logging_download(
                csv_path=os.path.join(logging_dir, "un_downloaded.csv"),
                mode="a",
                log_msg=[i, str(target), str(e)],
            )
        if is_debug:
            if i == 5:
                break
