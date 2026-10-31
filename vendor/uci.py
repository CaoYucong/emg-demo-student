import os
from pathlib import Path
from typing import Tuple, List, Union, Dict

import h5py
import numpy as np
import scipy.signal as signal
from scipy.signal import iirnotch

def sequence_to_seconds(seq_len: int, fs: float) -> float:
    """Converts a sequence length in samples to time in seconds.

    Args:
        seq_len (int): The number of samples in the sequence.
        fs (float): The sampling frequency in Hz.

    Returns:
        float: The duration of the sequence in seconds.
    """
    return seq_len / fs

def bandpass_filter_emg(
    emg: np.ndarray,
    lowcut: float = 20.0,
    highcut: float = 90.0,
    fs: float = 200.0,
    order: int = 4
) -> np.ndarray:
    """Applies a Butterworth bandpass filter to the EMG signal.

    Args:
        emg (np.ndarray): The input signal array of shape (T, D).
        lowcut (float, optional): Lower bound of the passband in Hz. Defaults to 20.0.
        highcut (float, optional): Upper bound of the passband in Hz. Defaults to 90.0.
        fs (float, optional): The sampling frequency in Hz. Defaults to 200.0.
        order (int, optional): The order of the filter. Defaults to 4.

    Returns:
        np.ndarray: The filtered signal array.
    """
    nyq = 0.5 * fs
    b, a = signal.butter(order, [lowcut / nyq, highcut / nyq], btype="bandpass")
    return signal.filtfilt(b, a, emg, axis=0)

def notch_filter_emg(
    emg: np.ndarray,
    notch_freq: float = 50.0,
    Q: float = 30.0,
    fs: float = 200.0
) -> np.ndarray:
    """Applies a notch filter to remove power line interference.

    Args:
        emg (np.ndarray): The input signal array of shape (T, D).
        notch_freq (float, optional): The frequency to be removed in Hz. Defaults to 50.0.
        Q (float, optional): The quality factor. Defaults to 30.0.
        fs (float, optional): The sampling frequency in Hz. Defaults to 200.0.

    Returns:
        np.ndarray: The filtered signal array.
    """
    b, a = iirnotch(notch_freq / (0.5 * fs), Q)
    return signal.filtfilt(b, a, emg, axis=0)

def read_emg_txt(txt_path: str) -> np.ndarray:
    """Reads a UCI EMG text file into a numpy array.

    The file is expected to have columns: [time, ch1, ..., ch8, class].

    Args:
        txt_path (str): Path to the .txt file.

    Returns:
        np.ndarray: A float32 array of shape (N, 10).
    """
    data = []
    with open(txt_path, "r") as f:
        for line in f.readlines()[1:]:  # skip header
            cols = line.strip().split()
            if len(cols) == 10:
                data.append(list(map(float, cols)))
    return np.asarray(data, dtype=np.float32)


def preprocess_emg(arr: np.ndarray, fs: float = 200.0, remove_class0: bool = True) -> np.ndarray:
    """Applies a standard preprocessing pipeline to the EMG data.

    Pipeline includes:
    1. Optional removal of rest (class 0).
    2. Bandpass filtering (20-90 Hz).
    3. Notch filtering (50 Hz).
    4. Z-score normalization per channel.

    Args:
        arr (np.ndarray): Raw data array of shape (N, 10).
        fs (float, optional): Sampling frequency in Hz. Defaults to 200.0.
        remove_class0 (bool, optional): Whether to remove the "rest" class. Defaults to True.

    Returns:
        np.ndarray: The preprocessed data array.
    """
    if remove_class0:
        arr = arr[arr[:, -1] >= 1]
    if arr.size == 0:
        return arr

    emg = arr[:, 1:9]  # (N, 8)
    emg = bandpass_filter_emg(emg, 20, 90, fs)
    emg = notch_filter_emg(emg, 50, 30, fs)

    mu = emg.mean(axis=0)
    sd = emg.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    emg = (emg - mu) / sd

    arr[:, 1:9] = emg
    return arr


def find_label_runs(arr: np.ndarray) -> List[Tuple[int, np.ndarray]]:
    """Groups consecutive rows with identical class labels.

    Args:
        arr (np.ndarray): Data array where the last column is the class label.

    Returns:
        List[Tuple[int, np.ndarray]]: A list of tuples (label, sub-array).
    """
    runs = []
    if arr.size == 0:
        return runs
    curr_lbl = int(arr[0, -1])
    start = 0
    for i in range(1, len(arr)):
        lbl = int(arr[i, -1])
        if lbl != curr_lbl:
            runs.append((curr_lbl, arr[start:i]))
            curr_lbl, start = lbl, i
    runs.append((curr_lbl, arr[start:]))
    return runs


def sliding_window_majority(
    seg_arr: np.ndarray,
    window_size: int = 1000,
    stride: int = 500
) -> Tuple[np.ndarray, np.ndarray]:
    """Segments a label-consistent array using a sliding window and majority voting.

    Args:
        seg_arr (np.ndarray): Data array of shape (T, 10).
        window_size (int, optional): Number of samples per window. Defaults to 1000.
        stride (int, optional): Number of samples to shift between windows. Defaults to 500.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - Windowed EMG segments (N, window_size, 8).
            - Majority vote labels (N,).
    """
    segs, labs = [], []
    for start in range(0, len(seg_arr) - window_size + 1, stride):
        win = seg_arr[start : start + window_size]
        maj = np.argmax(np.bincount(win[:, -1].astype(int)))
        segs.append(win[:, 1:9])  # keep 8-channel EMG
        labs.append(maj)
    return np.asarray(segs, dtype=np.float32), np.asarray(labs, dtype=np.int32)


def users_with_gesture(
    data_root: str,
    gesture_id: int,
    subj_range: range = range(1, 37),
    return_counts: bool = False
) -> Union[List[int], Dict[int, int]]:
    """Identifies which subjects performed a specific gesture.

    Args:
        data_root (str): Root directory of the dataset.
        gesture_id (int): The ID of the gesture to search for.
        subj_range (range, optional): Range of subject IDs to check. Defaults to range(1, 37).
        return_counts (bool, optional): If True, returns a dictionary with sample counts.
            Defaults to False.

    Returns:
        Union[List[int], Dict[int, int]]: Either a list of subject IDs or a dictionary
            mapping subject ID to occurrence count.
    """
    found = {}
    for subj in subj_range:
        subj_dir = os.path.join(data_root, f"{subj:02d}")
        if not os.path.isdir(subj_dir):
            continue
        count = 0
        for fname in os.listdir(subj_dir):
            if not fname.endswith(".txt"):
                continue
            txt_path = os.path.join(subj_dir, fname)
            try:
                arr = read_emg_txt(txt_path)
            except Exception:
                # skip files we can't parse
                continue
            if arr.size == 0:
                continue
            # last column is class label (as float). Compare as int.
            if np.any(arr[:, -1].astype(int) == int(gesture_id)):
                # count occurrences (rows) of that gesture in this file
                count += int((arr[:, -1].astype(int) == int(gesture_id)).sum())
        if count > 0:
            found[subj] = count

    if return_counts:
        return found  # dict subj -> count
    else:
        return sorted(found.keys())

def concat_data(lst: List[np.ndarray]) -> np.ndarray:
    """Concatenates a list of data arrays.

    Args:
        lst (List[np.ndarray]): List of arrays to concatenate.

    Returns:
        np.ndarray: Concatenated array or empty array if list is empty.
    """
    return np.concatenate(lst, axis=0) if lst else np.empty((0, 1000, 8), np.float32)

def concat_label(lst: List[np.ndarray]) -> np.ndarray:
    """Concatenates a list of label arrays.

    Args:
        lst (List[np.ndarray]): List of label arrays.

    Returns:
        np.ndarray: Concatenated array or empty array if list is empty.
    """
    return np.concatenate(lst, axis=0) if lst else np.empty((0,), np.int32)

if __name__ == "__main__":
    import argparse

    arg = argparse.ArgumentParser(description="Convert UCI EMG dataset to h5 format.")
    arg.add_argument("--download_data", action="store_true")
    arg.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory of the UCI EMG dataset",
    )
    arg.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory to save the output h5 files",
    )
    arg.add_argument(
        "--seq_len", type=int, help="Size of the window in samples for segmentation."
    )
    arg.add_argument(
        "--stride",
        type=int,
        help="Step size between windows in samples for segmentation.",
    )
    args = arg.parse_args()

    data_root = args.data_dir
    save_root = args.save_dir
    os.makedirs(save_root, exist_ok=True)

    # download data if requested
    if args.download_data:
        # https://archive.ics.uci.edu/dataset/481/emg+data+for+gestures
        base_url = (
            "https://archive.ics.uci.edu/static/public/481/emg+data+for+gestures.zip"
        )
        os.system(f"wget -O {data_root}/emg_gestures.zip '{base_url}'")
        os.system(f"unzip -o {data_root}/emg_gestures.zip -d {Path(data_root).parent}")
        os.system(f"rm {data_root}/emg_gestures.zip")
        print("Dataset downloaded and cleaned up.")

    fs = 200.0  # sampling rate of MYO bracelet
    window_size, stride = args.seq_len, args.stride

    window_seconds = sequence_to_seconds(window_size, fs)
    print(f"Window size: {window_size} samples ({window_seconds:.2f} seconds)")

    split_map = {
        "train": list(range(1, 25)),  # 1–24
        "val": list(range(25, 31)),  # 25–30
        "test": list(range(31, 37)),  # 31–36
    }
    # remove users that performed gesture 7
    gesture_id = 7
    gesture7_users = users_with_gesture(data_root, gesture_id)
    print(f"Users that performed gesture {gesture_id}:", gesture7_users)

    keep_subjs = []
    for k in split_map:
        split_map[k] = [u for u in split_map[k] if u not in gesture7_users]
        keep_subjs.extend(split_map[k])
    print("Updated split map after removing gesture-7 users:", keep_subjs)

    datasets = {k: {"data": [], "label": []} for k in split_map}

    for subj in keep_subjs:
        subj_dir = os.path.join(data_root, f"{subj:02d}")
        if not os.path.isdir(subj_dir):
            continue
        split_key = next(k for k, v in split_map.items() if subj in v)

        for fname in sorted(os.listdir(subj_dir)):
            if not fname.endswith(".txt"):
                continue
            arr = read_emg_txt(os.path.join(subj_dir, fname))
            arr = preprocess_emg(arr, fs)

            for lbl, seg_arr in find_label_runs(arr):
                segs, labs = sliding_window_majority(seg_arr, window_size, stride)
                if segs.size:
                    datasets[split_key]["data"].append(segs)
                    datasets[split_key]["label"].append(labs - 1)

    # concatenate, transpose & save
    for split in ["train", "val", "test"]:
        X = concat_data(datasets[split]["data"])  # (N,256,8)
        y = concat_label(datasets[split]["label"])
        X = X.transpose(0, 2, 1)  # (N,8,256)

        with h5py.File(os.path.join(save_root, f"{split}.h5"), "w") as f:
            f.create_dataset("data", data=X.astype(np.float32))
            f.create_dataset("label", data=y.astype(np.int32))
        uniq, cnt = np.unique(y, return_counts=True)
        print(
            f"{split.upper():5} → X={X.shape}, label dist:",
            dict(zip(uniq.tolist(), cnt.tolist())),
        )

    print("\nAll splits saved to:", save_root)
