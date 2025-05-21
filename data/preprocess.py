import h5py
import mne
import mne_bids
import numpy as np
import os
import typing as tp

from . import utils
from sklearn import preprocessing
import scipy.io

def _process_recording(
        raw, dataset, resample_freq, l_freq, h_freq, notch_freq, subject, session, task, run, save_root, subject_id
    ):

    print("[+] Preprocessing recording...")
    raw = utils.preprocess_meg(raw, resample_freq, l_freq, h_freq, notch_freq)
    data = raw.get_data()
    print("[+] Preprocessed MEG recording of length", data.shape[-1], "samples.")

    sensor_positions = []
    for ch in raw.info["chs"]:
        pos = ch["loc"][:3]
        sensor_positions.append(pos.tolist())

    robust_scaler = preprocessing.RobustScaler()

    print("[+] Fitting robust scaler...")
    robust_scaler = robust_scaler.fit(data.transpose(1, 0)) # [S, T] -> [T, S]
    print("[+] Scaler fitted.")

    info = {
        "subject": subject,
        "session": session,
        "subject_idx": subject_id,
        "task": task,
        "run": run,
        "dataset": dataset,
        "sfreq": resample_freq,
        "sensor_xyz": sensor_positions,
        "robust_scaler_center": robust_scaler.center_,
        "robust_scaler_scale": robust_scaler.scale_,
        "n_samples": data.shape[-1],
    }

    info["channel_means"] = np.mean(data, axis=1)
    info["channel_stds"] = np.std(data, axis=1)

    save_path = f"{save_root}/sub-{subject}_ses-{session}_task-{task}_run-{run}.h5"
    os.makedirs(save_root, exist_ok=True)
    with h5py.File(save_path, "w") as f:
        ds = f.create_dataset("data", data=data, dtype=np.float32, chunks=(data.shape[0], 40))
        for key, value in info.items():
            if value is None:
                continue
            ds.attrs[key] = value
        
    print(f"Finished preprocessing recording.")


def get_raw_from_serialized_libribrain(h5_path):

    with h5py.File(h5_path, "r") as f:
        data = f["data"][:]

        ch_names = f.attrs['channel_names'].split(', ')
        
        info = mne.create_info(
            ch_names=ch_names,
            sfreq=f.attrs["sample_frequency"],
            ch_types=["mag"] * data.shape[0],
        )
        raw = mne.io.RawArray(data, info)

        return raw

def get_raw_from_broderick(mat_path):
    mat_data = scipy.io.loadmat(mat_path)['eegData']
    n_channels = mat_data.shape[1]
    ch_names = [f"EEG_{i}" for i in range(n_channels)]
    info = mne.create_info(
        ch_names=ch_names,
        sfreq=128,
        ch_types=["eeg"] * mat_data.shape[1],
    )
    raw = mne.io.RawArray(mat_data.T, info)
    return raw

def preprocess(
    bids_root,
    resample_freq,
    l_freq,
    h_freq,
    notch_freq,
    save_root,
    subjects: list[str],
    sessions: tp.Optional[list[str]],
    tasks: tp.Optional[list[str]],
    dataset: str,
):

    subject_ids = {sub: i for i, sub in enumerate(subjects)}

    if dataset in ["armeni2022", "libribrain", "gwilliams2022", "broderick2018"]:

        run = None

        for j, subject in enumerate(subjects):
            for k, session in enumerate(sessions):

                for l, task in enumerate(tasks):
                        
                    print(
                        f"Processing subject {j + 1}/{len(subjects)}, session {k + 1}/{len(sessions)}, task {l + 1}/{len(tasks)}."
                    )

                    if dataset == "libribrain":
                        if task == "Sherlock1":
                            if session in ["11", "12"]:
                                run = "2"
                            else:
                                run = "1"
                        else:
                            run = "1"
                    
                    bids_path = mne_bids.BIDSPath(
                        root=bids_root,
                        subject=subject,
                        session=session,
                        task=task,
                        run=run,
                        datatype="meg",
                        suffix="meg",
                    )

                    if dataset == "armeni2022":
                        raw = mne_bids.read_raw_bids(bids_path, verbose=False)
                        raw = utils.pick_armeni_channels(raw)
                    elif dataset == "gwilliams2022":
                        try:
                            raw = mne_bids.read_raw_bids(bids_path, verbose=False)
                            raw = utils.pick_gwilliams_channels(raw)
                        except Exception:
                            print(f"Could not load {bids_path}. Skipping...")
                            continue
                    elif dataset == "libribrain":
                        try:
                            path = f"{bids_root}/sub-{subject}_ses-{session}_task-{task}_run-{run}_proc-bads+headpos+sss+notch+bp+ds_meg.h5"
                            raw = get_raw_from_serialized_libribrain(path)
                        except Exception:
                            print(f"Could not load {path}. Skipping...")
                            continue
                    elif dataset == "broderick2018":
                        try:
                            path = f"{bids_root}/EEG/Subject{subject}/Subject{subject}_Run{session}.mat"
                            raw = get_raw_from_broderick(path)
                        except Exception as e:
                            print(e)
                            print(f"Could not load {path}. Skipping...")
                            continue

                    _process_recording(
                        raw,
                        dataset,
                        resample_freq,
                        l_freq,
                        h_freq,
                        notch_freq,
                        subject,
                        session,
                        task,
                        run,
                        save_root,
                        subject_ids[subject],
                    )

if __name__ == "__main__":
    preprocess(
        bids_root="/data/engs-pnpl/datasets/armeni2022",
        resample_freq=50,
        l_freq=0.1,
        h_freq=40,
        notch_freq=50,
        save_root="/data/engs-pnpl/datasets/armeni2022/derivatives/sentences",
        subjects=[
            "002", "003"
        ],
        # sessions = ["002", "003", "004", "005", "006", "007", "008"],
        sessions=["001", "002", "003", "004", "005", "006", "007", "008", "009", "010"],
        tasks=["compr"],
        dataset="armeni2022"
    )

    preprocess(
        bids_root="/data/engs-pnpl/datasets/LibriBrain/serialized",
        resample_freq=50,
        l_freq=0.1,
        h_freq=40,
        notch_freq=50,
        save_root="/data/engs-pnpl/datasets/LibriBrain/sentences",
        subjects=[
            "0"
        ],
        # sessions=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"],
        sessions=["11", "12"],
        # tasks=["Sherlock1", "Sherlock2", "Sherlock3"],
        tasks=["Sherlock2", "Sherlock3"],
        dataset="libribrain"
    )

    preprocess(
        bids_root="/data/engs-pnpl/datasets/gwilliams2022",
        resample_freq=50,
        l_freq=0.1,
        h_freq=40,
        notch_freq=50,
        save_root="/data/engs-pnpl/datasets/gwilliams2022/derivatives/sentences",
        subjects=[
            "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18",
            "19", "20", "21", "22", "23", "24", "25", "26", "27"
        ],
        sessions=["0", "1"],
        tasks=["0", "1", "2", "3"],
        dataset="gwilliams2022"
    )

    # Read mat data from Broderick
    preprocess(
        bids_root="/data/engs-pnpl/datasets/broderick2018",
        resample_freq=50,
        l_freq=0.1,
        h_freq=40,
        notch_freq=50,
        save_root="/data/engs-pnpl/datasets/broderick2018/derivatives/sentences",
        subjects=[
            "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
            "11", "12", "13", "14", "15", "16", "17", "18", "19"
        ],
        sessions=[
            "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
            "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"
        ],
        tasks=["natural"],
        dataset="broderick2018"
    )