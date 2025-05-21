import dataclasses
import glob
import h5py
import numpy as np
import torch
import tqdm
import json
import os
import pandas as pd

from collections import defaultdict

from data.preprocess import preprocess
from data import utils

@dataclasses.dataclass
class Sample:
    h5_dataset: h5py.Dataset
    chunk: dict

def _scale_meg(meg_data, robust_scaler_center, robust_scaler_scale, threshold, sfreq):
    # Scale and center the data such that [-1, 1] is in IQR [0.25, 0.75]
    meg_data -= robust_scaler_center[:, None]
    meg_data /= robust_scaler_scale[:, None]

    # Clamp outliers above +threshold and below -threshold
    meg_data[np.abs(meg_data) > threshold] = (
        np.sign(meg_data[np.abs(meg_data) > threshold]) * threshold
    )

    # Apply baseline correction using the first 0.5 seconds of data
    meg_data -= np.mean(meg_data[..., :round(0.5 * sfreq)], axis=-1, keepdims=True)

    return meg_data

def find_top_words(bids_root, top_words):

    if "LibriBrain" in bids_root:
        words = utils.get_all_libribrain_words(bids_root)
    elif "gwilliams2022" in bids_root:
        words = utils.get_all_gwilliams_words(bids_root)
    elif "broderick2018" in bids_root:
        words = utils.get_all_broderick_words(bids_root)
    else:
        words = utils.get_all_armeni_words(bids_root)

    if top_words == 0:
        top_words = len(set(words))
    
    # Count the frequency of each word
    word_freqs = defaultdict(int)
    for word in words:
        word_freqs[word] += 1
    
    top_word_freqs = sorted(word_freqs.items(), key=lambda x: x[1], reverse=True)[:top_words]
    top_words_map = {w: i for i, (w, _) in enumerate(top_word_freqs)}

    other_words = list(set(words).difference(top_words_map.keys()))

    return top_words_map, other_words

class MEGDataset(torch.utils.data.Dataset):
    def __init__(
        self, bids_root, save_root, subjects, sessions,
        tasks, dataset, top_words_map, dataset_id=0, context=64, overlap=32, subject_id_shift=0,
        tmin=-0.5, tmax=2.5, debug=False,
    ):
        super().__init__()

        self.context = context
        self.top_words_map = top_words_map
        self.dataset_id = dataset_id
        self.subject_id_shift = subject_id_shift
        self.tmin = tmin
        self.tmax = tmax
        self.dataset = dataset

        if self.dataset == "libribrain":
            if os.path.exists("./data/libribrain_sensor_xyz.json"):
                with open("./data/libribrain_sensor_xyz.json") as f:
                    self.sensor_xyz = np.array(json.load(f))
            else:
                self.sensor_xyz = None
        elif self.dataset == "broderick2018":
            self.sensor_xyz = pd.read_csv('./data/broderick2018_sensor_xyz.csv', header=None).to_numpy()
        
        paths = sorted(glob.glob(f"{save_root}/*.h5"))

        if not paths:
            print(f"No preprocessed data found in {save_root}. Preprocessing data...")
            preprocess(
                bids_root,
                resample_freq=50,
                l_freq=0.1,
                h_freq=40,
                notch_freq=50,
                save_root=save_root,
                subjects=subjects,
                sessions=sessions,
                tasks=tasks,
                dataset=dataset,
            )
            paths = sorted(glob.glob(f"{save_root}/*.h5"))
        
        self.h5_datasets = [h5py.File(p, "r")["data"] for p in paths]

        self.samples = []

        self.subjects = subjects
        self.sessions = sessions

        n_datasets = 0
        self.seconds = 0

        for h5_dataset in tqdm.tqdm(self.h5_datasets):

            if n_datasets > 0 and debug:
                break

            info = dict(h5_dataset.attrs)
            subject = info["subject"]
            session = info.get("session")
            task = info["task"]
            run = info.get("run")
            sfreq = info["sfreq"]

            if not subject in subjects:
                print(f"Skipping {subject}, {session}, {task}")
                continue

            if not session in sessions:
                print(f"Skipping {subject}, {session}, {task}")
                continue

            if not task in tasks:
                print(f"Skipping {subject}, {session}, {task}")
                continue
            
            # Retrieve all word chunks for this recording
            if dataset == "libribrain":
                chunks = utils.get_libribrain_word_chunks(
                    bids_root, subject, session, task, run, sfreq, batch_size=self.context, overlap=overlap,
                )
            elif "gwilliams2022" in dataset:
                chunks = utils.get_gwilliams_word_chunks(
                    bids_root, subject, session, task, run, sfreq, batch_size=self.context, overlap=overlap,
                )
            elif dataset == "broderick2018":
                chunks = utils.get_broderick_word_chunks(
                    bids_root, subject, session, task, run, sfreq, batch_size=self.context, overlap=overlap,
                )
            else:
                chunks = utils.get_armeni_word_chunks(
                    bids_root, subject, session, task, run, sfreq, batch_size=self.context, overlap=overlap,
                )

            # In Armeni, there is a single inhomogeneous sample that we need to identify and skip the chunk of
            for chunk in chunks:

                skip_chunk = False
                for onset in chunk["onsets"]:
                    if h5_dataset[
                        ..., onset + round(self.tmin * sfreq) : onset + round(self.tmax * sfreq)
                    ].shape[-1] < round((self.tmax - self.tmin) * sfreq):
                        print("Skipping inhomogeneous sample")
                        skip_chunk = True
                        break

                if not skip_chunk:
                    self.samples.append(Sample(h5_dataset, chunk))
            
            dataset_samples = h5_dataset.shape[-1]
            dataset_seconds = dataset_samples / sfreq
            self.seconds += dataset_seconds
            
            n_datasets += 1

    def get_time(self):
        return self.seconds
    
    def n_subjects(self):
        return len(self.subjects)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]
        
        subject_id = sample.h5_dataset.attrs["subject_idx"].item()
        robust_scaler_center = sample.h5_dataset.attrs["robust_scaler_center"]
        robust_scaler_scale = sample.h5_dataset.attrs["robust_scaler_scale"]
        sfreq = sample.h5_dataset.attrs["sfreq"]

        if self.dataset in ["libribrain", "broderick2018"]:
            sensor_xyz = self.sensor_xyz
        else:
            sensor_xyz = sample.h5_dataset.attrs["sensor_xyz"] 

        onsets = sample.chunk["onsets"]
        meg = [
            sample.h5_dataset[
                ...,
                onset + round(self.tmin * sfreq) : onset + round(self.tmax * sfreq)
            ]
            for onset in onsets
        ]
        meg = np.array([
            _scale_meg(
                meg_sample, robust_scaler_center, robust_scaler_scale, threshold=5, sfreq=sfreq
            ) for meg_sample in meg
        ])
        
        words = []
        for word in sample.chunk["words"]:
            if word in self.top_words_map:
                words.append(self.top_words_map[word])
            else:
                words.append(-1)
        words = np.array(words)

        return {
            "meg": meg,
            "words": words,
            "words_raw": sample.chunk["words"],
            "subject_id": subject_id + self.subject_id_shift,
            "sensor_xyz": sensor_xyz,
            "dataset_id": self.dataset_id,
        }

if __name__ == "__main__":
    dataset = MEGDataset(
        bids_root="/data/engs-pnpl/datasets/armeni2022",
        save_root="/data/engs-pnpl/datasets/armeni2022/derivatives/sentences",
        subjects=["001"],
        sessions=["001"],
        tasks=["compr"],
        dataset="armeni2022",
        context=0,
    )

    x = dataset.__getitem__(0)
    breakpoint()