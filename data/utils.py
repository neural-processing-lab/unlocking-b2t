import ast
import glob
import mne
import numpy as np
import pandas as pd
import scipy.io

def preprocess_meg(raw, resample_freq, l_freq, h_freq, notch_freq):
    # Band-pass filter the data to remove low and high frequency noise
    raw.load_data()
    raw.filter(l_freq=l_freq, h_freq=h_freq, picks="all", n_jobs=-1, verbose=False)

    if h_freq > notch_freq:
        # Filter electric grid frequency and any harmonics if present in the signal
        raw.notch_filter(
            freqs=list(range(notch_freq, h_freq + 1, notch_freq)), verbose=False
        )

    # Decimate the signal by resampling (after cleaning up the signal already)
    raw.resample(sfreq=resample_freq, verbose=False)

    print("Preprocessed MEG. New sample rate", raw.info["sfreq"])

    return raw


def pick_triux_channels(raw):
    """Pick only the MEG channels (starting with 'MEG') from Triux scanners."""

    channel_names = raw.info["ch_names"]
    filtered_names = filtered_names = [
        name for name in channel_names if name.startswith("MEG")
    ]
    raw = raw.pick(filtered_names)

    return raw


def pick_sherlock_channels(raw):
    """Pick only the MEG channels (starting with 'MEG') from the Sherlock dataset."""

    channel_names = raw.info["ch_names"]
    filtered_names = filtered_names = [
        name for name in channel_names if name.startswith("MEG")
    ]
    raw = raw.pick(filtered_names)

    return raw


def pick_armeni_channels(raw):
    """Pick only the MEG channels (starting with 'M') from the Armeni dataset."""

    channel_names = raw.info["ch_names"]
    filtered_names = filtered_names = [
        name for name in channel_names if name.startswith("M")
    ]
    raw = raw.pick(filtered_names)

    # The dataset doesn't set a valid BIDS channel type (meggrad) so we set them as mag.
    raw.set_channel_types(dict(zip(filtered_names, ["mag"] * len(filtered_names))))

    return raw

def pick_gwilliams_channels(raw):
    """Pick only the MEG channels."""

    # Get only relevant MEG channels (ignore reference channels)
    meg_picks = mne.pick_types(raw.info, meg=True, ref_meg=False)
    raw = raw.pick(meg_picks, verbose=False)

    return raw

def get_all_armeni_words(bids_root):
    """Get all words from an events.tsv file."""
    event_paths = []
    for subject in ["001"]:
        for session in ["001", "002", "003", "004", "005", "006", "007", "008", "009", "010"]:
            path = f"{bids_root}/sub-{subject}/ses-{session}/meg"
            event_paths.append(path)
    event_files = [glob.glob(f"{path}/*compr_events.tsv")[0] for path in event_paths]

    words = []
    for events_path in event_files:
        events = pd.read_csv(events_path, sep="\t")
        word_events = events[["word_onset" in c for c in list(events["type"])]]
        words += [v for v in word_events["value"].values if v != "sp"]
    
    return words

def get_armeni_word_chunks(
    bids_root, subject, session, task, run, sample_freq, batch_size=64, overlap=0,
):
    """Get continuous chunks of words with their onsets from the Armeni dataset.
    
    Args:
        bids_root (str): Path to BIDS root directory
        subject (str): Subject ID
        session (str): Session ID
        task (str): Task name
        run (str): Run number
        sample_freq (float): Sampling frequency
        batch_size (int): Size of each continuous chunk of words
        overlap (int): Number of words to overlap between chunks
    
    Returns:
        list of dicts: Each dict contains onsets and words for a chunk
    """
    events_path = f"{bids_root}/sub-{subject}/ses-{session}/meg/sub-{subject}_ses-{session}_task-{task}_events.tsv"
    events = pd.read_csv(events_path, sep="\t")
    
    # Filter for word events and remove speech markers
    word_events = events[["word_onset" in c for c in list(events["type"])]]
    word_events = word_events[word_events['value'] != 'sp']
    
    chunks = []
    stride = batch_size - overlap
    
    # Generate chunks with overlap
    for start_idx in range(0, len(word_events) - batch_size + 1, stride):
        chunk = word_events.iloc[start_idx:start_idx + batch_size]
        
        # Convert onsets to samples
        onsets = [round(float(t) * sample_freq) for t in chunk["onset"].values]

        words = list(chunk["value"].values)
        
        chunks.append({
            "onsets": onsets,
            "words": words,
            "start_idx": start_idx,
            "end_idx": start_idx + batch_size
        })
    
    return chunks


def get_all_libribrain_words(bids_root):
    """Get all words from an events.tsv file."""
    tasks = [f"Sherlock{i}" for i in range(1, 7)]
    event_files = []
    for task in tasks:
        event_files += list(glob.glob(f"{bids_root}/{task}/derivatives/events/*events.tsv"))

    words = []
    for events_path in event_files:
        events = pd.read_csv(events_path, sep="\t")
        word_events = events[["word" in c for c in list(events["kind"])]]
        words += [str(v).strip().upper() for v in word_events["segment"].values]
    
    return words


def get_libribrain_word_chunks(
        bids_root, subject, session, task, run, sample_freq, batch_size=64, overlap=0
):
    """Get continuous chunks of words with their onsets from the Armeni dataset.
    
    Args:
        bids_root (str): Path to BIDS root directory
        subject (str): Subject ID
        session (str): Session ID
        task (str): Task name
        run (str): Run number
        sample_freq (float): Sampling frequency
        batch_size (int): Size of each continuous chunk of words
        overlap (int): Number of words to overlap between chunks
    
    Returns:
        list of dicts: Each dict contains onsets and words for a chunk
    """
    events_path = f"{bids_root}/{task}/derivatives/events/sub-{subject}_ses-{session}_task-{task}_run-{run}_events.tsv"
    events = pd.read_csv(events_path, sep="\t")
    
    # Filter for word events and remove speech markers
    word_events = events[["word" in c for c in list(events["kind"])]]
    
    chunks = []
    stride = batch_size - overlap
    
    # Generate chunks with overlap
    for start_idx in range(0, len(word_events) - batch_size + 1, stride):
        chunk = word_events.iloc[start_idx:start_idx + batch_size]
        
        # Convert onsets to samples
        onsets = [round(float(t) * sample_freq) for t in chunk["timemeg"].values]

        words = list(chunk["segment"].values)
        words = [str(w).strip().upper() for w in words]
        
        chunks.append({
            "onsets": onsets,
            "words": words,
            "start_idx": start_idx,
            "end_idx": start_idx + batch_size
        })
    
    return chunks

def get_all_gwilliams_words(bids_root):
    """Get all words from an events.tsv file."""
    event_paths = []
    for subject in ["01"]:
        for session in ["0"]:
            path = f"{bids_root}/sub-{subject}/ses-{session}/meg"
            event_paths.append(path)
    event_files = [glob.glob(f"{path}/*_events.tsv")[0] for path in event_paths]

    words = []
    for events_path in event_files:
        events = pd.read_csv(events_path, sep="\t")
        word_events = events[
            ["'kind': 'word'" in trial_type for trial_type in list(events["trial_type"])]
        ]
        words += [tt.split("'word': '")[-1].split("'")[0] for tt in list(word_events["trial_type"].values)]

    words = [str(w).strip().upper() for w in words]
    
    return words

def get_gwilliams_word_chunks(
        bids_root, subject, session, task, run, sample_freq, batch_size=64, overlap=0,
):
    """Get continuous chunks of words with their onsets from the Armeni dataset.
    
    Args:
        bids_root (str): Path to BIDS root directory
        subject (str): Subject ID
        session (str): Session ID
        task (str): Task name
        run (str): Run number
        sample_freq (float): Sampling frequency
        batch_size (int): Size of each continuous chunk of words
        overlap (int): Number of words to overlap between chunks
    
    Returns:
        list of dicts: Each dict contains onsets and words for a chunk
    """
    events_path = f"{bids_root}/sub-{subject}/ses-{session}/meg/sub-{subject}_ses-{session}_task-{task}_events.tsv"
    events = pd.read_csv(events_path, sep="\t")
    
    # Filter for word events and remove speech markers
    word_events = events[
        ["'kind': 'word'" in trial_type for trial_type in list(events["trial_type"])]
    ]
    
    chunks = []
    stride = batch_size - overlap
    
    # Generate chunks with overlap
    for start_idx in range(0, len(word_events) - batch_size + 1, stride):
        chunk = word_events.iloc[start_idx:start_idx + batch_size]
        
        # Convert onsets to samples
        onsets = [round(float(t) * sample_freq) for t in chunk["onset"].values]

        # tt format: ... 'word': 'the' ...
        words = [tt.split("'word': '")[-1].split("'")[0] for tt in list(chunk["trial_type"].values)]
        words = [str(w).strip().upper() for w in words]
        
        chunks.append({
            "onsets": onsets,
            "words": words,
            "start_idx": start_idx,
            "end_idx": start_idx + batch_size
        })
    
    return chunks

def get_broderick_word_chunks(
        bids_root, subject, session, task, run, sample_freq=50, batch_size=64, overlap=0,
    ):
    events_path = f"{bids_root}/Stimuli/Text/Run{session}.mat"
    events = scipy.io.loadmat(events_path)
    words = [str(w[0][0]).strip().upper() for w in events["wordVec"]]
    onsets = [float(t[0]) for t in events["onset_time"]]

    # FIXME: Note that broderick events are sampled at 50Hz

    chunks = []
    stride = batch_size - overlap
    
    # Generate chunks with overlap
    for start_idx in range(0, len(words) - batch_size + 1, stride):
        word_chunk = words[start_idx:start_idx + batch_size]
        onset_chunk = onsets[start_idx:start_idx + batch_size]
        
        # Convert onsets to samples
        onset_chunk = [round(t * sample_freq) for t in onset_chunk]
        
        chunks.append({
            "onsets": onset_chunk,
            "words": word_chunk,
            "start_idx": start_idx,
            "end_idx": start_idx + batch_size
        })

    return chunks

def get_all_broderick_words(bids_root):
    events_path = glob.glob(f"{bids_root}/Stimuli/Text/Run*.mat")

    words = []
    for event_path in events_path:
        events = scipy.io.loadmat(event_path)
        words.extend([str(w[0][0]).strip().upper() for w in events["wordVec"]])
    
    return words

if __name__ == "__main__":
    # Example usage
    bids_root = "/data/engs-pnpl/datasets/broderick2018"
    subject = "1"
    session = "1"
    task = "natural"
    run = None
    sample_freq = 50
    batch_size = 64
    overlap = 0

    chunks = get_broderick_word_chunks(bids_root, subject, session, task, run, sample_freq, batch_size, overlap)
    words = get_all_broderick_words(bids_root)
    # print(chunks)