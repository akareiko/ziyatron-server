import mne
import datetime
import numpy as np

from enum import Enum

class ChannelConfig(Enum):
    DOUBLE_BANANA = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FZ-CZ', 'CZ-PZ', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2']

DEFAULT_CONFIG = {
    "channel_conf": ChannelConfig.DOUBLE_BANANA,
    "sampling_rate": 250,
    "notch_filter": 60,
    "filter": (0.5, 45),
    "normalize": True,
    "window": 10,
    "batch_size": 32,
}

def z_normalize(data: np.ndarray) -> np.ndarray:
    data = np.nan_to_num(data, nan=0.0)  # Handle NaN values
    mean = np.mean(data, axis=-1, keepdims=True)
    std = np.std(data, axis=-1, keepdims=True)
    std[std == 0] = 1  # Avoid division by zero for flat signals
    return (data - mean) / std

def preprocess_eeg_data(
        file_path: str, 
        channel_conf: ChannelConfig = DEFAULT_CONFIG['channel_conf'],
        sampling_rate: int = DEFAULT_CONFIG['sampling_rate'],
        notch_filter: int = DEFAULT_CONFIG['notch_filter'],
        filter: tuple = DEFAULT_CONFIG['filter'],
        normalize: bool = DEFAULT_CONFIG['normalize'],
        window: int = DEFAULT_CONFIG['window'],
        batch_size: int = DEFAULT_CONFIG['batch_size'],
    ) -> np.ndarray:

    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    channels_to_drop = []
    channels_to_rename = {}
    channels_to_be_found = set(channel_conf.value)

    for channel in raw.ch_names:
        # Dropping non EEG channels
        if 'EOG' in channel or 'ECG' in channel or 'EMG' in channel:
            channels_to_drop.append(channel)
        
        proper_name = None
        for proper_channel in channel_conf.value:
            if proper_channel in channel:
                proper_name = proper_channel
                break
        if proper_name and proper_name in channels_to_be_found:
            channels_to_rename[channel] = proper_name
            channels_to_be_found.remove(proper_name)
        else:
            channels_to_drop.append(channel)

    raw.drop_channels(channels_to_drop)
    raw.rename_channels(channels_to_rename)

    raw.pick(channel_conf.value)

    # preprocessing
    raw.resample(sampling_rate, npad='auto')

    if notch_filter:
        raw.notch_filter(freqs=notch_filter, verbose=False)

    if filter:
        raw.filter(*filter, fir_design='firwin', verbose=False)

    raw.set_meas_date(datetime.datetime.now().replace(tzinfo=datetime.timezone.utc))

    # extracting windows and batching
    raw_data = raw.get_data()
    sfreq = raw.info['sfreq']

    samples_per_batch = int(sfreq * window)
    num_batches = raw_data.shape[1] // samples_per_batch

    raw_data = raw_data[:, :num_batches * samples_per_batch]

    batches = raw_data.T.reshape(num_batches, samples_per_batch, -1).transpose(0, 2, 1)

    if normalize:
        batches = z_normalize(batches)

    batches = batches.astype(np.float32)

    batches = np.array_split(batches, batches.shape[0] // batch_size, axis=0)

    
    print(f"[INFO] Number of batches: {len(batches)}")
    return batches