import mne
import yaml
import datetime
import numpy as np

def z_normalize(data: np.ndarray) -> np.ndarray:
    data = np.nan_to_num(data, nan=0.0)  # Handle NaN values
    mean = np.mean(data, axis=-1, keepdims=True)
    std = np.std(data, axis=-1, keepdims=True)
    std[std == 0] = 1  # Avoid division by zero for flat signals
    return (data - mean) / std

def preprocess_eeg_data(file_path: str, config_file: str = './eeg_inference/default_configs.yaml') -> np.ndarray:

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
        channel_conf = config['channel_conf']
        sampling_rate = config['sampling_rate']
        notch_filter = config['notch_filter']
        filter = config['filter']
        normalize = config['normalize']
        window = config['window']
        batch_size = config['batch_size']

    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

    channels_to_drop = []
    channels_to_rename = {}
    channels_to_be_found = set(channel_conf)

    for channel in raw.ch_names:
        # Dropping non EEG channels
        if 'EOG' in channel or 'ECG' in channel or 'EMG' in channel:
            channels_to_drop.append(channel)
        
        proper_name = None
        for proper_channel in channel_conf:
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

    raw.pick(channel_conf)

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
    print(batches[0].shape)
    return batches, config