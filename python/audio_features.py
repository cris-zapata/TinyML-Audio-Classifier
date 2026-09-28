import numpy as np
from scipy.io import wavfile
import os
def extract_features(audio_window, sample_rate):
    hann = np.hanning(len(audio_window))
    windowed_audio = audio_window * hann
    fft_values = np.fft.rfft(windowed_audio)
    frequencies = np.fft.rfftfreq(
        len(audio_window),
        d=1/sample_rate
    )
    magnitude = np.abs(fft_values)
    num_bands = 16
    max_frequency = 8000
    band_edges = np.linspace(0, max_frequency, num_bands + 1)
    band_features = np.zeros(num_bands)
    for i in range(num_bands):
       in_band = (frequencies >= band_edges[i]) & (frequencies < band_edges[i + 1])
       band_features[i] = np.sum(magnitude[in_band] ** 2)
    max_energy = np.max(band_features)
    db_features = 10 * np.log10(
        (band_features + 1e-12) / max_energy
    )
    db_features = np.clip(
        db_features,
        -60,
        0
    )
    normalized_features = (db_features + 60) / 60
    int8_features = np.round(
        normalized_features * 127
    ).astype(np.int8)
    return int8_features
whistle_windows = {
    "whistle_01.wav": [19],
    "whistle_02.wav": [0, 1, 2, 3, 4, 5],
    "whistle_03.wav": [2, 3, 4, 5, 6, 7],
    "whistle_04.wav": [],
    "whistle_05.wav": [],
    "whistle_06.wav": [4, 5, 7, 8, 12, 13, 14, 16, 17, 18],
    "whistle_07.wav": [0, 1, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19],
    "whistle_08.wav": [11, 12, 13, 14, 15, 16, 17, 18],
    "whistle_09.wav": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19],
    "whistle_10.wav": [],
    "whistle_11.wav": [],
}
clap_windows = {
    "clap_01.wav": [0, 1, 2, 3, 4, 5, 6, 7],
    "clap_02.wav": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    "clap_03.wav": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    "clap_04.wav": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    "clap_05.wav": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "clap_06.wav": [4, 5, 6, 7, 8, 9, 10, 11],
}
snap_windows = {
    "snap_01.wav": list(range(24)),
    "snap_02.wav": list(range(20)),  
    "snap_03.wav": list(range(2)),   
    "snap_04.wav": [],  
    "snap_05.wav": list(range(16)),  
    "snap_06.wav": [],  
    "snap_08.wav": list(range(20)),
    "snap_09.wav": list(range(100)),
    "snap_10.wav": list(range(100)),
    "snap_11.wav": list(range(100)),
    "snap_12.wav": list(range(100)),
    "snap_13.wav": list(range(100)),
}
speech_windows = {
    "speech_01.wav": list(range(20)),
    "speech_02.wav": list(range(20)),
    "speech_03.wav": list(range(20)),
    "speech_04.wav": list(range(20)),
    "speech_05.wav": list(range(30)),
    "speech_06.wav": list(range(20)),
    "speech_07.wav": list(range(20)),
    "speech_08.wav": list(range(20)),
}
dataset_config = {
    "clap": {
        "label": 0,
        "windows": clap_windows
    },
    "whistle": {
        "label": 1,
        "windows": whistle_windows
    },
    "speech": {
        "label": 2,
        "windows": speech_windows
    },
    "snap": {
        "label": 3,
        "windows": snap_windows
    }
}
X = []
y = []
recording_ids = []
for class_name, config in dataset_config.items():
    audio_folder = os.path.join(
        "Audio Features/audio",
        class_name
    )
    audio_files = sorted([
    file for file in os.listdir(audio_folder)
    if(
        file.endswith(".wav")
        and file.startswith(f"{class_name}_")
        and file != "whistle_test.wav"

    )
])
    print("Files found:")
    for file in audio_files:
        file_path = os.path.join(audio_folder, file)
        sample_rate, audio = wavfile.read(file_path)
        print(f"\n==== {file} ====")
        print("Sample rate:", sample_rate)
        print("Number of samples:", len(audio))
        print("Shape:", audio.shape)
        print("Data type:", audio.dtype)
        print("First 10 samples:", audio[:10])
        window_duration = 0.5
        window_size = int(sample_rate * window_duration)
        num_windows = len(audio) // window_size
        print("Number of windows", num_windows)
        rms_threshold = 500
        for i in range(num_windows):
            start = i * window_size
            end = start + window_size
            audio_window = audio[start:end]
            audio_float = audio_window.astype(np.float64)
            rms = np.sqrt(np.mean(audio_float ** 2))
            if rms < rms_threshold:
                print(f"Window {i}: rejected | RMS = {rms:.1f}")
                continue
            if i not in config["windows"][file]:
                continue
            features = extract_features(audio_window, sample_rate)
            X.append(features)
            y.append(config["label"])
            recording_ids.append(file)
            print(
                f"Window {i}: accepted |" f"RMS = {rms:.1f} |" f"features = {features}"
            )
X = np.array(X)
y = np.array(y)
recording_ids = np.array(recording_ids)
print("\n===== DATASET COMPLETE =====")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Recording IDs shape:", recording_ids.shape)
print("Clap samples:", np.sum(y == 0))
print("Whistle samples:", np.sum(y == 1))
print("Speech samples:", np.sum(y == 2))
print("Snap samples:", np.sum(y == 3))
print("\n==== SAMPLES PER RECORDING ====")
for recording in np.unique(recording_ids):
    count = np.sum(recording_ids == recording)
    print(recording, ":", count)
    test_recordings = [
        "clap_04.wav",
        "whistle_09.wav",
        "speech_08.wav",
        "snap_08.wav"
    ]
    test_mask = np.isin(recording_ids, test_recordings)
    X_test = X[test_mask]
    y_test = y[test_mask]
    X_train = X[~test_mask]
    y_train = y[~test_mask]
    recording_ids_test = recording_ids[test_mask]
    recording_ids_train = recording_ids[~test_mask]
print("\n===== TRAIN / TEST SPLIT =====")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)
print("\nTest class counts:")
print("Clap:", np.sum(y_test == 0))
print("Whistle:", np.sum(y_test == 1))
print("Speech:", np.sum(y_test == 2))
print("Snap:", np.sum(y_test == 3))
np.savez(
    "Audio Features/audio_dataset.npz",
    X_train=X_train,
    y_train=y_train,
    recording_ids_train=recording_ids_train,
    X_test=X_test,
    y_test=y_test,
    recording_ids_test=recording_ids_test
    )