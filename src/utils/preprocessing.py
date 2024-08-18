def get_ecg_idx(segment_path: str, lead: list):
    record = wfdb.rdrecord(segment_path)
    channel_names = record.sig_name
    lead_idx = []

    for l in lead:
        ecg_idx = [e for e in range(len(channel_names)) if channel_names[e] == l][0]
        lead_idx.append(ecg_idx)

    return lead_idx

def get_channel_record(segment_path: str, channel_idx: int):
    if ".hea" in segment_path:
        segment_path = segment_path.replace(".hea", "")

    record = wfdb.rdrecord(segment_path, channels=[channel_idx])
    digital_sig = np.array(record.p_signal).squeeze()

    return digital_sig

def convert_to_npy(segment_path: str, lead: str = 'II') -> np.ndarray:
    lead_idx = get_ecg_idx(segment_path, [lead])[0]
    signal = get_channel_record(segment_path, lead_idx)
    return signal

def convert_directory_to_npy(root_dir: str, output_dir: str, lead: str = 'II'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.hea'):
                segment_path = os.path.join(root, file.replace('.hea', ''))
                signal = convert_to_npy(segment_path, lead)
                output_path = os.path.join(output_dir, f"{os.path.basename(segment_path)}_{lead}.npy")
                np.save(output_path, signal)
                print(f"Converted {segment_path} to {output_path}")

class Resample:
    def __init__(self, signal_time_length: int, sample_rate_to: int) -> None:
        self.signal_time_length = signal_time_length
        self.sample_rate_to = sample_rate_to

    def __call__(self, x: np.ndarray, original_sample_rate: int) -> np.ndarray:
        num_samples = self.signal_time_length * self.sample_rate_to
        resampled_signal = scipy_resample(x, num_samples)
        return resampled_signal

class Standardize:
    def __init__(self, dim: int = -1, eps: float = 1e-6) -> None:
        self.dim = dim
        self.eps = eps

    def __call__(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=self.dim, keepdims=True)
        std = np.std(x, axis=self.dim, keepdims=True) + self.eps
        standardized_signal = (x - mean) / std
        return standardized_signal

class Filter:
    def __init__(self, lowcut: float, highcut: float, sample_rate: int, order: int = 5) -> None:
        nyquist = 0.5 * sample_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        self.b, self.a = butter(order, [low, high], btype='band')

    def __call__(self, x: np.ndarray) -> np.ndarray:
        filtered_signal = filtfilt(self.b, self.a, x)
        return filtered_signal