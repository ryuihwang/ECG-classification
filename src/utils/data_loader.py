def add_noise(data, noise_level=0.05):
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise

def time_shift(data, shift):
    return np.roll(data, shift)

def time_stretch(data, rate):
    input_length = len(data)
    data = np.interp(np.arange(0, input_length, rate), np.arange(0, input_length), data)
    if len(data) < input_length:
        data = np.pad(data, (0, input_length - len(data)), 'constant')
    return data[:input_length]

def scaling(data, scale):
    return data * scale

def invert_signal(data):
    return -data

def augment_data(data):
    augmented_data = []
    augmented_data.append(add_noise(data))
    augmented_data.append(time_shift(data, shift=100))
    augmented_data.append(time_stretch(data, rate=1.2))
    augmented_data.append(scaling(data, scale=1.5))
    augmented_data.append(invert_signal(data))
    return augmented_data

def augment_dataset(X, y):
    X_augmented = []
    y_augmented = []
    
    for i in range(len(X)):
        augmented_samples = augment_data(X[i])
        X_augmented.extend(augmented_samples)
        y_augmented.extend([y[i]] * len(augmented_samples))
    
    return np.array(X_augmented), np.array(y_augmented)

def load_data_and_labels(preprocessed_dir, labels_df):
    labels_df['study_id'] = labels_df['study_id'].astype(str).str.split('.').str[0]
    print(labels_df.head())
    
    X = []
    y = []
    study_ids = []
    for idx, row in labels_df.iterrows():
        study_id = row['study_id']
        label = row['LABEL']
        file_path = os.path.join(preprocessed_dir, f"{study_id}.npy")
        if os.path.exists(file_path):
            data = np.load(file_path)
            X.append(data)
            y.append(label)
            study_ids.append(study_id)
    X = np.array(X)
    y = np.array(y)
    study_ids = np.array(study_ids)
    return X, y, study_ids

def undersample_data(X, y, study_ids, target_class=1):
    class_counts = np.bincount(y)
    min_class_count = class_counts[target_class]

    indices_to_keep = np.where(y == target_class)[0]
    indices_to_remove = np.where(y != target_class)[0]

    np.random.shuffle(indices_to_remove)
    indices_to_remove = indices_to_remove[:min_class_count]

    undersampled_indices = np.concatenate([indices_to_keep, indices_to_remove])
    np.random.shuffle(undersampled_indices)

    return X[undersampled_indices], y[undersampled_indices], study_ids[undersampled_indices]

base_dir = 'new_directory/data'

def remove_nan_samples(data, labels, study_ids):
    nan_indices = np.isnan(data).any(axis=1)
    return data[~nan_indices], labels[~nan_indices], study_ids[~nan_indices]