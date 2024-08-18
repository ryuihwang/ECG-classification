measurements = pd.read_csv('new_directory/data/machine_measurements.csv')

npy_files = os.listdir('new_directory/data/all')
npy_study_ids = [int(filename.split('.')[0]) for filename in npy_files]

filtered_measurements = measurements[measurements['study_id'].isin(npy_study_ids)]

labels = []
for _, row in filtered_measurements.iterrows():
    report = str(row['report_0']).lower()
    if 'atrial fibrillation' in report or 'atrial flutter' in report:
        label = 1
    else:
        label = 0
    labels.append({
        'subject_id': row['subject_id'],
        'study_id': row['study_id'],
        'LABEL': label
    })

labels_df = pd.DataFrame(labels)
labels_df.to_csv('/Users/hri/Downloads/SNUH_VITALLAB_RECRUITING_PROJECT/data/labels/labels.csv', index=False)

label_counts = labels_df['LABEL'].value_counts()
print(label_counts)

root_dir = '/Users/hri/Downloads/SNUH_VITALLAB_RECRUITING_PROJECT/files'
output_dir = '/Users/hri/Downloads/SNUH_VITALLAB_RECRUITING_PROJECT/data/all'
convert_directory_to_npy(root_dir, output_dir)

npy_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.npy')]
all_signals = [np.load(file) for file in npy_files]

input_dir = '/Users/hri/Downloads/SNUH_VITALLAB_RECRUITING_PROJECT/data/all'
output_dir = '/Users/hri/Downloads/SNUH_VITALLAB_RECRUITING_PROJECT/data/preprocessed'
os.makedirs(output_dir, exist_ok=True)

# 10초 신호 250Hz로(가장 적합하다고 판단..)
resample = Resample(signal_time_length=10, sample_rate_to=250) 
standardize = Standardize()
# 0.5-50Hz가 적절할듯..
filter_signal = Filter(lowcut=0.5, highcut=50.0, sample_rate=250)
npy_files = os.listdir(input_dir)

for npy_file in npy_files:

    file_path = os.path.join(input_dir, npy_file)
    signal = np.load(file_path)
    
    signal_resampled = resample(signal, original_sample_rate=500)
    signal_filtered = filter_signal(signal_resampled)
    signal_standardized = standardize(signal_filtered)
    
    output_path = os.path.join(output_dir, npy_file)
    np.save(output_path, signal_standardized)

input_dir = '/Users/hri/Downloads/SNUH_VITALLAB_RECRUITING_PROJECT/data/preprocessed'
train_dir = '/Users/hri/Downloads/SNUH_VITALLAB_RECRUITING_PROJECT/data/train'
validation_dir = '/Users/hri/Downloads/SNUH_VITALLAB_RECRUITING_PROJECT/data/validation'
evaluation_dir = '/Users/hri/Downloads/SNUH_VITALLAB_RECRUITING_PROJECT/data/evaluation'

npy_files = os.listdir(input_dir)

for npy_file in npy_files:
    
    study_id = int(npy_file.split('.')[0])
   
    subject_id = measurements.loc[measurements['study_id'] == study_id, 'subject_id'].values[0]
    
    subject_prefix = int(str(subject_id)[:4])
    
    if 1000 <= subject_prefix <= 1021:
        shutil.move(os.path.join(input_dir, npy_file), os.path.join(train_dir, npy_file))
    elif 1022 <= subject_prefix <= 1026:
        shutil.move(os.path.join(input_dir, npy_file), os.path.join(validation_dir, npy_file))
    elif 1027 <= subject_prefix <= 1031:
        shutil.move(os.path.join(input_dir, npy_file), os.path.join(evaluation_dir, npy_file))