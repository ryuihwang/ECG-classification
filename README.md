# ECG-classification
binary classification with ECG Lead-II with positive labels of atrial fibrillation and the other is atrial flutter, negative labels of others.

---

**MODEL REPORT**

1. **Preprocessing**

   - **Data Transformation:**  
   The original `.hea` files were read, and the signal corresponding to the II lead was extracted using the `wfdb` library. The extracted signals were converted to numpy arrays and saved.

   - **Resampling:**  
   The signal's length and sample rate were adjusted to maintain consistency in the ECG signals. This was done using the `scipy` library, resampling the signals to a 250Hz sample rate over a 10-second duration, resulting in a total of 2500 samples.

   - **Standardization:**  
   For each signal, the mean and standard deviation were calculated. Standardization was performed by subtracting the mean and dividing by the standard deviation. A small epsilon value was added to avoid small standard deviations.

   - **Filtering:**  
   To remove noise from the signal and allow only the necessary frequencies for analysis, filters were designed and applied using the `butter` and `filtfilt` functions from the `scipy` library. A frequency range of 0.5Hz to 50Hz, which contains the most relevant information, was applied.

   - **TensorDataset Undersampling:**  
   The preprocessed signals were converted to a `TensorDataset` and loaded into batches using the `DataLoader`. The training data (p1000-p1021) consisted of 16,734 sets of samples, but there was a large imbalance between class 1 (Atrial Fibrillation and Atrial Flutter) and class 0 (others). Therefore, undersampling was performed to balance the classes, resulting in 1,806 samples for both classes, with a total of 3,612 sets used for training.

   - **Data Augmentation:**  
   Various transformations were applied to increase the data and improve the model's generalization capability. This included time stretching (altering the length of the signal), adding random noise, and shifting the signal in the time axis to introduce variation.

2. **Modeling & Training**

   A basic convolutional neural network (1D CNN) architecture was applied. The model consisted of three convolutional layers, batch normalization, ReLU activation functions, dropout layers, and fully connected layers. It was designed to effectively extract features from the time-series signal data.

   - **Hyperparameters:**
     - Learning Rate: 0.0001
     - Batch Size: 16
     - Number of Epochs: 15
     - Weight Initialization: Xavier Initialization
     - Loss Function: Binary Cross Entropy Loss (`BCEWithLogitsLoss`)
     - Optimizer: Adam Optimizer with L2 Regularization (Weight Decay)

   - **Training and Validation:**
     - The model was trained using cross-validation to prevent overfitting. Early stopping was also applied. Training and validation losses were visualized, and the model's performance was evaluated using a confusion matrix.
     - The optimal classification threshold was determined using ROC curves and Youden's J statistic. Given the significant data imbalance, the cutoff threshold was adjusted to 0.1 instead of the default 0.5 after analyzing the normal distribution of the sigmoid function's output probabilities.

3. **Evaluation**

   - **Training/Validation Loss:**
     - The training loss decreased rapidly during the early stages of training and became relatively stable as the epochs progressed. Overall, the validation loss was lower than the training loss in each fold, indicating no overfitting.
     - However, in the last 10th fold, a spike in validation loss was observed midway through the epochs. This suggests that the model may have overfitted to specific features in that fold. Such rises in validation loss during training could be attributed to the data's complexity and imbalance.
     - To address this, further evaluation using K-Fold Cross Validation is recommended to assess the model's stability and calculate the average performance across different folds.

   - **Confusion Matrix:**
     - True Positive: 2,261
     - True Negative: 16,094
     - False Positive: 6,106
     - False Negative: 538
     - The model showed no significant issues with over-detection or under-detection in the negative class. However, it did display considerable over-detection in the positive class.
     - **Strengths:** The model achieved a high ratio of True Positives and True Negatives, indicating accurate predictions.
     - **Areas for Improvement:** The model exhibited a relatively high ratio of False Positives and False Negatives, particularly in the positive class. This indicates that further steps are needed to address over-detection in the positive class.

   - While the overall accuracy of the model was 80%, the precision for class 1 (positive class) was notably lower, resulting in a much lower F1 score compared to class 0.
   - The model demonstrated high precision and relatively high recall for class 0 (Negative Class), meaning it correctly predicted most negative samples but missed some positive samples.
   - Precision for class 1 (Positive Class) was low, indicating that many samples predicted as positive were actually negative. While recall was relatively high, the low precision led to a low overall F1 score for class 1.
   - To improve precision for class 1, techniques such as SMOTE (Synthetic Minority Over-sampling Technique) or cost-sensitive learning should be considered.

   - **Reference:**  
   The results without any preprocessing (e.g., resampling, undersampling, standardization, etc.) show that the model recorded low scores for AUROC, F1 score, precision, and recall due to noise and imbalance in the data. These results emphasize the importance of preprocessing in improving model performance.
