ECG Arrhythmia Detection using a CNN-LSTM Model

This project uses a deep learning model to classify heartbeats from 2-channel ECG signals into the 5 standard AAMI arrhythmia classes. The model is trained on the MIT-BIH Arrhythmia Database and achieves 99.12% accuracy on the held-out test set.
It is capable of loading a raw, unseen patient ECG file (in wfdb format), processing it, and generating a beat-by-beat analysis and a final diagnostic summary.
How It Works: The 6-Step Pipeline
The project is built as a complete pipeline, from raw data to final diagnosis.


1. Data Loading

Source: The MIT-BIH Arrhythmia Database.
Format: Uses the wfdb library to read the raw .dat (signal) and .atr (annotation) files from all 48 records.
Channels Used: The model is trained on the 2-channel signals provided by the database, which are typically Lead 1 (MLII) and Lead 2 (V5).


2. Preprocessing
Beat Segmentation: The signal is segmented into 180-sample (0.5-second) windows centered around each R-peak.
Filtering: A 4th-order Butterworth bandpass filter (0.5 Hz - 45 Hz) is applied to each segment to remove baseline wander and high-frequency noise.
Normalization: A Global Z-Score Normalization is applied. The mean (GLOBAL_MEAN) and standard deviation (GLOBAL_STD) are calculated only from the training set (to prevent data leakage) and then applied to all segments (train, test, and new data).


3. Class Handling (The Imbalance Problem)

Consolidation: The 23+ raw annotation symbols from the .atr files are mapped to the 5 standard AAMI classes:

0: N (Normal)
1: S (Supraventricular)
2: V (Ventricular)
3: F (Fusion)
4: Q (Unclassifiable)

Balancing (SMOTE): The training set is severely imbalanced (91.35% Class N). To fix this, SMOTE (Synthetic Minority Over-sampling Technique) is used to generate new, synthetic samples of the minority classes (S, V, F, Q) until the training set is perfectly balanced.


4. Model Architecture (CNN-LSTM)

The model is a hybrid deep learning architecture designed for time-series classification:
Conv1D Layers: Three blocks of 1D convolutions act as feature extractors, learning to identify the morphological shapes of QRS complexes, P-waves, and T-waves.
LSTM Layers: Two stacked LSTM layers analyze the sequence of features extracted by the CNN, allowing the model to understand temporal patterns within the 0.5-second beat.
Regularization: The model uses BatchNormalization, Dropout, L2 Regularization, and a low Adam learning rate (0.0001) to prevent overfitting on the synthetic training data.


5. Training & Evaluation

The model is trained on the balanced SMOTE data.
It is validated against the original, imbalanced X_test data. This is crucial, as it measures the model's true performance on real-world data.
EarlyStopping (monitoring val_loss) is used to stop training at the peak of model performance and automatically restore the best weights.


6. Real-World Diagnosis (Heuristic Engine)

This is the final, "smart" layer that interprets the model's beat-by-beat predictions. It is not a machine learning model, but a rule-based engine that analyzes the sequence of predicted labels to generate a "Yes/No" diagnosis.
"Heart Disease Detection: YES" is ONLY triggered if one of these clinically significant events is found:
Ventricular Tachycardia: A run of 3 or more 'V' beats in a row.
Supraventricular Tachycardia: A run of 3 or more 'S' beats in a row.
Significant Burden: The total percentage of 'V' beats OR 'S' beats is greater than 5% of all heartbeats.
"Heart Disease Detection: NO" is triggered in all other cases, including:
Normal: 100% of beats are 'N'.
Benign/Occasional: A few isolated arrhythmia beats are found (like in your screenshot, e.g., 1 'V' and 1 'S' beat). This is considered clinically benign and does not trigger a "YES" diagnosis.


*** CRITICAL: Requirements for Real-World Prediction ***

This model was trained under very specific conditions. To use the saved final_ecg_model_90plus.keras file for a new prediction, your input data MUST meet these requirements:
Data Format: The input signal must be a 2-channel recording.
Channel Order (Most Important): The model is highly sensitive to channel order. The input channels MUST be in this specific order:
Channel 0 (index 0): MLII lead
Channel 1 (index 1): V5 lead

If your input file has V5 first, you must swap the channels in the NumPy array before preprocessing.

Preprocessing: The new signal MUST be processed with the exact same bandpass_filter and global_zscore_normalize functions.

Normalization Stats: The normalization function MUST use the GLOBAL_MEAN and GLOBAL_STD values calculated from the original training set (found in dataset_access_FIXED.ipynb, Cell 7).

GLOBAL_MEAN = [ 0.02254995 -0.00905314]

GLOBAL_STD = [ 0.41518577  0.33936556]

Input Shape: The final input shape for a single beat prediction must be (1, 180, 2). The prediction script handles this automatically.
