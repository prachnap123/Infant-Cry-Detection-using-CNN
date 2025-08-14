from google.colab import drive
drive.mount('/content/drive')
# 2.class distribution plot
import os # Import the os module
import seaborn as sns # Import seaborn
import matplotlib.pyplot as plt # Import matplotlib.pyplot

# Define the data directory
data_dir = '/content/drive/MyDrive/infant_cry_dataset'

# Count the number of files in each subfolder (each class)
class_counts = {}
for class_name in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_name)
    if os.path.isdir(class_path):
        class_counts[class_name] = len(os.listdir(class_path))

# Sort the dictionary (optional for prettier plot)
class_counts = dict(sorted(class_counts.items(), key=lambda x: x[0]))

# Plotting
sns.set(style='whitegrid')
plt.figure(figsize=(10, 6))
sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()), palette='deep')
plt.title('Infant Cry Class Distribution', fontsize=16)
plt.xlabel('Cry Type', fontsize=12)
plt.ylabel('Number of Samples', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# 3.Install the necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
!pip install librosa audiomentations
import numpy as np
import librosa
import random
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Gain
!pip install --upgrade audiomentations
!pip install tqdm
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
#Data Augumentation Library
!pip install librosa audiomentations
#Define Augmentation Functions
import numpy as np
import librosa
import random
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift#, FrequencyMask, TimeMask

# Set a consistent sample rate
SAMPLE_RATE = 22050

# Augmentation pipeline using audiomentations
augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    #FrequencyMask(),
    #TimeMask()
])

def load_audio(file_path, sr=SAMPLE_RATE):
    try:
        audio, _ = librosa.load(file_path, sr=sr)
        return audio
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def extract_mfcc_with_aug(file_path, augment_audio=True):
    y = load_audio(file_path)
    if y is None:
        return None

    if augment_audio:
        y = augment(samples=y, sample_rate=SAMPLE_RATE)

    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=40)
    return mfcc.T  # Transpose to shape (time, features)
!pip install --upgrade audiomentations
!pip install numpy==1.25.2
!pip install --upgrade --force-reinstall librosa
# 3.defines the audio augmentation pipeline (augment)
# Set a consistent sample rate
SAMPLE_RATE = 22050

# Import necessary classes from audiomentations
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Gain, Shift

# Augmentation pipeline using available audiomentations techniques
augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    Gain(min_gain_db=-6, max_gain_db=6, p=0.5),
    Shift(min_shift=-0.5, max_shift=0.5, p=0.5),
])
#generate a spectrogram
def compute_spectrogram(audio_path, sr=SAMPLE_RATE, n_fft=2048, hop_length=512):
    y = load_audio(audio_path, sr)
    if y is None:
        return None

    # Compute the short-time Fourier Transform (STFT)
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)

    # Convert to magnitude spectrogram
    spectrogram = np.abs(D)

    # Convert to decibels (log scale)
    spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)

    return spectrogram_db
import librosa
import soundfile as sf # Import soundfile
import numpy as np # Import numpy

def load_audio(audio_path, sr=SAMPLE_RATE):
    try:
        # Use soundfile to load audio
        y, current_sr = sf.read(audio_path)

        # Resample if necessary to match the target sample rate
        if current_sr != sr:
            # soundfile loads in float64, librosa resampling expects float32
            y = librosa.resample(y.astype(np.float32), orig_sr=current_sr, target_sr=sr)

        # soundfile loads multichannel audio as (samples, channels), convert to mono if needed
        if y.ndim > 1:
            y = np.mean(y, axis=1)

        print(f"  load_audio: Loaded {audio_path} using soundfile, shape: {y.shape}, sample rate: {sr}") # Updated print
        return y
    except Exception as e:
        print(f"Error loading audio file {audio_path} with soundfile: {e}") # Updated error message
        return None

def extract_mfcc_with_aug(audio_path, sr=SAMPLE_RATE, n_mfcc=40, augment_audio=False):
    y = load_audio(audio_path, sr)
    if y is None:
        return None

    if augment_audio:
        try:
            # The augment pipeline requires the sample rate
            y = augment(samples=y, sample_rate=sr) # Pass sample_rate here
        except Exception as e:
            print(f"Error applying augmentation to {audio_path}: {e}")
            return None

    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    return mfccs
# Uninstall the current version of numpy
!pip uninstall numpy -y
# Install a compatible version of numpy (e.g., 2.0.0)
!pip install numpy==2.0.0
# Reinstall soundfile and librosa to ensure compatibility
!pip install soundfile librosa
# 3.Feature extraction cell
features = []
labels = []

import os # Import the os module
from tqdm import tqdm # Import tqdm

# Define the data directory - UPDATE THIS PATH to your dataset location in Google Drive
data_dir = '/content/drive/MyDrive/infant_cry_dataset'

for label in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, label)
    if not os.path.isdir(class_dir): continue

    for file in tqdm(os.listdir(class_dir), desc=f'Processing {label}'):
        if not file.endswith('.wav'): continue
        file_path = os.path.join(class_dir, file)

        # Original sample
        mfcc = extract_mfcc_with_aug(file_path, augment_audio=False)
        # print(f"  Processed file (original): {file_path}, MFCC shape: {getattr(mfcc, 'shape', 'None')}, is None: {mfcc is None}") # Removed print
        if mfcc is not None and mfcc.ndim == 2:
            features.append(mfcc)
            labels.append(label)
            # print("    Appended original MFCC and label.") # Removed print
        # else: # Removed else block
            # print("    Skipped original sample due to invalid MFCC.") # Removed print


        # Augmented sample
        mfcc_aug = extract_mfcc_with_aug(file_path, augment_audio=True)
        # print(f"  Processed file (augmented): {file_path}, MFCC shape: {getattr(mfcc_aug, 'shape', 'None')}, is None: {mfcc_aug is None}") # Removed print
        if mfcc_aug is not None and mfcc_aug.ndim == 2:
            features.append(mfcc_aug)
            labels.append(label)
            # print("    Appended augmented MFCC and label.") # Removed print
        # else: # Removed else block
            # print("    Skipped augmented sample due to invalid MFCC.") # Removed print

# print(f"\nFeature extraction complete. Total features extracted: {len(features)}") # Removed print
# print(f"Total labels extracted: {len(labels)}") # Removed print
# 15. Visualize Spectrogram of a Sample Audio File, frequency content of a sample audio file from your dataset
import matplotlib.pyplot as plt
import librosa.display # Import librosa.display for plotting spectrograms
import numpy as np # Ensure numpy is imported

# Get a sample audio file path (assuming data_dir is defined and contains audio files)
# Use the same logic as in the soundfile test cell (bb9c8047) to find a sample file
data_dir = '/content/drive/MyDrive/infant_cry_dataset' # Ensure this path is correct
sample_file_path = None
sample_file_found = False
for root, _, files in os.walk(data_dir):
    for file in files:
        if file.endswith('.wav'):
            sample_file_path = os.path.join(root, file)
            sample_file_found = True
            break
    if sample_file_found:
        break

if sample_file_path:
    print(f"Computing and visualizing spectrogram for: {sample_file_path}")

    # Compute the spectrogram using the defined function
    spectrogram_db = compute_spectrogram(sample_file_path, sr=SAMPLE_RATE)

    if spectrogram_db is not None:
        plt.figure(figsize=(12, 6))
        librosa.display.specshow(spectrogram_db, sr=SAMPLE_RATE, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram (dB)')
        plt.tight_layout()
        plt.show()
    else:
        print("Could not compute spectrogram for the sample file.")

else:
    print(f"No .wav files found in the data directory: {data_dir}. Cannot visualize spectrogram.")
!pip install scikit-learn
# 7.balances your dataset using SMOTE
# Create a DataFrame from features and labels
# Assuming features is a list of numpy arrays and labels is a list of strings
# from sklearn.utils import resample # Remove resample
import pandas as pd
from imblearn.over_sampling import SMOTE # Import SMOTE
import numpy as np # Ensure numpy is imported
from collections import Counter # Import Counter

# Before applying SMOTE, ensure 'features' is a 2D array suitable for SMOTE
# If features are MFCCs of shape (time, features), they need to be flattened or processed
# to be (samples, features) for SMOTE.
# Assuming 'features' is a list of 2D numpy arrays (time, n_mfcc) and they have been padded/truncated
# to a consistent time length (e.g., 100 time steps with 40 features) and then flattened
# in a previous step to shape (samples, time * n_mfcc).
# Based on cell -sux5MWbD1_T, X_padded is already in the correct (n_samples, time_steps * n_mfcc) shape.

# Use X_padded and original labels (before upsampling)
# Assuming X_padded is the flattened features and labels is the list of original labels
X_flat = X_padded # Use the flattened, padded features

# Convert labels list to a numpy array for SMOTE
y_original = np.array(labels)


print("Original dataset shape %s" % Counter(y_original)) # Use Counter to show original distribution

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_flat, y_original)

print('Resampled dataset shape %s' % Counter(y_resampled)) # Show distribution after SMOTE

# Now X_resampled and y_resampled contain the balanced dataset.
# You would use X_resampled and y_resampled in subsequent steps (scaling, splitting, training)
# instead of the balanced_df DataFrame created by upsampling.
# 8.padding and flattening logic
import numpy as np # Ensure numpy is imported

# Pad or truncate to fixed length (e.g., 100 time steps)
def pad_or_truncate(mfcc, max_len=100, n_mfcc=40): # Added n_mfcc parameter
    if not isinstance(mfcc, np.ndarray):
        print(f"Warning: Input to pad_or_truncate is not a numpy array. Shape: {getattr(mfcc, 'shape', 'N/A')}")
        return None # Return None for non-array inputs

    if mfcc.ndim != 2:
        print(f"Warning: Input to pad_or_truncate is not a 2D numpy array. Shape: {mfcc.shape}")
        # Attempt to flatten and then reshape if possible, otherwise return None
        try:
            # Assuming the flattened size should be max_len * n_mfcc if it was a 2D array
            expected_flat_size = max_len * n_mfcc
            if mfcc.size == expected_flat_size:
                 return mfcc.reshape(max_len, n_mfcc)
            else:
                 print(f"Warning: Input to pad_or_truncate is not 2D and size {mfcc.size} does not match expected flattened size {expected_flat_size}.")
                 return None
        except Exception as e:
            print(f"Warning: Could not reshape non-2D input in pad_or_truncate: {e}")
            return None


    if mfcc.shape[1] != n_mfcc: # Check if the number of features matches expected
        print(f"Warning: Input to pad_or_truncate has unexpected number of features. Expected {n_mfcc}, got {mfcc.shape[1]}. Shape: {mfcc.shape}")
        # Decide how to handle: try to proceed or return None
        # For now, let's return None to indicate an issue
        return None

    if mfcc.shape[0] > max_len:  # Check time steps (axis 0 after transpose)
        return mfcc[:max_len, :]
    else:
        pad_width = max_len - mfcc.shape[0]
        return np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant') # Pad along time steps (axis 0)
# 9.Ensure your feature arrays have a consistent length.
import numpy as np # Ensure numpy is imported

X_padded = []
expected_flattened_size = 100 * 40 # max_len * n_mfcc

for i, mfcc in enumerate(features):
    # The pad_or_truncate function now handles non-2D and incorrect feature dimensions
    padded = pad_or_truncate(mfcc.T, max_len=100, n_mfcc=40) # Pass n_mfcc and transpose here

    if padded is not None:
        # Flatten the padded MFCCs
        flattened = padded.flatten()

        # Check if the flattened array has the expected size before appending
        if flattened.shape[0] == expected_flattened_size:
            X_padded.append(flattened)
        else:
            print(f"Skipping sample {i} due to unexpected flattened size after padding: Expected {expected_flattened_size}, got {flattened.shape[0]}")
    else:
        # This message will be printed if pad_or_truncate returned None
        print(f"Skipping sample {i} because padding failed or input was invalid.")


X_padded = np.array(X_padded)  # shape: (n_samples, time_steps * n_mfcc)
print(f"\nFinished padding and flattening. Shape of X_padded: {X_padded.shape}") # Added print
# 10.Performs feature scaling and label encoding
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical

# Use X_resampled and y_resampled generated from cell fGTVQCQnDucf (SMOTE)
# Reshape X_resampled to (n_samples, max_len, n_mfcc) before scaling if necessary.
# Assuming max_len = 100 and n_mfcc = 40 based on previous cells for reshaping
max_len = 100
n_mfcc = 40

if X_resampled.shape[0] > 0:
    # Reshape X_resampled to (n_samples, max_len, n_mfcc) before scaling
    # This assumes X_resampled is currently (n_samples, max_len * n_mfcc) which it should be after SMOTE on flattened features
    X_reshaped = X_resampled.reshape(-1, max_len, n_mfcc)

    # Encode labels (using y_resampled from SMOTE)
    # Fit the LabelEncoder on the *original* labels to ensure it knows about all classes
    # But transform the resampled labels
    if 'y_original' in locals() and y_original is not None and y_original.size > 0:
        le = LabelEncoder()
        le.fit(y_original) # Fit on original labels
        y_int = le.transform(y_resampled) # Transform resampled labels
        y = to_categorical(y_int, num_classes=len(le.classes_))  # One-hot encoding

        # Feature scaling - apply scaler on the reshaped 3D array
        # Scale across the feature dimension (axis=2)
        scaler = StandardScaler()
        # Reshape X_reshaped for scaling (flatten timesteps and features)
        X_scaled_flat = scaler.fit_transform(X_reshaped.reshape(-1, X_reshaped.shape[-1]))
        # Reshape back to the original 3D shape
        X = X_scaled_flat.reshape(X_reshaped.shape)


        print("Data successfully processed and scaled using resampled data.")
        print(f"Shape of X: {X.shape}")
        print(f"Shape of y: {y.shape}")
    else:
        print("Original labels (y_original) not found or empty. Cannot encode labels or proceed with scaling.")
        # Set X and y to empty arrays or None to avoid subsequent errors
        X = np.array([])
        y = np.array([])

else:
    print("X_resampled is empty. SMOTE may have failed or input was empty.")
    # Set X and y to empty arrays or None to avoid subsequent errors
    X = np.array([])
    y = np.array([])
import numpy as np
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# X and y must be pre-defined before this cell
# Ensure y is one-hot encoded
y_int = np.argmax(y, axis=1)  # Convert one-hot to integer labels

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_int), y=y_int)
class_weights_dict = dict(enumerate(class_weights))

# Hyperparameters
learning_rate = 0.001
epochs = 100
batch_size = 64
k_folds = 5

# Store history for each fold
history_list = []
last_fold_model = None
last_fold_X_val = None
last_fold_y_val = None

# Early Stopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# KFold Cross Validation
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
fold = 1

for train_idx, val_idx in kf.split(X):
    print(f"\n🚀 Fold {fold}")
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Define the CNN model
    model = Sequential()
    input_shape = (X_train.shape[1], X_train.shape[2])  # (timesteps, features)

    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))

    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))

    model.add(GlobalAveragePooling1D())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(y.shape[1], activation='softmax'))

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train model and store history
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights_dict,
        callbacks=[early_stop],
        verbose=1
    )
    history_list.append(history)

    if fold == k_folds:
        last_fold_model = model
        last_fold_X_val = X_val
        last_fold_y_val = y_val

    fold += 1

# Print final model summary
print("\n📋 Final CNN Model Summary:")
if last_fold_model:
    last_fold_model.summary()
else:
    print("Model summary not available.")

# 12.calculates and prints the model evaluation metrics
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Replace these with your actual predictions and labels
# Example dummy data to show the format
# If you already have y_true and y_pred, skip the two lines below
# y_true = ['cry', 'laugh', 'hungry', 'belly_pain', 'discomfort', 'noise', 'silence', 'burping'] * 100
# y_pred = ['cry', 'laugh', 'hungry', 'discomfort', 'discomfort', 'noise', 'silence', 'burping'] * 100

# Use the model and validation data from the last fold of training (cell gbKmpjtGNrA-)
if 'last_fold_model' in locals() and last_fold_model is not None and 'last_fold_X_val' in locals() and last_fold_X_val is not None and 'last_fold_y_val' in locals() and last_fold_y_val is not None:
    # Get true labels (from one-hot encoded y_val)
    y_true_int = np.argmax(last_fold_y_val, axis=1)
    # Convert integer labels back to original label names using the LabelEncoder from cell o0CqzYpo5kN1
    if 'le' in locals() and le is not None:
        y_true = le.inverse_transform(y_true_int)
    else:
         print("LabelEncoder (le) not found. Cannot convert integer labels to names. Using integer labels for evaluation.")
         y_true = y_true_int


    # Get model predictions
    y_pred_prob = last_fold_model.predict(last_fold_X_val)
    y_pred_int = np.argmax(y_pred_prob, axis=1)
    # Convert predicted integer labels back to original label names
    if 'le' in locals() and le is not None:
         y_pred = le.inverse_transform(y_pred_int)
    else:
         y_pred = y_pred_int # Use integer predictions if LabelEncoder not found


    print("Using trained model predictions and validation data for evaluation.")

    # ✅ Ensure correct format (should be 1D arrays of labels or integers)
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    # Sanity check
    print("y_true shape:", y_true.shape)
    print("y_pred shape:", y_pred.shape)

    # ✅ Calculate metrics
    # Check if using string labels or integer labels for sklearn metrics
    if isinstance(y_true[0], str):
         # Use labels argument if using string labels
         # Ensure all unique labels in y_true and y_pred are in le.classes_ for comprehensive report
         all_possible_labels = np.unique(np.concatenate((y_true, y_pred)))
         # Filter to only include labels that were present in the training data
         valid_labels_in_eval = [lbl for lbl in all_possible_labels if lbl in le.classes_]


         accuracy = accuracy_score(y_true, y_pred)
         precision = precision_score(y_true, y_pred, average='macro', zero_division=0, labels=valid_labels_in_eval)
         recall = recall_score(y_true, y_pred, average='macro', zero_division=0, labels=valid_labels_in_eval)
         f1 = f1_score(y_true, y_pred, average='macro', zero_division=0, labels=valid_labels_in_eval)

         # ✅ Full Classification Report (Use labels argument for clarity with string labels)
         print("\n----- Detailed Classification Report -----")
         print(classification_report(y_true, y_pred, zero_division=0, labels=le.classes_)) # Use le.classes_ to include all potential classes in the report

    else: # Assume using integer labels
         # No labels argument needed if using integer labels 0 to n-1
         accuracy = accuracy_score(y_true, y_pred)
         precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
         recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
         f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

         # ✅ Full Classification Report
         print("\n----- Detailed Classification Report -----")
         # Add target_names if le is available and using integer labels for report readability
         if 'le' in locals() and le is not None:
             print(classification_report(y_true, y_pred, zero_division=0, target_names=le.classes_))
         else:
              print(classification_report(y_true, y_pred, zero_division=0))


    # ✅ Print evaluation results
    print("\n----- Model Evaluation Metrics -----") # Moved printing of metrics to after report for better flow
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")


else:
    print("Model or validation data from the last fold not found. Please ensure cell gbKmpjtGNrA- ran successfully.")
    # Set dummy data to avoid errors in subsequent plotting cells if needed, or skip plotting
    y_true = np.array([])
    y_pred = np.array([])
#13. generates and displays the confusion matrix plot
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Assuming y_true and y_pred are available and are label names
# Also assuming 'le' is your LabelEncoder and has .classes_ defined

cm = confusion_matrix(y_true, y_pred, labels=le.classes_)
labels = le.classes_

plt.figure(figsize=(10, 8))

# ✅ Store the heatmap object to use for colorbar if needed
ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)

plt.title('Confusion Matrix', fontsize=16)
plt.xlabel('Predicted Labels', fontsize=12)
plt.ylabel('Actual Labels', fontsize=12)
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.tight_layout()

# ✅ Colorbar is automatically added, no need to call plt.colorbar()
plt.show()
#displays a comparison of the first 10 predicted labels versus the actual labels
import pandas as pd

# Create a comparison table
comparison_df = pd.DataFrame({
    'Index': range(10),
    'Predicted Label': y_pred[:10],
    'Actual Label': y_true[:10]
})

# Display the first 10 predictions vs actuals
print("\n----- First 10 Predictions vs Actual Labels -----")
print(comparison_df.to_string(index=False))
#training and validation history.
import matplotlib.pyplot as plt

# Get training and validation history from the history object
# Use the history from the last fold
if history_list:
    history = history_list[-1]

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.suptitle("Fig. 6\nTraining and Validation Progress", fontsize=16, y=1.03)
    plt.tight_layout(rect=[0, 0, 1, 0.97]) # Adjust layout to prevent title overlap
    plt.show()
else:
    print("No training history available. Please run the model training cell (gbKmpjtGNrA-) first.")
