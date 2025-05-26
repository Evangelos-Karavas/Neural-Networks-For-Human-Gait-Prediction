import os
import pandas as pd
import numpy as np
import joblib

from scipy.signal import butter, filtfilt, savgol_filter
from sklearn.preprocessing import StandardScaler

from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Reshape
from tensorflow.python.keras.optimizer_v2 import adam
import tensorflow.python.keras as tf_keras
from keras import __version__
tf_keras.__version__ = __version__

import matplotlib.pyplot as plt
from matplotlib import pyplot

#This is for saving the model (There were issues with __version__ when calling the function save_model)
import tensorflow.python.keras as tf_keras
from keras import __version__
tf_keras.__version__ = __version__


# ==============================================
#  Butter-worth Filtering - Smoothing
# ==============================================
def butterworth_filter(data, cutoff, fs, order, filter_type):
    """
    Applies a Butterworth filter (low-pass or high-pass).
    - cutoff: Cutoff frequency (Hz)
    - fs: Sampling frequency (Hz)
    - order: Filter order
    - filter_type: 'low' for low-pass, 'high' for high-pass
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)
    return filtfilt(b, a, data)

def savitzky_golay_filter(data, window_size=7, poly_order=3):
    """
    Applies Savitzky-Golay filter for trajectory smoothing.
    - window_size: Must be an odd number
    - poly_order: Degree of polynomial to fit
    """
    return savgol_filter(data, window_length=window_size, polyorder=poly_order)


# ================================
# Load Typical Data
# ================================
columns_to_read = ['LHipAngles (1)', 'LKneeAngles (1)', 'LAnkleAngles (1)',
                   'RHipAngles (1)', 'RKneeAngles (1)', 'RAnkleAngles (1)']

# Load Typical Data
data = pd.read_excel('Normal/randomized_data_healthy.xlsx', usecols=columns_to_read)
data.fillna(0, inplace=True)

# Define Sampling Frequency
fs = 1000  # Adjust if needed (depends on dataset)

# Apply Filters to Each Column
for col in columns_to_read:
    data[col] = butterworth_filter(data[col].values, cutoff=250, fs=fs, order=2, filter_type='low')
    data[col] = butterworth_filter(data[col].values, cutoff=15, fs=fs, order=2, filter_type='high')
    data[col] = savitzky_golay_filter(data[col].values)

# ==============================================
# Load and Process Cerebral Palsy Data
# ==============================================
folder_path = "Data/"
all_cp_data = []
cp_data_counter = 0
for file_name in os.listdir(folder_path):
    if file_name.endswith('.xlsx'):
        file_path = os.path.join(folder_path, file_name)
        df1 = pd.read_excel(file_path, 'Data', usecols=columns_to_read, skiprows=[1, 2])
        df1.fillna(0, inplace=True)

        # Apply Filters
        for col in columns_to_read:
            df1[col] = butterworth_filter(df1[col].values, cutoff=250, fs=fs, order=2, filter_type='low')
            df1[col] = butterworth_filter(df1[col].values, cutoff=15, fs=fs, order=2, filter_type='high')
            df1[col] = savitzky_golay_filter(df1[col].values)
            if cp_data_counter > 500:
                break
            cp_data_counter =+ 1
        all_cp_data.append(df1)

# Merge CP Data
data_cerebral_palsy = pd.concat(all_cp_data, ignore_index=True)

# Properly Scale the data to input in Neural Network
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
data_cp_scaled = scaler.fit_transform(data_cerebral_palsy)

#Save scaler for ROS node
joblib.dump(scaler, "Scaler/standard_scaler.save")

# ==========================
# Reshape Data for CNN
# ==========================
num_samples_typical = len(data_scaled) // 51  
num_samples_cp = len(data_cp_scaled) // 51

data_cnn_typical = data_scaled[:num_samples_typical * 51].reshape(num_samples_typical, 51, 6)
data_cnn_cp = data_cp_scaled[:num_samples_cp * 51].reshape(num_samples_cp, 51, 6)

# ==========================
# Split for Train / Val / Test
# ==========================
X_train = data_cnn_typical[:450]
X_test = data_cnn_cp[:450]
X_val = np.vstack((data_cnn_typical[450:500], data_cnn_cp[450:500]))  

# ==========================
# Separate Inputs (X) and Outputs (Y)
# ==========================
X_train_input, Y_train_output = X_train[:, :, :6], X_train[:, :, :6]  
X_val_input, Y_val_output = X_val[:, :, :6], X_val[:, :, :6]  
X_test_input, Y_test_output = X_test[:, :, :6], X_test[:, :, :6]  

# ==============================================
# Build CNN Model
# ==============================================
def build_cnn():
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', padding='same', input_shape=(51, 6)),
        MaxPooling1D(pool_size=2),
        Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        Flatten(),
        Dense(64, activation='relu'),  # bottleneck
        Dense(64, activation='relu'),
        Dense(51 * 6, activation='linear'),
        Reshape((51, 6))
    ])
    
    model.compile(optimizer=adam.Adam(learning_rate=0.001), loss='mse', metrics=['accuracy', 'mae'])
    return model

# Build model
model = build_cnn()

# Check model summary
model.summary()

# Train model
history = model.fit(X_train_input, Y_train_output, epochs=200, batch_size=51, validation_data=(X_val_input, Y_val_output))

# ==============================================
# Evaluate Model
# ==============================================
def evaluate_model(model, X, Y, label):
    loss, accuracy, mae = model.evaluate(X, Y, verbose=1)
    print(f"🔹 {label} Evaluation:")
    print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, MAE: {mae:.4f}")
    print("="*90)

evaluate_model(model, X_train_input, Y_train_output, "Training Data")
evaluate_model(model, X_val_input, Y_val_output, "Validation Data")
evaluate_model(model, X_test_input, Y_test_output, "Testing Data")

model.save("Saved_Models/Timestamp_cnn_model.keras", include_optimizer=True)

# ==============================================
# Predict Next 4 Steps
# ==============================================
def predict_next_steps(model, last_51_rows, num_steps=4):
    """Predicts the next N steps using the trained CNN model."""
    predicted_steps = []
    current_input = last_51_rows.reshape(1, 51, 6)  # Ensure shape is correct

    for _ in range(num_steps):  
        predicted_step = model.predict(current_input)  # Predict full 6 features
        predicted_steps.append(predicted_step.reshape(51, 6))  
        current_input = predicted_step.reshape(1, 51, 6)  # Use predicted values as new input

    predicted_steps = np.vstack(predicted_steps)  # Stack predictions into a sequence
    return scaler.inverse_transform(predicted_steps)  # Convert back to original scale

# Get last known step for prediction
last_known_step = X_train_input[-1]
last_known_step_cp = X_test_input[-1] 
num_steps=4
next_steps_prediction = predict_next_steps(model, last_known_step, num_steps)
next_steps_prediction_cp = predict_next_steps(model, last_known_step_cp, num_steps=4)
# Get actual next 4 steps from dataset for plotting
actual_next_4_steps = data.iloc[len(data) - 204:, :].values  
actual_next_4_steps_cp = data_cerebral_palsy.iloc[len(data_cerebral_palsy) - 204:, :].values  

# Convert the numpy array to a pandas DataFrame
column_names = ['LHipAngles (1)', 'LKneeAngles (1)', 'LAnkleAngles (1)',
                'RHipAngles (1)', 'RKneeAngles (1)', 'RAnkleAngles (1)']
next_steps_prediction_df = pd.DataFrame(next_steps_prediction, columns=column_names)
next_steps_prediction_cp_df = pd.DataFrame(next_steps_prediction_cp, columns=column_names)


next_steps_prediction_df = next_steps_prediction_df[column_names]
next_steps_prediction_cp_df = next_steps_prediction_cp_df[column_names]



# ==========================
# Save Predictions
# ==========================
predicted_df = pd.DataFrame(next_steps_prediction_df, columns=column_names)
predicted_df.to_excel("Predictions/timestamps_typical_cnn.xlsx", index=False)

predicted_df_cp = pd.DataFrame(next_steps_prediction_cp_df, columns=column_names)
predicted_df_cp.to_excel("Predictions/timestamps_cp_cnn.xlsx", index=False)

print("Prediction complete. Data saved")


# ====================================================
# Plot for Data (Typical_Data  ||  CP_Data)
# ====================================================
# def plot_comparison(predicted, actual):
#     """Plots actual vs predicted joint angles."""
#     time = np.arange(actual.shape[0])  # Time index for the dataset

#     labels_left = ['LHipAngles', 'LKneeAngles', 'LAnkleAngles']
#     labels_right = ['RHipAngles', 'RKneeAngles', 'RAnkleAngles']

#     fig, axes = plt.subplots(6, 1, figsize=(12, 16), sharex=True)

#     # Left Leg Joint Angles (Columns 0,1,2)
#     for i in range(len(labels_left)):
#         axes[i].plot(time, actual[:, i], label=f"Actual {labels_left[i]}", color='blue')
#         axes[i].plot(time, predicted[:, i], label=f"Predicted {labels_left[i]}", linestyle='dashed', color='red')
#         axes[i].set_ylabel("Angle")
#         axes[i].legend()
#         axes[i].set_title(f"Comparison: {labels_left[i]}")

#     # Right Leg Joint Angles (Columns 3,4,5)
#     for i in range(len(labels_right)):
#         axes[i + 3].plot(time, actual[:, i + 3], label=f"Actual {labels_right[i]}", color='blue')
#         axes[i + 3].plot(time, predicted[:, i + 3], label=f"Predicted {labels_right[i]}", linestyle='dashed', color='red')
#         axes[i + 3].set_ylabel("Angle")
#         axes[i + 3].legend()
#         axes[i + 3].set_title(f"Comparison: {labels_right[i]}")
        
#     axes[-1].set_xlabel("Time (Phase Progression)")

#     plt.tight_layout()
#     plt.show()

# plot_comparison(next_steps_prediction_df.values, actual_next_4_steps)
# plot_comparison(next_steps_prediction_cp_df.values, actual_next_4_steps_cp)

def plot_multiple_knee_predictions(actual_data, predicted_steps_list, label="Typical"):
    """
    Plots actual knee angles for multiple strides and overlays multiple predicted strides.
    
    Parameters:
    - actual_data: shape (N, 8)
    - predicted_steps_list: list of predicted arrays (each of shape (51, 8))
    """
    stride_length = 51
    time = np.arange(stride_length)

    # Define color map for predictions
    colors = pyplot.get_cmap('tab10', len(predicted_steps_list))

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    axs[0].set_title(f"{label} - Left Ankle Angles")
    axs[1].set_title(f"{label} - Right Ankle Angles")

    # Plot actual left and right knee angles for each stride
    num_actual_strides = len(actual_data) // stride_length
    for i in range(num_actual_strides):
        start = i * stride_length
        end = start + stride_length
        axs[0].plot(time, actual_data[start:end, 4], color='lightgray', alpha=0.5)
        axs[1].plot(time, actual_data[start:end, 5], color='lightgray', alpha=0.5)

    # Plot each predicted stride
    # for idx, pred in enumerate(predicted_steps_list):
    #     axs[0].plot(time, pred[:, 2], color=colors(idx), label=f"Prediction {idx+1}", linewidth=2)
    #     axs[1].plot(time, pred[:, 3], color=colors(idx), label=f"Prediction {idx+1}", linewidth=2)

    axs[0].set_xlabel("Timestep")
    axs[0].set_ylabel("Left Ankle Angle")
    axs[0].legend()

    axs[1].set_xlabel("Timestep")
    axs[1].set_ylabel("Right Ankle Angle")
    axs[1].legend()

    plt.tight_layout()
    plt.show()


actual_next_20_steps = data.iloc[len(data) - 1020:, :].values
actual_next_20_steps_cp = data_cerebral_palsy.iloc[len(data_cerebral_palsy) - 1020:, :].values

predicted_strides = []
current_input = last_known_step.reshape(1, 51, 6)

for _ in range(5):
    pred = model.predict(current_input)
    predicted_strides.append(scaler.inverse_transform(pred[0]))
    current_input = pred  # Use last prediction as next input

predicted_strides_cp = []
last_known_step_cp = last_known_step_cp.reshape(1, 51, 6)

for _ in range(5):
    pred = model.predict(last_known_step_cp)
    predicted_strides_cp.append(scaler.inverse_transform(pred[0]))
    last_known_step_cp = pred  # Use last prediction as next input

plot_multiple_knee_predictions(actual_next_20_steps, predicted_strides, label="Typical")
plot_multiple_knee_predictions(actual_next_20_steps, predicted_strides_cp, label="CP")

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Combined Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()