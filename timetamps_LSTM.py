import os
import pandas as pd
import numpy as np
from scipy import signal
#from scipy.signal import butter, filtfilt
#import statsmodels.api as sm

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.python.keras.metrics import RootMeanSquaredError
from tensorflow.python.keras.optimizer_v2 import adam
import tensorflow.python.keras as tf_keras
from keras import __version__
tf_keras.__version__ = __version__

import matplotlib.pyplot as plt

#This is for saving the model (There were issues with __version__ when calling the function save_model)
import tensorflow.python.keras as tf_keras
from keras import __version__
tf_keras.__version__ = __version__

# ================================
# Load Typical Data
# ================================
data_typical = "randomized_data_healthy.xlsx"
columns_to_read = ['LHipAngles (1)', 'LKneeAngles (1)', 'LAnkleAngles (1)',
                   'RHipAngles (1)', 'RKneeAngles (1)', 'RAnkleAngles (1)']
data = pd.read_excel(data_typical, usecols=columns_to_read)

if not data.empty:
    for col in columns_to_read:
        last_value = data[col].values[-1]
        first_value = data[col].values[0]
        divergence = np.abs(last_value - first_value)
        if divergence > 5:
            mean_value = (last_value + first_value) / 2
            data.loc[data.index[-1], col] = mean_value
            data.loc[data.index[0], col] = mean_value
            for col in columns_to_read:
                last_value = data[col].values[-1]
                first_value = data[col].values[0]
                divergence = np.abs(last_value - first_value)
                if divergence > 2:
                    mean_value = (last_value + first_value) / 2
                    data.loc[data.index[-1], col] = mean_value
                    data.loc[data.index[0], col] = mean_value
data.fillna(0, inplace=True)  # Fill any missing values with 0

folder_path = 'Data/'
file_counter = 0
data_cerebral_palsy = pd.DataFrame()
for file_name in os.listdir(folder_path):
    if file_name.endswith('.xlsx'):
        file_counter += 1
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_excel(file_path, 'Data', usecols=columns_to_read, skiprows=[1, 2])
        data_cerebral_palsy = pd.concat([data_cerebral_palsy, df], ignore_index=True)
        if file_counter >= 500:
            break
# Check for divergence and fix by averaging first and last values
if not data_cerebral_palsy.empty:
    for col in columns_to_read:
        last_value = data_cerebral_palsy[col].values[-1]
        first_value = data_cerebral_palsy[col].values[0]
        divergence = np.abs(last_value - first_value)
        if divergence > 5:
            mean_value = (last_value + first_value) / 2
            data_cerebral_palsy.loc[data_cerebral_palsy.index[-1], col] = mean_value
            data_cerebral_palsy.loc[data_cerebral_palsy.index[0], col] = mean_value
            for col in columns_to_read:
                last_value = data_cerebral_palsy[col].values[-1]
                first_value = data_cerebral_palsy[col].values[0]
                divergence = np.abs(last_value - first_value)
                if divergence > 2:
                    mean_value = (last_value + first_value) / 2
                    data_cerebral_palsy.loc[data_cerebral_palsy.index[-1], col] = mean_value
                    data_cerebral_palsy.loc[data_cerebral_palsy.index[0], col] = mean_value

data_cerebral_palsy.fillna(0, inplace=True)  # Fill any missing values with 0
# Properly Scale the data to input in Neural Network
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
data_cp_scaled = scaler.fit_transform(data_cerebral_palsy)

# Reshape Data for LSTM (samples, timesteps, features)
num_samples_typical = len(data_scaled) // 51  
num_samples_cp = len(data_cp_scaled) // 51

data_lstm_typical = data_scaled[:num_samples_typical * 51].reshape(num_samples_typical, 51, 6)
data_lstm_cp = data_cp_scaled[:num_samples_cp * 51].reshape(num_samples_cp, 51, 6)

# Ensure at least 1 sample for validation
split_idx_typical = max(1, int(0.2 * num_samples_typical))  
split_idx_cp = max(1, int(0.2 * num_samples_cp))  

# Train and Validation Split
X_train = data_lstm_typical[split_idx_typical:]  # Train only on typical data (exclude validation part)
X_val = np.vstack((data_lstm_typical[:split_idx_typical], data_lstm_cp[:split_idx_cp]))  # Validation mix of typical & CP
X_test = data_lstm_cp  # Test only on CP data

print("num_samples_typical:", num_samples_typical)
print("num_samples_cp:", num_samples_cp)
print("split_idx_typical:", split_idx_typical)
print("split_idx_cp:", split_idx_cp)
print("X_val shape after split:", X_val.shape)

# ==============================================
# Separate Inputs (X) and Outputs (Y) (Both 6 Features)
# ==============================================
X_train_input, Y_train_output = X_train[:, :, :6], X_train[:, :, :6]
X_val_input, Y_val_output = X_val[:, :, :6], X_val[:, :, :6]
X_test_input, Y_test_output = X_test[:, :, :6], X_test[:, :, :6]

# ==============================================
# Build LSTM Model
# ==============================================
model = Sequential([
    LSTM(100, activation='tanh', return_sequences=True, input_shape=(51, 6)),
    LSTM(100, activation='tanh', return_sequences=True, input_shape=(51, 6)),
    #LSTM(64, activation='tanh', return_sequences=True, input_shape=(51, 6)),  # Activation: tanh, sigmoid, relu
    Dropout(0.4),
    Dense(6, activation='linear')])

model.compile(optimizer=adam.Adam(learning_rate=0.003), loss='mse', metrics=['accuracy', 'mae', RootMeanSquaredError()])

# Train Model
history = model.fit(X_train_input, Y_train_output, epochs=150, batch_size=102, validation_data=(X_val_input, Y_val_output))

# ==============================================
# Evaluate Model
# ==============================================
def evaluate_model(model, X, Y, label):
    loss, accuracy, mae = model.evaluate(X, Y, verbose=1)
    rmse = np.sqrt(loss)  # Compute RMSE from MSE loss
    
    print(f"🔹 {label} Evaluation:")
    print(f"Loss (MSE): {loss:.4f}, RMSE: {rmse:.4f}, Accuracy: {accuracy:.4f}, MAE: {mae:.4f}")
    print("="*90)

# Run evaluations with RMSE
evaluate_model(model, X_train_input, Y_train_output, "Training Data")
evaluate_model(model, X_val_input, Y_val_output, "Validation Data")
evaluate_model(model, X_test_input, Y_test_output, "Testing Data")

model.save("Saved_Models/Timestamp_lstm_model.keras", include_optimizer=True)

# ==============================================
# Predict Next 4 Steps
# ==============================================
def predict_next_steps(model, last_51_rows, num_steps=4):
    """Predicts the next N steps using the trained LSTM model."""
    predicted_steps = []
    current_input = last_51_rows.reshape(1, 51, 6)  # Ensure shape is correct

    for _ in range(num_steps):  
        predicted_step = model.predict(current_input)  # Predict full 6 features
        predicted_steps.append(predicted_step.reshape(51, 6))  
        current_input = predicted_step.reshape(1, 51, 6)  # Use predicted values as new input

    predicted_steps = np.vstack(predicted_steps)  # Stack predictions into a sequence
    return scaler.inverse_transform(predicted_steps)  # Convert back to original scale

# Get last known step for prediction
last_known_step = data_lstm_typical[-1]
last_known_step_cp = data_lstm_cp[-1] 

next_steps_prediction = predict_next_steps(model, last_known_step, num_steps=4)
next_steps_prediction_cp = predict_next_steps(model, last_known_step_cp, num_steps=4)

data_typically_developed_reshaped = data_lstm_typical.reshape(-1, 6)
data_cerebral_palsy_reshaped = data_lstm_cp.reshape(-1, 6)

# Convert back to DataFrame
data_typically_developed_df = pd.DataFrame(data_typically_developed_reshaped, columns=[
    'LHipAngles (1)', 'LKneeAngles (1)', 'LAnkleAngles (1)',
    'RHipAngles (1)', 'RKneeAngles (1)', 'RAnkleAngles (1)'
])
data_cerebral_palsy_df = pd.DataFrame(data_cerebral_palsy_reshaped, columns=[
    'LHipAngles (1)', 'LKneeAngles (1)', 'LAnkleAngles (1)',
    'RHipAngles (1)', 'RKneeAngles (1)', 'RAnkleAngles (1)'
])
actual_next_4_steps = data_typically_developed_df.iloc[-204:, :].values
actual_next_4_steps_cp = data_cerebral_palsy_df.iloc[-204:, :].values


# Convert the numpy array to a pandas DataFrame
column_names = ['LHipAngles (1)', 'LKneeAngles (1)', 'LAnkleAngles (1)',
                'RHipAngles (1)', 'RKneeAngles (1)', 'RAnkleAngles (1)']
next_steps_prediction_df = pd.DataFrame(next_steps_prediction, columns=column_names)
next_steps_prediction_cp_df = pd.DataFrame(next_steps_prediction_cp, columns=column_names)

next_steps_prediction_df = next_steps_prediction_df[column_names]

next_steps_prediction_cp_df = next_steps_prediction_cp_df[column_names]
columns_to_read = column_names

for col in columns_to_read:
    for i in range(204):
        last_value = next_steps_prediction_df[col].values[i-1]
        first_value = next_steps_prediction_df[col].values[i]
        divergence = np.abs(last_value - first_value)
        if divergence > 2:
            mean_value = (last_value + first_value) / 2
            next_steps_prediction_df[col].values[i] = mean_value
            next_steps_prediction_df[col].values[i-1] = mean_value
        last_value = next_steps_prediction_cp_df[col].values[i-1]
        first_value = next_steps_prediction_cp_df[col].values[i]
        divergence = np.abs(last_value - first_value)
        if divergence > 2:
            mean_value = (last_value + first_value) / 2
            next_steps_prediction_cp_df[col].values[i] = mean_value
            next_steps_prediction_cp_df[col].values[i-1] = mean_value


# ====================================================
# Plot for Data (Typical_Data  ||  CP_Data)
# ====================================================
def plot_comparison(predicted, actual):
    """Plots actual vs predicted joint angles and marks swing phase when the predicted phase variable changes sharply."""
    time = np.arange(actual.shape[0])  # Time index for the dataset

    labels_left = ['LHipAngles', 'RHipAngles', 'LKneeAngles']
    labels_right = ['RKneeAngles', 'LAnkleAngles', 'RAnkleAngles']

    fig, axes = plt.subplots(6, 1, figsize=(12, 16), sharex=True)

    # Left Leg Joint Angles (Columns 0,1,2)
    for i in range(len(labels_left)):
        axes[i ].plot(time, actual[:, i], label=f"Actual {labels_left[i]}", color='blue')
        axes[i ].plot(time, predicted[:, i], label=f"Predicted {labels_left[i]}", linestyle='dashed', color='red')
        axes[i ].set_ylabel("Angle")
        axes[i ].legend()
        axes[i ].set_title(f"Comparison: {labels_left[i]}")

    # Right Leg Joint Angles (Columns 3,4,5)
    for i in range(len(labels_right)):
        axes[i + 3].plot(time, actual[:, i + 3], label=f"Actual {labels_right[i]}", color='blue')
        axes[i + 3].plot(time, predicted[:, i + 3], label=f"Predicted {labels_right[i]}", linestyle='dashed', color='red')
        axes[i + 3].set_ylabel("Angle")
        axes[i + 3].legend()
        axes[i + 3].set_title(f"Comparison: {labels_right[i]}")

    axes[-1].set_xlabel("Time (Phase Progression)")

    plt.tight_layout()
    plt.show()

actual_next_4_steps = scaler.inverse_transform(actual_next_4_steps)
actual_next_4_steps_cp = scaler.inverse_transform(actual_next_4_steps_cp)
plot_comparison(next_steps_prediction_df.values, actual_next_4_steps)
plot_comparison(next_steps_prediction_cp_df.values, actual_next_4_steps_cp)

# ==========================
# Save Predictions
# ==========================
predicted_df = pd.DataFrame(next_steps_prediction_df, columns=['LHipAngles', 'RHipAngles', 'LKneeAngles',
                                                            'RKneeAngles', 'LAnkleAngles', 'RAnkleAngles'])
predicted_df.to_excel("Predictions/timestamps_typical_lstm.xlsx", index=False)

predicted_df_cp = pd.DataFrame(next_steps_prediction_cp_df, columns=['LHipAngles', 'RHipAngles', 'LKneeAngles',
                                                            'RKneeAngles', 'LAnkleAngles', 'RAnkleAngles'])
predicted_df_cp.to_excel("Predictions/timestamps_cp_lstm.xlsx", index=False)
print("Prediction complete. Data saved")
