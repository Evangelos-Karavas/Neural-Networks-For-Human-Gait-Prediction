import os
import pandas as pd
import numpy as np

# Set the folder path
folder_path = "Normal"

# Define columns to read
columns_to_read = ['LHipAngles (1)', 'RHipAngles (1)', 'LKneeAngles (1)', 'RKneeAngles (1)', 
                   'LAnkleAngles (1)', 'RAnkleAngles (1)', 'Left Foot Off', 'Right Foot Off']

# Collect all data
all_data = []

for file in os.listdir(folder_path):
    if file.endswith(".xlsx"):
        file_path = os.path.join(folder_path, file)
        df = pd.read_excel(file_path, sheet_name="Data", usecols=columns_to_read, skiprows=[1, 2])
        all_data.append(df)

merged_data = pd.concat(all_data, ignore_index=True)
column_std = merged_data.std()

num_total_cycles = 500
num_original_cycles = 60
num_noisy_cycles = num_total_cycles - num_original_cycles
randomized_data = []
base_noise = 0.2

# Helper function to broadcast foot-off values
def broadcast_foot_off(cycle):
    lfo_value = cycle['Left Foot Off'].dropna().iloc[0]
    rfo_value = cycle['Right Foot Off'].dropna().iloc[0]
    cycle['Left Foot Off'] = lfo_value
    cycle['Right Foot Off'] = rfo_value
    return cycle

# Keep 60 original gait cycles with foot-off broadcasting
for i in range(num_original_cycles):
    start = np.random.randint(0, len(merged_data) // 51) * 51
    cycle = merged_data.iloc[start:start + 51].copy()
    cycle = broadcast_foot_off(cycle)
    randomized_data.append(cycle)

# Generate 440 noisy versions
for i in range(num_noisy_cycles):
    base_cycle = randomized_data[i % num_original_cycles].copy()

    noise = np.random.normal(loc=0, scale=base_noise * column_std.values, size=base_cycle.shape)
    noise = np.clip(noise, -0.5, 0.5)

    noisy_cycle = base_cycle + noise

    # Restore foot-off values (we don't want to add noise to these)
    noisy_cycle['Left Foot Off'] = base_cycle['Left Foot Off']
    noisy_cycle['Right Foot Off'] = base_cycle['Right Foot Off']

    randomized_data.append(noisy_cycle)

# Combine all cycles and export
final_df = pd.concat(randomized_data, ignore_index=True)
output_path = "randomized_data_healthy.xlsx"
final_df.to_excel(output_path, index=False)

print("Shape of df:", final_df.shape)
print(f"Saved randomized data to {output_path}")