import os
import datetime
import pandas as pd
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from sklearn.model_selection import train_test_split
# Define the directory where the CSV files are stored
input_dir = '1. Raw battery data'

# Load each CSV file into a separate DataFrame
B0005_charge = pd.read_csv(os.path.join(input_dir, 'B0005_charge.csv'))
B0006_charge = pd.read_csv(os.path.join(input_dir, 'B0006_charge.csv'))
B0007_charge = pd.read_csv(os.path.join(input_dir, 'B0007_charge.csv'))
B0018_charge = pd.read_csv(os.path.join(input_dir, 'B0018_charge.csv'))

B0005_discharge = pd.read_csv(os.path.join(input_dir, 'B0005_discharge.csv'))
B0006_discharge = pd.read_csv(os.path.join(input_dir, 'B0006_discharge.csv'))
B0007_discharge = pd.read_csv(os.path.join(input_dir, 'B0007_discharge.csv'))
B0018_discharge = pd.read_csv(os.path.join(input_dir, 'B0018_discharge.csv'))

def process_cycle_data(cycle_data):
    # Ensure the length is divisible by 10
    remainder = len(cycle_data) % 10
    if remainder != 0:
        cycle_data = cycle_data[:-remainder]
    
    # Reshape the data to have 10 samples per row
    reshaped_data = cycle_data.reshape(-1, 10)
    
    # Average each column
    averaged_data = reshaped_data.mean(axis=0)
    
    return averaged_data

def process_battery_data(df):
    processed_data = []

    # Get the unique cycles
    unique_cycles = df['cycle'].unique()

    for cycle in unique_cycles:
        # Filter data for the current cycle
        cycle_data = df[df['cycle'] == cycle]

        # Process 'voltage_measured'
        voltage_data = process_cycle_data(cycle_data['voltage_measured'].values)
        
        # Process 'current_measured'
        current_data = process_cycle_data(cycle_data['current_measured'].values)
        
        # Process 'temperature_measured'
        temperature_data = process_cycle_data(cycle_data['temperature_measured'].values)
        
        # Concatenate the averaged data into a single row
        combined_data = np.concatenate([voltage_data, current_data, temperature_data])
        processed_data.append(combined_data)
    
    # Create a DataFrame from the processed data
    processed_df = pd.DataFrame(processed_data, columns=[f'v{i+1}' for i in range(10)] + 
                                                  [f'c{i+1}' for i in range(10)] + 
                                                  [f't{i+1}' for i in range(10)])
    
    return processed_df

def process_battery_data_combined(df, df2):
    processed_data = []

    # Get the unique cycles
    unique_cycles_ch = df['cycle'].unique()
    # unique_cycles_disch = df2['cycle'].unique()
    
    for cycle in unique_cycles_ch:
        # Filter data for the current cycle
        cycle_data_ch = df[df['cycle'] == cycle]
        cycle_data_disch = df2[df2['cycle'] == cycle]
        # Process 'voltage_measured'
        voltage_data_ch = process_cycle_data(cycle_data_ch['voltage_measured'].values)
        voltage_data_disch = process_cycle_data(cycle_data_disch['voltage_measured'].values)
        # Process 'current_measured'
        current_data_ch = process_cycle_data(cycle_data_ch['current_measured'].values)
        current_data_disch = process_cycle_data(cycle_data_disch['current_measured'].values)
        # Process 'temperature_measured'
        temperature_data_ch = process_cycle_data(cycle_data_ch['temperature_measured'].values)
        temperature_data_disch = process_cycle_data(cycle_data_disch['temperature_measured'].values)
        # Concatenate the averaged data into a single row
        combined_data = np.concatenate([voltage_data_ch, current_data_ch, temperature_data_ch, voltage_data_disch, current_data_disch,temperature_data_disch])
        processed_data.append(combined_data)
    
    # Create a DataFrame from the processed data
    processed_df = pd.DataFrame(processed_data, columns=[f'v_ch{i+1}' for i in range(10)] + 
                                                  [f'c_ch{i+1}' for i in range(10)] + 
                                                  [f't_ch{i+1}' for i in range(10)]+
                                                  [f'v_dis{i+1}' for i in range(10)] + 
                                                  [f'c_dis{i+1}' for i in range(10)] + 
                                                  [f't_dis{i+1}' for i in range(10)])
    
    return processed_df
# Example usage
# Assuming B0005_charge is already loaded as a pandas DataFrame
# B0005_charge = pd.read_csv('B0005_charge.csv')

processed_VIT_B0005_df = process_battery_data(B0005_charge)
processed_VIT_B0006_df = process_battery_data(B0006_charge)
processed_VIT_B0007_df = process_battery_data(B0007_charge)
processed_VIT_B0018_df = process_battery_data(B0018_charge)

processed_disVIT_B0005_df = process_battery_data(B0005_discharge)
processed_disVIT_B0006_df = process_battery_data(B0006_discharge)
processed_disVIT_B0007_df = process_battery_data(B0007_discharge)
processed_disVIT_B0018_df = process_battery_data(B0018_discharge)

combined_VIT_B0005_df = process_battery_data_combined(B0005_charge, B0005_discharge)
combined_VIT_B0006_df = process_battery_data_combined(B0006_charge, B0006_discharge)
combined_VIT_B0007_df = process_battery_data_combined(B0007_charge, B0007_discharge)
combined_VIT_B0018_df = process_battery_data_combined(B0018_charge, B0018_discharge)

processed_VIT_B0005 = process_battery_data(B0005_charge).to_numpy()
processed_VIT_B0006 = process_battery_data(B0006_charge).to_numpy()
processed_VIT_B0007 = process_battery_data(B0007_charge).to_numpy()
processed_VIT_B0018 = process_battery_data(B0018_charge).to_numpy()

processed_disVIT_B0005 = process_battery_data(B0005_discharge).to_numpy()
processed_disVIT_B0006 = process_battery_data(B0006_discharge).to_numpy()
processed_disVIT_B0007 = process_battery_data(B0007_discharge).to_numpy()
processed_disVIT_B0018 = process_battery_data(B0018_discharge).to_numpy()

combined_VIT_B0005 = process_battery_data_combined(B0005_charge, B0005_discharge).to_numpy()
combined_VIT_B0006 = process_battery_data_combined(B0006_charge, B0006_discharge).to_numpy()
combined_VIT_B0007 = process_battery_data_combined(B0007_charge, B0007_discharge).to_numpy()
combined_VIT_B0018 = process_battery_data_combined(B0018_charge, B0018_discharge).to_numpy()


def remove_duplicate_cycles(df):
    # Drop duplicates based on 'cycle' column and keep the first occurrence
    unique_cycles_df = df.drop_duplicates(subset='cycle', keep='first')
    
    # Create a new DataFrame with unique 'cycle' and 'capacity' columns
    processed_df = unique_cycles_df[['cycle', 'capacity']].reset_index(drop=True)
    
    return processed_df

# Example usage
# Assuming B0005_charge, B0006_charge, B0007_charge, B0018_charge are already loaded as pandas DataFrames
# B0005_charge = pd.read_csv('B0005_charge.csv')
# B0006_charge = pd.read_csv('B0006_charge.csv')
# B0007_charge = pd.read_csv('B0007_charge.csv')
# B0018_charge = pd.read_csv('B0018_charge.csv')

# CC: Capacity and Cycle
processed_B0005_CC = remove_duplicate_cycles(B0005_charge)
processed_B0006_CC = remove_duplicate_cycles(B0006_charge)
processed_B0007_CC = remove_duplicate_cycles(B0007_charge)
processed_B0018_CC = remove_duplicate_cycles(B0018_charge)

# Extract capacity data
capacities_B0005 = processed_B0005_CC['capacity'].values.reshape(-1, 1) # reshape from numpy row to column
capacities_B0006 = processed_B0006_CC['capacity'].values.reshape(-1, 1)
capacities_B0007 = processed_B0007_CC['capacity'].values.reshape(-1, 1)
capacities_B0018 = processed_B0018_CC['capacity'].values.reshape(-1, 1)


def minmax_norm(charInput, InitC, cap):
    # Initialize the scaler
    scaler_x = MinMaxScaler()
    
    # Normalize xData
    xData = scaler_x.fit_transform(charInput)
    
    # Calculate the difference in lengths
    comp = len(charInput) - len(cap)  # will be zero 
    # Ensure cap is a 1-dimensional array
    cap = cap.flatten()
        
    # Create yData array
    yData = np.concatenate([InitC * np.ones(comp), cap])  # InitC*comp compensates for missing data but it will be zero.
    
    # Normalize yData
    scaler_y = MinMaxScaler()
    yData = scaler_y.fit_transform(yData.reshape(-1, 1))
    
    return xData, yData, scaler_x, scaler_y

# To later reverse this normalization
def inverse_minmax_norm(xData, yData, scaler_x, scaler_y):
    # Reverse normalization for xData
    original_xData = scaler_x.inverse_transform(xData)
    
    # Reverse normalization for yData
    original_yData = scaler_y.inverse_transform(yData)
    
    return original_xData, original_yData

# Constants
Init_Cap_B0005 = 1.8565
Init_Cap_B0006 = 2.0353
Init_Cap_B0007 = 1.8911
Init_Cap_B0018 = 1.8550

# Apply normalization
xData_B0005, yData_B0005, scaler_x_B0005, scaler_y_charge_mm_B0005 = minmax_norm(processed_VIT_B0005, Init_Cap_B0005, capacities_B0005)
xData_B0006, yData_B0006, scaler_x_B0006, scaler_y_charge_mm_B0006 = minmax_norm(processed_VIT_B0006, Init_Cap_B0006, capacities_B0006)
xData_B0007, yData_B0007, scaler_x_B0007, scaler_y_charge_mm_B0007 = minmax_norm(processed_VIT_B0007, Init_Cap_B0007, capacities_B0007)
#xData_B0018, yData_B0018, scaler_x_B0018, scaler_y_B0018 = minmax_norm(processed_VIT_B0018, Init_Cap_B0018, capacities_B0018)

xData_dis_B0005, yData_dis_B0005, scaler_x_dis_B0005, scaler_y_dis_mm_B0005 = minmax_norm(processed_disVIT_B0005, Init_Cap_B0005, capacities_B0005)
xData_dis_B0006, yData_dis_B0006, scaler_x_dis_B0006, scaler_y_dis_mm_B0006 = minmax_norm(processed_disVIT_B0006, Init_Cap_B0006, capacities_B0006)
xData_dis_B0007, yData_dis_B0007, scaler_x_dis_B0007, scaler_y_dis_mm_B0007 = minmax_norm(processed_disVIT_B0007, Init_Cap_B0007, capacities_B0007)
#xData_dis_B0018, yData_dis_B0018, scaler_x_dis_B0018, scaler_y_dis_B0018 = minmax_norm(processed_disVIT_B0018, Init_Cap_B0018, capacities_B0018)

xData_comb_B0005, yData_comb_B0005, scaler_x_comb_B0005, scaler_y_comb_mm_B0005 = minmax_norm(combined_VIT_B0005, Init_Cap_B0005, capacities_B0005)
xData_comb_B0006, yData_comb_B0006, scaler_x_comb_B0006, scaler_y_comb_mm_B0006 = minmax_norm(combined_VIT_B0006, Init_Cap_B0006, capacities_B0006)
xData_comb_B0007, yData_comb_B0007, scaler_x_comb_B0007, scaler_y_comb_mm_B0007 = minmax_norm(combined_VIT_B0007, Init_Cap_B0007, capacities_B0007)
#xData_comb_B0018, yData_comb_B0018, scaler_x_comb_B0018, scaler_y_comb_B0018 = minmax_norm(combined_VIT_B0018, Init_Cap_B0018, capacities_B0018)



def standard_scaler(charInput, cap):
    """
    Standardizes the given data using StandardScaler from scikit-learn.
    
    Parameters:
    charInput (np.ndarray): The feature data to be standardized.
    cap (np.ndarray): The capacity data to be standardized.
    
    Returns:
    np.ndarray: Standardized feature data.
    np.ndarray: Standardized capacity data.
    StandardScaler: Scaler object for feature data.
    StandardScaler: Scaler object for capacity data.
    """
    # Initialize the scalers
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    # Standardize charInput
    standardized_charInput = scaler_x.fit_transform(charInput)
    
    # Ensure cap is 1-dimensional
    cap = cap.flatten()
    
    # Standardize yData
    standardized_yData = scaler_y.fit_transform(cap.reshape(-1, 1))
    
    return standardized_charInput, standardized_yData, scaler_x, scaler_y

def inverse_standard_scaler(standardized_charInput, standardized_yData, scaler_x, scaler_y):
    """
    Converts standardized data back to the original scale using the provided StandardScaler objects.
    
    Parameters:
    standardized_charInput (np.ndarray): The standardized feature data.
    standardized_yData (np.ndarray): The standardized capacity data.
    scaler_x (StandardScaler): Scaler object used to standardize feature data.
    scaler_y (StandardScaler): Scaler object used to standardize capacity data.
    
    Returns:
    np.ndarray: Original feature data.
    np.ndarray: Original capacity data.
    """
    # Reverse standardization for charInput
    original_charInput = scaler_x.inverse_transform(standardized_charInput)
    
    # Reverse standardization for yData
    original_yData = scaler_y.inverse_transform(standardized_yData)
    
    return original_charInput, original_yData

# Constants --> no need
Init_Cap_B0005 = 1.8565
Init_Cap_B0006 = 2.0353
Init_Cap_B0007 = 1.8911
Init_Cap_B0018 = 1.8550

# Standardize the data
xData_std_B0005, yData_std_B0005, scaler_x_B0005, scaler_y_charge_std_B0005 = standard_scaler(processed_VIT_B0005, capacities_B0005)
xData_std_B0006, yData_std_B0006, scaler_x_B0006, scaler_y_charge_std_B0006 = standard_scaler(processed_VIT_B0006, capacities_B0006)
xData_std_B0007, yData_std_B0007, scaler_x_B0007, scaler_y_charge_std_B0007 = standard_scaler(processed_VIT_B0007, capacities_B0007)
#xData_std_B0018, yData_std_B0018, scaler_x_B0018, scaler_y_B0018 = standard_scaler(processed_VIT_B0018, capacities_B0018)

xData_dis_std_B0005, yData_dis_std_B0005, scaler_x_dis_B0005, scaler_y_dis_std_B0005 = standard_scaler(processed_disVIT_B0005, capacities_B0005)
xData_dis_std_B0006, yData_dis_std_B0006, scaler_x_dis_B0006, scaler_y_dis_std_B0006 = standard_scaler(processed_disVIT_B0006, capacities_B0006)
xData_dis_std_B0007, yData_dis_std_B0007, scaler_x_dis_B0007, scaler_y_dis_std_B0007 = standard_scaler(processed_disVIT_B0007, capacities_B0007)
#xData_dis_std_B0018, yData_dis_std_B0018, scaler_x_dis_B0018, scaler_y_dis_B0018 = standard_scaler(processed_disVIT_B0018, capacities_B0005)

xData_comb_std_B0005, yData_comb_std_B0005, scaler_x_comb_B0005, scaler_y_comb_std_B0005 = standard_scaler(combined_VIT_B0005, capacities_B0005)
xData_comb_std_B0006, yData_comb_std_B0006, scaler_x_comb_B0006, scaler_y_comb_std_B0006 = standard_scaler(combined_VIT_B0006, capacities_B0006)
xData_comb_std_B0007, yData_comb_std_B0007, scaler_x_comb_B0007, scaler_y_comb_std_B0007 = standard_scaler(combined_VIT_B0007, capacities_B0007)
#xData_comb_std_B0018, yData_comb_std_B0018, scaler_x_comb_B0018, scaler_y_comb_B0018 = standard_scaler(combined_VIT_B0018, capacities_B0018)

