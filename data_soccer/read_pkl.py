# import pickle
# import numpy as np

# # Step 1: Load the data
# with open('train_data.pkl', 'rb') as file:
#     data = pickle.load(file)

# # Step 2: Inspect the data type
# print("Data Type:", type(data))

# # Step 3: Explore the content
# if isinstance(data, dict):
#     print("Number of Instances:", len(data))  # Count instances if it's a dictionary
#     print("Keys (Instance Identifiers):", list(data.keys())[:5])  # Show first 5 keys

#     # Check the shape of the first instance
#     first_instance = data[list(data.keys())[0]]
#     print("Shape of First Instance:", first_instance.shape)  # Shape of the first instance array

#     # Step 4: Visualize Sample Data
#     print("Sample Data from First Instance:\n", first_instance)

#     # If the instances are arrays, determine the number of objectives
#     num_objectives = first_instance.shape[1] if len(first_instance.shape) > 1 else 1
#     print("Number of Objectives:", num_objectives)

# elif isinstance(data, list):
#     print("Number of Instances:", len(data))  # Count instances if it's a list
#     print("Sample Data from First Instance:\n", data[0])  # Show first instance

#     # If the instances are arrays, determine the number of objectives
#     num_objectives = data[0].shape[1] if isinstance(data[0], np.ndarray) and len(data[0].shape) > 1 else 1
#     print("Number of Objectives:", num_objectives)

# else:
#     print("Unknown data structure. Please check the contents of the file.")
import pickle
import pandas as pd
import numpy as np

# Load the data from the pickle file
with open('train_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Check the shapes of the sequences
for key, value in data.items():
    print(f"{key}: {value.shape}")

# Standardize the data by padding with NaNs
max_length = max(value.shape[0] for value in data.values())
max_width = max(value.shape[1] for value in data.values())

for key in data.keys():
    if data[key].shape[0] < max_length:
        # Pad the array with NaNs
        padding = np.full((max_length - data[key].shape[0], max_width), np.nan)
        data[key] = np.vstack([data[key], padding])
    if data[key].shape[1] < max_width:
        # Pad the array with NaNs
        padding = np.full((max_length, max_width - data[key].shape[1]), np.nan)
        data[key] = np.hstack([data[key], padding])

# Create a list to store the DataFrames
dataframes = []

# Create a DataFrame for each sequence
for key, value in data.items():
    df = pd.DataFrame(value)
    df['sequence'] = key
    dataframes.append(df)

# Concatenate the DataFrames
df = pd.concat(dataframes, ignore_index=True)

print(df.head())