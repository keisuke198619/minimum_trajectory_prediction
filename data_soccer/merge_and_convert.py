import os
import pandas as pd

def merge_csv_to_pickle(input_dir, output_pickle):
    # Initialize an empty list to store DataFrames
    dataframes = []
    
    # Iterate over all files in the input directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith("tracking.csv"):
            file_path = os.path.join(input_dir, file_name)
            df = pd.read_csv(file_path)  # Read each CSV file into a DataFrame
            dataframes.append(df)  # Append DataFrame to the list
    
    # Merge all DataFrames into a single DataFrame
    merged_df = pd.concat(dataframes, ignore_index=True)
    
    # Save the merged DataFrame as a pickle file
    merged_df.to_pickle(output_pickle)
    print(f"Successfully merged and saved as pickle: {output_pickle}")

# For standalone execution
if __name__ == "__main__":
    input_directory = "tracking_csv_files3"  # Directory containing CSV files
    output_pickle_file = "merged_data.pkl"  # Output pickle file name
    merge_csv_to_pickle(input_directory, output_pickle_file)
