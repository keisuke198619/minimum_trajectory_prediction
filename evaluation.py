import glob, os
import argparse

import numpy as np 
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--n_roles', type=int, default=23)
parser.add_argument('--burn_in', type=int, default=20)
parser.add_argument('--submit', type=str, required=True)
parser.add_argument('--gt', type=str, required=True)
args, _ = parser.parse_known_args()

def get_csv_files(folder):
    """
    Retrieve a sorted list of CSV file paths from the specified folder.
    """
    try: csv_files = sorted(glob.glob(os.path.join(folder, '*.csv')))
    except: import pdb; pdb.set_trace()
    return csv_files

def compute_errors(submit_df, gt_df, n_roles, burn_in):
    """
    Compute average and endpoint errors between submission and ground truth dataframes.
    
    Parameters:
        submit_df (pd.DataFrame): Submission dataframe.
        gt_df (pd.DataFrame): Ground truth dataframe.
        n_roles (int): Number of agents.
        burn_in (int): Number of initial time steps to exclude.
        
    Returns:
        tuple: (average_error, endpoint_error)
    """
    errors = []
    endpoint_errors = []
    
    # Iterate over each agent
    for agent in range(n_roles):
        x_pred = submit_df[f'agent_{agent}_x'].values
        y_pred = submit_df[f'agent_{agent}_y'].values
        x_gt = gt_df[f'agent_{agent}_x'].values
        y_gt = gt_df[f'agent_{agent}_y'].values
        
        # Exclude burn-in steps
        x_pred_burn = x_pred[burn_in:]
        y_pred_burn = y_pred[burn_in:]
        x_gt_burn = x_gt[burn_in:]
        y_gt_burn = y_gt[burn_in:]
        
        # Compute Euclidean distances for average error
        distances = np.sqrt((x_pred_burn - x_gt_burn) ** 2 + (y_pred_burn - y_gt_burn) ** 2)
        errors.append(distances)
        
        # Compute endpoint error (last time step)
        end_distance = np.sqrt((x_pred[-1] - x_gt[-1]) ** 2 + (y_pred[-1] - y_gt[-1]) ** 2)
        endpoint_errors.append(end_distance)
    
    # Concatenate all errors and compute mean
    all_errors = np.concatenate(errors)
    average_error = np.mean(all_errors)
    
    # Compute mean endpoint error
    mean_endpoint_error = np.mean(endpoint_errors)
    
    return average_error, mean_endpoint_error

def main():
    submit_folder = args.submit
    gt_folder = args.gt
    n_roles = args.n_roles
    burn_in = args.burn_in
    
    # Retrieve CSV files from both submission and ground truth folders
    submit_files = get_csv_files(submit_folder)
    gt_files = get_csv_files(gt_folder)
    
    # Extract file names for matching
    submit_filenames = [os.path.basename(f) for f in submit_files]
    gt_filenames = [os.path.basename(f) for f in gt_files]
    
    # Find common files
    common_files = set(submit_filenames).intersection(set(gt_filenames))
    
    if not common_files:
        print("No matching CSV files found between submission and ground truth folders.")
        import pdb; pdb.set_trace()
        return
    
    # Initialize lists to store all errors
    all_average_errors = []
    all_endpoint_errors = []
    
    # Iterate over each common file
    for filename in sorted(common_files):
        submit_path = os.path.join(submit_folder, filename)
        gt_path = os.path.join(gt_folder, filename)
        
        # Read CSV files
        try:
            submit_df = pd.read_csv(submit_path, index_col='time')
            gt_df = pd.read_csv(gt_path, index_col='time')
        except Exception as e:
            print(f"Error reading files {filename}: {e}")
            continue
        
        # Check if required columns exist
        expected_columns = []
        for agent in range(n_roles):
            expected_columns.extend([f'agent_{agent}_x', f'agent_{agent}_y'])
        
        if not all(col in submit_df.columns for col in expected_columns):
            print(f"Submission file {filename} is missing required columns.")
            continue
        if not all(col in gt_df.columns for col in expected_columns):
            print(f"Ground truth file {filename} is missing required columns.")
            continue
        
        # Compute errors for the current file
        avg_error, end_error = compute_errors(submit_df, gt_df, n_roles, burn_in)
        all_average_errors.append(avg_error)
        all_endpoint_errors.append(end_error)
    
    if not all_average_errors:
        print("No errors were computed. Please check the input files.")
        return
    
    # Compute overall metrics
    overall_average_error = np.mean(all_average_errors)
    overall_endpoint_error = np.mean(all_endpoint_errors)
    
    # Print evaluation results
    print("Trajectory Prediction Evaluation Results:")
    print(f"Number of Agents: {n_roles}")
    print(f"Burn-in Steps: {burn_in}")
    print(f"Number of Evaluated Sequences: {len(all_average_errors)}")
    print(f"Average Error (after burn-in): {overall_average_error:.4f}")
    print(f"Endpoint Error: {overall_endpoint_error:.4f}")

if __name__ == "__main__":
    main()