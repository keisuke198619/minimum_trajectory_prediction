import os
import argparse
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import requests
from bs4 import BeautifulSoup
from subprocess import call  # Import to execute another Python file


parser = argparse.ArgumentParser()
parser.add_argument('--base_url', type=str, default='http://alab.ise.ous.ac.jp/robocupdata/', help='Base URL for downloading data')
parser.add_argument('--subpaths', type=str, nargs='+', required=True, help='List of subpaths to download data from')
parser.add_argument('--save_dir', type=str, default='robocup2d_data', help='Directory to save downloaded files')
parser.add_argument('--option', type=str, default=None)
parser.add_argument('--Challenge', action='store_true')
args, _ = parser.parse_known_args()

# url = "http://alab.ise.ous.ac.jp/robocupdata/rc2021-roundrobin/normal/alice2021-helios2021/"
urls = [args.base_url + os.sep + subpath + os.sep for subpath in args.subpaths]
save_dir = args.save_dir
os.makedirs(args.save_dir, exist_ok=True)
debug = True

# Function to download data
def download_data(file_name, file_url):
    file_path = os.path.join(save_dir, file_name)
    with requests.get(file_url, stream=True) as file_response:
        with open(file_path, 'wb') as file:
            for chunk in file_response.iter_content(chunk_size=8192):
                file.write(chunk)
    print(f"Downloaded {file_name}")

# Ensure matching tracking and event files
def match_tracking_and_event_files(links):
    tracking_files = [link for link in links if link.endswith("tracking.csv")]
    event_files = [link for link in links if link.endswith("event.csv")]

    # Extract unique identifiers (e.g., sim25)
    tracking_ids = {file.split('-')[-1].split('.')[0]: file for file in tracking_files}
    event_ids = {file.split('-')[-1].split('.')[0]: file for file in event_files}

    # Find common IDs
    matched_pairs = []
    for identifier in tracking_ids:
        if identifier in event_ids:
            matched_pairs.append((tracking_ids[identifier], event_ids[identifier]))
    return matched_pairs


# def extract_tracking_data(file_path):
#     file_name = os.path.basename(file_path)
#     df = pd.read_csv(file_path)
#     df = df[['time', 'sim', 'unum', 'x', 'y', 'vx', 'vy', 'body', 'neck', 'ball']]
#     df.to_csv(file_path, index=False)  # Save back
#     print(f"Extracted tracking data from {file_name}")

# # Downloading files
# for url in urls:
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, 'html.parser')
#     i = 0
#     with ThreadPoolExecutor() as executor:
#         for link in soup.find_all('a', href=True):
#             if debug and i == 5:
#                 break
#             file_name = link['href']
#             if file_name.endswith("tracking.csv"):
#                 executor.submit(download_data, file_name) # download_data called as a closure 
#                 i += 1

# Create separate directories for tracking and event files
tracking_dir = os.path.join(save_dir, "tracking")
event_dir = os.path.join(save_dir, "event")
os.makedirs(tracking_dir, exist_ok=True)
os.makedirs(event_dir, exist_ok=True)

# Updated function to save files in respective directories
def download_data(file_name, file_type):
    file_url = url + file_name
    if file_type == "tracking":
        file_path = os.path.join(tracking_dir, file_name)
    elif file_type == "event":
        file_path = os.path.join(event_dir, file_name)

    with requests.get(file_url, stream=True) as file_response:
        with open(file_path, 'wb') as file:
            for chunk in file_response.iter_content(chunk_size=8192):
                file.write(chunk)
    print(f"Downloaded {file_name} to {file_path}")

# Updated main logic for downloading files
for url in urls:
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    i = 0
    with ThreadPoolExecutor() as executor:
        links = [link['href'] for link in soup.find_all('a', href=True)]
        matched_pairs = match_tracking_and_event_files(links)
        
        for tracking_file, event_file in matched_pairs:
            if debug and i == 5:
                break
            executor.submit(download_data, tracking_file, "tracking")
            executor.submit(download_data, event_file, "event")
            i += 1

# preprocess.py iterable list, stored_data as an input for preprocess.py # sim update