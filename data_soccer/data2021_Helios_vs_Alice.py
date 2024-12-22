# import os
# import requests
# from bs4 import BeautifulSoup4
# from concurrent.futures import ThreadPoolExecutor

# # Base URL for downloading files
# url = "http://alab.ise.ous.ac.jp/robocupdata/rc2021-roundrobin/normal/alice2021-helios2021/"

# # Define directories for saving tracking and event files
# tracking_save_dir = 'Tracking_data'
# event_save_dir = 'Event_data'

# # Check if directories exist, if not create them
# if not os.path.exists(tracking_save_dir):
#     os.makedirs(tracking_save_dir)
#     print(f"Created directory: {tracking_save_dir}")

# if not os.path.exists(event_save_dir):
#     os.makedirs(event_save_dir)
#     print(f"Created directory: {event_save_dir}")

# def download_data(file_name, file_type):
#     """
#     Downloads a file from the specified URL and saves it to the appropriate directory.
    
#     Parameters:
#     - file_name: str, the name of the file to download.
#     - file_type: str, the type of file ('tracking' or 'event').
#     """
#     file_url = url + file_name
#     # Choose the directory based on the file type
#     save_dir = tracking_save_dir if file_type == 'tracking' else event_save_dir
#     file_path = os.path.join(save_dir, file_name)

#     # Download the file
#     with requests.get(file_url, stream=True) as file_response:
#         with open(file_path, 'wb') as file:
#             for chunk in file_response.iter_content(chunk_size=8192):
#                 file.write(chunk)
#     print(f"Downloaded {file_name} to {save_dir}")

# # Fetch the webpage and parse links
# response = requests.get(url)
# soup = BeautifulSoup4(response.text, 'html.parser')

# # Use ThreadPoolExecutor to download files
# with ThreadPoolExecutor() as executor:
#     for link in soup.find_all('a', href=True):
#         file_name = link['href']
#         if file_name.endswith("tracking.csv"):
#             executor.submit(download_data, file_name, 'tracking')
#         elif file_name.endswith("event.csv"):  # Assuming event files have the extension "event.csv"
#             executor.submit(download_data, file_name, 'event')

import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import requests
from bs4 import BeautifulSoup

# Base URL for downloading files
url = "http://alab.ise.ous.ac.jp/robocupdata/rc2021-roundrobin/normal/alice2021-helios2021/"

# Create directories for saving tracking and event files
tracking_save_dir = 'tracking_csv_files'
event_save_dir = 'event_csv_files'
os.makedirs(tracking_save_dir, exist_ok=True)
os.makedirs(event_save_dir, exist_ok=True)

# Function to download data
def download_data(file_name, file_type):
    file_url = url + file_name
    # Choose the directory based on the file type
    save_dir = tracking_save_dir if file_type == 'tracking' else event_save_dir
    file_path = os.path.join(save_dir, file_name)
    with requests.get(file_url, stream=True) as file_response:
        file_response.raise_for_status()  # Raise an error for bad responses
        with open(file_path, 'wb') as file:
            for chunk in file_response.iter_content(chunk_size=8192):
                file.write(chunk)
    print(f"Downloaded {file_name} to {save_dir}")

# Function to add 'id' column to each CSV file
def add_sim_column(file_path):
    file_name = os.path.basename(file_path)
    sim_value = file_name.split('-')[-1].split('.')[0]  # Extracting sim value
    df = pd.read_csv(file_path)
    df['Match_ID'] = sim_value  # Add new column
    df.to_csv(file_path, index=False)  # Save back
    print(f"Updated {file_name} with 'Match_ID' column")

# Downloading files
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

with ThreadPoolExecutor() as executor:
    for link in soup.find_all('a', href=True):
        file_name = link['href']
        if file_name.endswith("tracking.csv"):
            executor.submit(download_data, file_name, 'tracking')
        elif file_name.endswith("event.csv"):  # Assuming event files have the extension "event.csv"
            executor.submit(download_data, file_name, 'event')

# Adding 'sim' column to each downloaded tracking file
for file_name in os.listdir(tracking_save_dir):
    if file_name.endswith("tracking.csv"):
        add_sim_column(os.path.join(tracking_save_dir, file_name))

# Adding 'sim' column to each downloaded event file (if needed)
for file_name in os.listdir(event_save_dir):
    if file_name.endswith("event.csv"):
        add_sim_column(os.path.join(event_save_dir, file_name))