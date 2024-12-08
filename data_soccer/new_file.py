import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import requests
from bs4 import BeautifulSoup
from subprocess import call  # Import to execute another Python file

url = "http://alab.ise.ous.ac.jp/robocupdata/rc2021-roundrobin/normal/alice2021-helios2021/"
save_dir = 'tracking_csv_files3'
os.makedirs(save_dir, exist_ok=True)

# Function to download data
def download_data(file_name):
    file_url = url + file_name
    file_path = os.path.join(save_dir, file_name)
    with requests.get(file_url, stream=True) as file_response:
        with open(file_path, 'wb') as file:
            for chunk in file_response.iter_content(chunk_size=8192):
                file.write(chunk)
    print(f"Downloaded {file_name}")

# Function to add 'sim' column to each CSV file
def add_sim_column(file_path):
    file_name = os.path.basename(file_path)
    sim_value = file_name.split('-')[-1].split('.')[0]  # Extracting sim value
    df = pd.read_csv(file_path)
    df['sim'] = sim_value  # Add new column
    df.to_csv(file_path, index=False)  # Save back
    print(f"Updated {file_name} with 'sim' column")

# Downloading files
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
with ThreadPoolExecutor() as executor:
    for link in soup.find_all('a', href=True):
        file_name = link['href']
        if file_name.endswith("tracking.csv"):
            executor.submit(download_data, file_name)

# Adding 'sim' column to each downloaded file
for file_name in os.listdir(save_dir):
    if file_name.endswith("tracking.csv"):
        add_sim_column(os.path.join(save_dir, file_name))

# Call the script to merge and convert to pickle
call(["python", "merge_and_convert.py"])