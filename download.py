import os, argparse, time, requests

from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup

import pandas as pd

# from subprocess import call  # Import to execute another Python file


# Function to download data
# def download_data(file_name, file_url):
#     file_path = os.path.join(save_dir, file_name)
#     with requests.get(file_url, stream=True) as file_response:
#         with open(file_path, "wb") as file:
#             for chunk in file_response.iter_content(chunk_size=8192):
#                 file.write(chunk)
#     print(f"Downloaded {file_name}")


# Ensure matching tracking and event files
def match_used_files(urls):
    file_types = ["tracking", "event", "player_types"]
    file_ids = {file_type: {} for file_type in file_types}
    for file_type in file_types:
        files = [url for url in urls if url.endswith(f"{file_type}.csv")]
        for file in files:
            splits = file.split("-")
            id = (
                splits[0]
                + "-"
                + splits[1]
                + "-"
                + splits[-2]
                + "-"
                + splits[-1].split(".")[0]
            )
            file_ids[file_type][id] = file

    if any(
        len(file_ids[file_type]) != len(file_ids[file_type]) for file_type in file_types
    ):
        raise ValueError(
            "Number of tracking, event, and player_types files do not match"
        )

    matched_pairs = []
    for identifier in file_ids["tracking"]:
        if all(identifier in file_ids[file_type] for file_type in file_types):
            matched_pairs.append(
                (
                    file_ids["tracking"][identifier],
                    file_ids["event"][identifier],
                    file_ids["player_types"][identifier],
                )
            )
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
#         for url in soup.find_all('a', href=True):
#             if debug and i == 5:
#                 break
#             file_name = url['href']
#             if file_name.endswith("tracking.csv"):
#                 executor.submit(download_data, file_name) # download_data called as a closure
#                 i += 1


# Updated function to save files in respective directories
def download_data(file_name, file_type, dirs):
    file_url = url + file_name
    if file_type == "tracking":
        file_path = os.path.join(dirs["tracking"], file_name)
    elif file_type == "event":
        file_path = os.path.join(dirs["event"], file_name)
    elif file_type == "player_types":
        file_path = os.path.join(dirs["player_types"], file_name)

    with requests.get(file_url, stream=True) as file_response:
        with open(file_path, "wb") as file:
            for chunk in file_response.iter_content(chunk_size=8192):
                file.write(chunk)
    print(f"Downloaded {file_name} to {file_path}")


def get_target_urls(base_url, search_str):
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, "html.parser")
    return [
        base_url + os.sep + url["href"]
        for url in soup.find_all("a", href=True)
        if search_str in url["href"]
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_url",
        type=str,
        default="http://alab.ise.ous.ac.jp/robocupdata/",
        help="Base URL for downloading data",
    )
    # parser.add_argument(
    #     "--subpaths",
    #     type=str,
    #     nargs="+",
    #     required=True,
    #     help="List of subpaths to download data from",
    # )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="robocup2d_data",
        help="Directory to save downloaded files",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--option", type=str, default=None)
    parser.add_argument("--Challenge", action="store_true")
    args, _ = parser.parse_known_args()

    # url = "http://alab.ise.ous.ac.jp/robocupdata/rc2021-roundrobin/normal/alice2021-helios2021/"
    # urls = [args.base_url + subpath + os.sep for subpath in args.subpaths]
    if args.Challenge:
        save_dir = args.save_dir + "/Challenge"
    else:
        save_dir = args.save_dir
    os.makedirs(args.save_dir, exist_ok=True)

    # Create separate directories for tracking and event files
    dirs = {
        "tracking": os.path.join(save_dir, "tracking"),
        "event": os.path.join(save_dir, "event"),
        "player_types": os.path.join(save_dir, "player_types"),
    }
    os.makedirs(os.path.join(save_dir, "tracking"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "event"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "player_types"), exist_ok=True)

    start_time = time.time()

    base_url = (
        args.base_url + "rc2024-roundrobin"
        if args.Challenge
        else args.base_url + "rc2021-roundrobin/normal"
    )
    search_str = "helios2024" if args.Challenge else ""
    target_urls = get_target_urls(base_url, search_str)

    print("Got target urls : ", time.time() - start_time)

    # Updated main logic for downloading files
    for url in target_urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        i = 0

        with ThreadPoolExecutor() as executor:
            urls = [url["href"] for url in soup.find_all("a", href=True)]
            matched_pairs = match_used_files(urls)

            for tracking_file, event_file, player_types in matched_pairs:
                if args.debug and i == 1:
                    break
                executor.submit(download_data, tracking_file, "tracking", dirs)
                executor.submit(download_data, event_file, "event", dirs)
                executor.submit(download_data, player_types, "player_types", dirs)
                i += 1

        print("------")
        print(f"Download data on {url} : ", time.time() - start_time)
        print("------")

        if args.debug:
            break

    print("-----------------")
    print("Finished downloading data!!")


# preprocess.py iterable list, stored_data as an input for preprocess.py # sim update
