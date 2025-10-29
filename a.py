# import os
# import csv
# import requests
# from urllib.parse import urljoin
# from tqdm import tqdm
# import time

# # Base config
# BASE_URL = "https://gisco-services.ec.europa.eu/lucas/photos/2018/"
# DOWNLOAD_DIR = "newlucas_images"
# TIMEOUT = 15
# RETRIES = 3

# # Load CSV
# def load_pointid_map(csv_path):
#     pointid_map = {}
#     with open(csv_path, mode='r', encoding='utf-8') as file:
#         reader = csv.DictReader(file)
#         for row in reader:
#             pid = row['POINTID'].strip()
#             nut0 = row['NUTS_0'].strip().upper()
#             if pid and nut0:
#                 pointid_map[pid] = nut0
#     return pointid_map

# # Download image with retry
# def download_image(url, path):
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     if os.path.exists(path):
#         return True

#     for attempt in range(RETRIES):
#         try:
#             r = requests.get(url, stream=True, timeout=TIMEOUT)
#             if r.status_code == 200:
#                 with open(path, 'wb') as f:
#                     for chunk in r.iter_content(1024):
#                         f.write(chunk)
#                 return True
#             else:
#                 raise Exception(f"Status code: {r.status_code}")
#         except Exception as e:
#             tqdm.write(f"Retry {attempt+1} for {url}: {e}")
#             time.sleep(2)

#     tqdm.write(f"❌ Failed to download: {url}")
#     return False

# # Download all matched images
# def process_all_images(pointid_map):
#     session = requests.Session()
#     total = len(pointid_map)
#     downloaded = 0

#     for pointid, nut0 in tqdm(pointid_map.items(), desc="Processing images"):
#         sub1 = pointid[:3]
#         sub2 = pointid[3:6]
#         folder_url = urljoin(BASE_URL, f"{nut0}/{sub1}/{sub2}/")

#         # Try variants with suffixes: E, N, S, W, etc.
#         suffixes = ['E', 'N', 'S', 'W','P']
#         for suffix in suffixes:
#             filename = f"{pointid}{suffix}.jpg"
#             full_url = urljoin(folder_url, filename)
#             local_path = os.path.join(DOWNLOAD_DIR, nut0, sub1, sub2, filename)

#             if download_image(full_url, local_path):
#                 tqdm.write(f"✅ Downloaded: {filename}")
#                 downloaded += 1
#                 break  # Stop after first successful variant

#     print(f"\n✅ Done. Downloaded {downloaded}/{total} matching images.")

# # === Main ===
# if __name__ == "__main__":
#     CSV_PATH = "LUCASSOIL2018.csv"  # Replace with your path

#     if not os.path.exists(CSV_PATH):
#         print(f"❌ CSV not found: {CSV_PATH}")
#     else:
#         pointid_map = load_pointid_map(CSV_PATH)
#         print(f"📊 Loaded {len(pointid_map)} point IDs from CSV")
#         process_all_images(pointid_map)


import os
import csv
import requests
from urllib.parse import urljoin
from tqdm import tqdm
import time

# Configuration
BASE_URL = "https://gisco-services.ec.europa.eu/lucas/photos/2018/"
DOWNLOAD_DIR = "newlucas_images"
TIMEOUT = 15
RETRIES = 3
SUFFIXES = ['E', 'N', 'S', 'W', 'P']  # All variants to check

# Load CSV
def load_pointid_map(csv_path):
    pointid_map = {}
    with open(csv_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            pid = row['POINTID'].strip()
            nut0 = row['NUTS_0'].strip().upper()
            if pid and nut0:
                pointid_map[pid] = nut0
    return pointid_map

# Download image with retries
def download_image(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        return False  # Already exists, no need to download

    for attempt in range(RETRIES):
        try:
            r = requests.get(url, stream=True, timeout=TIMEOUT)
            if r.status_code == 200:
                with open(path, 'wb') as f:
                    for chunk in r.iter_content(1024):
                        f.write(chunk)
                return True
        except Exception as e:
            tqdm.write(f"⚠️ Retry {attempt+1} for {url}: {e}")
            time.sleep(2)

    tqdm.write(f"❌ Failed to download: {url}")
    return False

# Download all image variants per POINTID
def process_all_images(pointid_map):
    total_pids = len(pointid_map)
    downloaded_images = 0
    skipped_images = 0

    for pointid, nut0 in tqdm(pointid_map.items(), desc="Processing POINTIDs"):
        sub1 = pointid[:3]
        sub2 = pointid[3:6]
        folder_url = urljoin(BASE_URL, f"{nut0}/{sub1}/{sub2}/")

        for suffix in SUFFIXES:
            filename = f"{pointid}{suffix}.jpg"
            file_url = urljoin(folder_url, filename)
            local_path = os.path.join(DOWNLOAD_DIR, nut0, sub1, sub2, filename)

            result = download_image(file_url, local_path)
            if result:
                tqdm.write(f"✅ Downloaded: {filename}")
                downloaded_images += 1
            else:
                skipped_images += 1  # Either already exists or failed

    print(f"\n✅ Completed download process.")
    print(f"📥 Images downloaded: {downloaded_images}")
    print(f"⏭️ Images skipped or already existed: {skipped_images}")
    print(f"📦 Total POINTIDs processed: {total_pids}")

# === Entry Point ===
if __name__ == "__main__":
    CSV_PATH = "LUCASSOIL2018.csv"  # Update this path as needed

    if not os.path.exists(CSV_PATH):
        print(f"❌ CSV not found: {CSV_PATH}")
    else:
        pointid_map = load_pointid_map(CSV_PATH)
        print(f"📊 Loaded {len(pointid_map)} point IDs from CSV")
        process_all_images(pointid_map)
