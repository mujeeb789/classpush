# import os
# import requests
# from bs4 import BeautifulSoup
# from urllib.parse import urljoin
# from tqdm import tqdm


# BASE_URL = "https://gisco-services.ec.europa.eu/lucas/photos/2018/"
# COUNTRIES = ['FR']  # Add more like 'FR', 'DE', etc.
# DOWNLOAD_DIR = "lucas_images/"

# visited_urls = set()

# def download_file(url, path):
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     r = requests.get(url, stream=True)
#     if r.status_code == 200:
#         with open(path, 'wb') as f:
#             for chunk in r.iter_content(1024):
#                 f.write(chunk)

# def is_image_link(href):
#     return href.lower().endswith(('.jpg', '.jpeg'))

# def scrape_recursive(url, local_root):
#     if url in visited_urls:
#         return
#     visited_urls.add(url)

#     try:
#         r = requests.get(url)
#         r.raise_for_status()
#         soup = BeautifulSoup(r.text, 'html.parser')
#         links = soup.find_all('a')

#         for link in links:
#             href = link.get('href')
#             if not href or href == "../":
#                 continue
#             full_url = urljoin(url, href)
#             if is_image_link(href):
#                 local_path = os.path.join(local_root, href)
#                 if not os.path.exists(local_path):  # Avoid re-downloading
#                     tqdm.write(f"⬇️ Downloading: {full_url}")
#                     download_file(full_url, local_path)
#             elif href.endswith('/'):
#                 # It's a subdirectory
#                 subfolder = os.path.join(local_root, href.strip('/'))
#                 scrape_recursive(full_url, subfolder)
#     except Exception as e:
#         tqdm.write(f"❌ Failed to access {url}: {e}")

# if __name__ == "__main__":
#     for country in COUNTRIES:
#         start_url = urljoin(BASE_URL, f"{country}/")
#         local_dir = os.path.join(DOWNLOAD_DIR, country)
#         tqdm.write(f"📂 Scraping images from: {start_url}")
#         scrape_recursive(start_url, local_dir)
#     print("✅ Done downloading all images.")
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tqdm import tqdm
import time

BASE_URL = "https://gisco-services.ec.europa.eu/lucas/photos/2018/"
COUNTRIES = ['AT']  # You can add more countries here
DOWNLOAD_DIR = "lucas_images"
visited_urls = set()

def download_file(url, path, max_retries=3):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Skip if already downloaded
    if os.path.exists(path):
        return

    for attempt in range(max_retries):
        try:
            r = requests.get(url, stream=True, timeout=10)
            if r.status_code == 200:
                with open(path, 'wb') as f:
                    for chunk in r.iter_content(1024):
                        f.write(chunk)
                return
            else:
                raise Exception(f"Status code: {r.status_code}")
        except Exception as e:
            tqdm.write(f"Retry {attempt+1}/{max_retries} for {url} due to: {e}")
            time.sleep(2)

    tqdm.write(f"❌ Failed after {max_retries} attempts: {url}")

def is_image_link(href):
    return href.lower().endswith(('.jpg', '.jpeg'))

def scrape_recursive(url, local_root):
    if url in visited_urls:
        return
    visited_urls.add(url)

    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')
        links = soup.find_all('a')

        for link in links:
            href = link.get('href')
            if not href or href == "../":
                continue

            full_url = urljoin(url, href)
            if is_image_link(href):
                local_path = os.path.join(local_root, href)
                tqdm.write(f"⬇️ Downloading: {full_url}")
                download_file(full_url, local_path)
            elif href.endswith('/'):
                subfolder = os.path.join(local_root, href.strip('/'))
                scrape_recursive(full_url, subfolder)
    except Exception as e:
        tqdm.write(f"❌ Failed to access {url}: {e}")
        time.sleep(2)  # Wait before retrying the parent

if __name__ == "__main__":
    for country in COUNTRIES:
        start_url = urljoin(BASE_URL, f"{country}/")
        local_dir = os.path.join(DOWNLOAD_DIR, country)
        tqdm.write(f"📂 Resuming scraping for: {start_url}")
        scrape_recursive(start_url, local_dir)

    print("✅ All available images downloaded or resumed.")
