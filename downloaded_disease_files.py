import os
import re
import requests

# Create a folder to store downloaded files
os.makedirs("downloaded_disease_files", exist_ok=True)

# Read the list of all file links
with open("all_links.txt", "r") as f:
    lines = f.readlines()

# Filter only those links that match the pattern 'sub_XXX_disease.json'
disease_links = [
    line.strip()
    for line in lines
    if re.search(r"sub_\d{3}_disease\.json", line)
]

print(f"Found {len(disease_links)} disease.json files. Starting download...")

# Loop over filtered links and download each JSON file
for url in disease_links:
    match = re.search(r"sub_(\d{3})_disease\.json", url)
    if not match:
        continue
    sub_id = match.group(1)
    filename = f"sub_{sub_id}_disease.json"
    filepath = os.path.join("downloaded_disease_files", filename)

    try:
        r = requests.get(url)
        r.raise_for_status()
        with open(filepath, "w", encoding="utf-8") as out_file:
            out_file.write(r.text)
        print(f"✅ Downloaded: {filename}")
    except Exception as e:
        print(f"❌ Error downloading {filename}: {e}")
