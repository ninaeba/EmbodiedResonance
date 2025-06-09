import os
import re
import requests

# Load the list of all file links
with open("alllinks.txt", "r") as f:
    all_links = f.readlines()

# Define subject groups by diagnostic category
groups = {
    "healthy": ["112", "114", "115", "119"],
    "mental":  ["091", "094", "095", "100"],
    "ptsd":    ["064", "044", "075", "054"],
    "sick":    ["031", "005", "007", "016"]
}

# File patterns to download: ECG recordings from three phases and one metadata JSON
ecg_patterns = [
    "baseline-data_rest-ecg-500hz.csv",
    "experimental-data_recover-ecg-500hz.csv",
    "experimental-data_stress-ecg-500hz.csv"
]
json_pattern = "baseline-data_chief-complaint.json"

# Create output folders for each group
for group in groups:
    os.makedirs(group, exist_ok=True)

# Iterate over all links and download matching files
for line in all_links:
    line = line.strip()
    for group, subs in groups.items():
        for sub in subs:
            if f"sub_{sub}_" in line:
                for pattern in ecg_patterns + [json_pattern]:
                    if pattern in line:
                        try:
                            # Extract the filename from the URL
                            filename = re.search(r"fileName=([^&]+)", line).group(1)
                            out_path = os.path.join(group, filename)
                            print(f"⬇️  Downloading {filename} into folder {group}...")
                            
                            # Perform the download
                            r = requests.get(line)
                            r.raise_for_status()
                            
                            # Save the file to the designated folder
                            with open(out_path, "wb") as f_out:
                                f_out.write(r.content)
                            print(f"✅ Done: {filename}")
                        except Exception as e:
                            print(f"❌ Error downloading from {line}: {e}")
