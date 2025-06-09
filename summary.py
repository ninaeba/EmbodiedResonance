import os
import json
import pandas as pd

# Path to the folder with downloaded .json files
folder_path = "/Users/admin/Documents/simulation/download data/downloaded_disease_files"

# A list to collect dictionaries for each subject
rows = []

# Loop over files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith("_disease.json"):
        filepath = os.path.join(folder_path, filename)
        with open(filepath, 'r') as f:
            content = json.load(f)

        # Extract subject ID from the filename
        subject_id = filename.replace("_disease.json", "")

        # Add subject ID as a new field
        content["subject_id"] = subject_id

        # Append the record to the list
        rows.append(content)

# Create a DataFrame from all collected records
df = pd.DataFrame(rows)

# Save the table to CSV
df.to_csv("patients_disease_table.csv", index=False)

print("✅ Table saved as 'patients_disease_table.csv'")
