"""
Demo script: ecosound.datasets
- List available datasets
- Download the minke whale dataset (annotations only)
- Load and explore annotations
"""

import ecosound.datasets

# ---- 1. List available datasets -----------------------------------------
print("=== Available datasets ===")
ecosound.datasets.list_datasets()

# ---- 2. Set cache directory (run once, persists across sessions) ----------
ecosound.datasets.init("D:/my_datasets")   # uncomment to change location

# ---- 3. Download and load annotations ------------------------------------
print("\n=== Downloading and loading annotations ===")
annots = ecosound.datasets.load("minke-whale-mouy-2026-TEST")
# ---- 4. Explore annotations ----------------------------------------------
print(f"\nTotal annotations : {len(annots.data)}")
print(f"Deployments       : {annots.data['deployment_ID'].nunique()}")
print(f"Label classes     : {sorted(annots.data['label_class'].unique())}")
print(f"\nLabel distribution:")
print(annots.data["label_class"].value_counts().to_string())
print(f"\nFirst 5 rows:")
print(annots.data[["deployment_ID", "audio_file_name", "label_class", "time_min_date", "duration"]].head().to_string())
