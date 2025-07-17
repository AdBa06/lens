import subprocess
import sys

steps = [
    ("Ingesting customer Excel data...", [sys.executable, "excel_import.py", "import", "synthetic_data.xlsx"]),
    ("Generating embeddings...", [sys.executable, "run_embeddings.py", "--regenerate"]),
    ("Clustering embeddings...", [sys.executable, "clustering.py", "--regenerate"]),
    ("Generating cluster summaries...", [sys.executable, "regenerate_all_clusters_direct.py"]),

]

for message, cmd in steps:
    print(f"\n=== {message} ===")
    try:
        result = subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Step failed: {e}")
        sys.exit(1)

print("\nCustomer Analysis Done")
