import subprocess
import sys

steps = [
    ("Ingesting Excel data...", [sys.executable, "excel_import.py", "import", "synthetic_data.xlsx"]),
    ("Generating embeddings...", [sys.executable, "run_embeddings.py", "--regenerate"]),
    ("Clustering embeddings...", [sys.executable, "clustering.py", "--regenerate"]),
    ("Generating cluster summaries...", [sys.executable, "regenerate_all_clusters_direct.py"]),
    ("Starting dashboard (Ctrl+C to stop)...", [sys.executable, "enhanced_dashboard.py"])
]

for message, cmd in steps:
    print(f"\n=== {message} ===")
    try:
        result = subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Step failed: {e}")
        sys.exit(1)

print("\nAll steps completed. Dashboard is running.")
