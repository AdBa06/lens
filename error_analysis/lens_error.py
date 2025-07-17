import subprocess
import sys

steps = [
    ("Ingesting error Excel data...", [sys.executable, "excel_import.py", "import", "errors.xlsx"]),
    ("Generating embeddings...", [sys.executable, "run_embeddings.py", "--regenerate"]),
    ("Clustering embeddings...", [sys.executable, "clustering.py", "--regenerate"]),
    ("Generating error cluster root cause and fixes...", [sys.executable, "error_analysis/regenerate_error_clusters.py"]),
    # ("Starting dashboard (Ctrl+C to stop)...", [sys.executable, "enhanced_dashboard.py"])
]

for message, cmd in steps:
    print(f"\n=== {message} ===")
    try:
        result = subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Step failed: {e}")
        sys.exit(1)

print("\Error Analysis Done. Dashboard is running.")
