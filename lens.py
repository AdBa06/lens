import subprocess
import sys

steps = [
    ("Running Customer Analysis Pipeline...", [sys.executable, "customer_analysis/lens_customer.py"]),
    ("Running Error Analysis Pipeline...", [sys.executable, "error_analysis/lens_error.py"]),
    ("Starting dashboard (Ctrl+C to stop)...", [sys.executable, "enhanced_dashboard.py"])
]

for message, cmd in steps:
    print(f"\n=== {message} ===")
    try:
        result = subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Step failed: {e}")
        sys.exit(1)

print("\nAll steps completed.")
